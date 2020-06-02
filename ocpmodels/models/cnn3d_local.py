import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ocpmodels.common.display import Display
from ocpmodels.common.preprocess import Preprocess
from ocpmodels.common.registry import registry
from ocpmodels.models.base import BaseModel
from ocpmodels.modules.layers import CGCNNConv


@registry.register_model("cnn3d_local")
class CNN3D_LOCAL(BaseModel):
    def __init__(
        self,
        num_atoms,
        bond_feat_dim,
        num_targets,
        regress_forces=False,  # compute the forces directly?
        grid_resolution=0.02,  # in angstroms
        gaussian_std=2.0,  # standard deviation in angstroms
        grid_size=15,
        freq_scalar=0.015,
        freq_base=1.9,
        max_atomic_number=90,
        num_input_filters=10,  # must be multiple of 2
        num_conv1_filters=32,
        conv1_kernal_size=5,
        num_conv2_filters=64,
        conv2_kernal_size=1,
        num_conv3_filters=64,
        conv3_kernal_size=3,
        num_conv4_filters=32,
        conv4_kernal_size=3,
        cell_batch_size=10000,  # Can go up to 20000 with 16GB RAM with grid_size=15
        show_timing_info=False,
        display_weights=True,
        display_base_name="",
        display_frequency=200,
    ):
        super(CNN3D_LOCAL, self).__init__(
            num_atoms, bond_feat_dim, num_targets
        )
        self.conv1 = nn.ModuleList(
            [
                nn.Conv3d(
                    num_input_filters, num_conv1_filters, conv1_kernal_size
                )
                for i in range(max_atomic_number)
            ]
        )

        self.conv2 = nn.Conv3d(
            num_conv1_filters, num_conv2_filters, conv2_kernal_size
        )

        self.conv3 = nn.Conv3d(
            num_conv2_filters, num_conv3_filters, conv3_kernal_size
        )
        self.conv4 = nn.Conv3d(
            num_conv3_filters, num_conv4_filters, conv4_kernal_size
        )
        self.conv4_forces = nn.Conv3d(
            num_conv3_filters, num_conv4_filters, conv4_kernal_size
        )
        self.fc1_energy = nn.Linear(num_conv4_filters, max_atomic_number)
        self.fc1_forces = nn.Linear(num_conv4_filters, max_atomic_number * 3)
        self.num_input_filters = num_input_filters
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution
        self.gaussian_std = gaussian_std
        self.freq_scalar = freq_scalar
        self.freq_base = freq_base
        self.max_atomic_number = max_atomic_number
        self.regress_forces = regress_forces
        self.num_conv1_filters = num_conv1_filters
        self.num_conv2_filters = num_conv2_filters
        self.num_conv4_filters = num_conv4_filters
        self.show_timing_info = show_timing_info
        self.show_timing_info = show_timing_info
        self.display_weights = display_weights
        self.display_frequency = display_frequency
        self.forward_counter = 0
        self.cell_batch_size = cell_batch_size
        self.display_base_name = display_base_name

    def forward(self, data):
        if (
            self.display_weights is True
            and (self.forward_counter % self.display_frequency) == 0
        ):
            Display.display_model_conv_weights(
                self.conv1[8].weight.data, 1, self.display_base_name
            )
            Display.display_model_conv_weights(
                self.conv2.weight.data, 2, self.display_base_name
            )
            Display.display_model_conv_weights(
                self.conv3.weight.data, 3, self.display_base_name
            )
            Display.display_model_conv_weights(
                self.conv4.weight.data, 4, self.display_base_name
            )

        self.forward_counter = self.forward_counter + 1

        start_time = time.time()
        device = data.x.device
        num_atoms = len(data.x)
        atomic_numbers = data.atomic_numbers.long()

        (
            cell_grids,
            channel_sum_index,
            channel_atomic_number,
            inv_rot,
        ) = self._create_grid_local(data)
        num_cells = len(cell_grids)

        act_function = nn.LeakyReLU(negative_slope=0.01)
        act_function1 = nn.Softplus()

        cell_grids = cell_grids.view(
            -1,
            self.num_input_filters,
            self.grid_size,
            self.grid_size,
            self.grid_size,
        )

        # x1 = self.conv1(cell_grids)
        # x1 = act_function1(x1)
        # x1 = F.avg_pool3d(x1, 2)
        x1_grid_size = 11

        x1 = torch.zeros(
            num_atoms,
            self.num_conv1_filters,
            x1_grid_size,
            x1_grid_size,
            x1_grid_size,
            device=device,
        )
        unique_atomic_numbers = torch.unique(channel_atomic_number)
        # Does a convolution on all grid cells of the same atomic number at once.
        # This is necessary since the input features are very sparse, only elements
        # that are in the system have non-zero values.
        # There might be a more efficient way of doing the following. I couldn't find a
        # solution that didn't need a for loop.
        for atomic_number in unique_atomic_numbers:
            indices = torch.arange(num_cells, device=device)
            indices = torch.masked_select(
                indices, channel_atomic_number.eq(atomic_number)
            )
            cell_grids_subset = torch.index_select(cell_grids, 0, indices)
            x1_subset = self.conv1[atomic_number](cell_grids_subset)
            channel_sum_index_subset = torch.index_select(
                channel_sum_index, 0, indices
            )
            x1.index_add_(0, channel_sum_index_subset, x1_subset)

        x1 = act_function1(x1)
        x1 = F.avg_pool3d(x1, 2)

        x2 = self.conv2(x1)
        x2 = act_function1(x2)

        x3 = self.conv3(x2)
        x3 = act_function(x3)
        # x3 = F.avg_pool3d(x3, 2)

        x4 = self.conv4(x3)
        x4 = act_function(x4)
        x4_max, indices = torch.max(
            x4.view(num_atoms, self.num_conv4_filters, -1), dim=2
        )

        x_energy_atom = self.fc1_energy(x4_max).view(
            num_atoms * self.max_atomic_number
        )

        # Select the correct computed energy based on the atom's atomic number
        offset = (
            torch.arange(num_atoms, device=device) * self.max_atomic_number
        )
        x_energy_atom = torch.index_select(
            x_energy_atom, 0, atomic_numbers + offset
        )
        x_energy = torch.zeros(len(data.y), device=device)
        x_energy.index_add_(0, data.batch, x_energy_atom)

        if self.regress_forces is True:
            x4_forces = self.conv4_forces(x3)
            x4_forces = act_function(x4_forces)
            x4_max_forces, indices = torch.max(
                x4_forces.view(num_atoms, self.num_conv4_filters, -1), dim=2
            )

            # Select the correct computed forces based on the atom's atomic number
            x_forces = self.fc1_forces(x4_max_forces).view(
                num_atoms * self.max_atomic_number, 3
            )
            x_forces = torch.index_select(x_forces, 0, atomic_numbers + offset)
            # Invert the 3D rotation applyied the the system
            inv_rot = torch.transpose(inv_rot, 0, 1)
            x_forces = torch.mm(x_forces, inv_rot)

            if self.show_timing_info is True:
                print("Time forward: {}".format(time.time() - start_time))

            return x_energy, x_forces

        if self.show_timing_info is True:
            print("Time forward: {}".format(time.time() - start_time))

        return x_energy

    def _create_grid_local(self, data):
        """
        Initializes the input convolutional input features from the atom positions
        """
        start_time = time.time()
        grid_size = self.grid_size
        device = data.x.device
        batch_size = len(data.y)

        # get the position of the atoms
        atom_pos = data.pos
        atomic_numbers = data.atomic_numbers

        # randomly tranlate and rotate the system of atoms
        atom_pos_perturb, inv_rot = Preprocess.perturb_atom_positions(
            atom_pos, self.grid_resolution, randomly_rotate=False
        )

        grid_pos = torch.zeros(
            3, grid_size * grid_size * grid_size, device=device
        )
        # Initialize the grid position values - probably a better way to do this
        grid_pos_ct = torch.arange(
            grid_size * grid_size * grid_size, device=device
        ).float()
        grid_pos[2] = torch.fmod(grid_pos_ct, float(grid_size))
        grid_pos_ct = (grid_pos_ct - grid_pos[2]) / float(grid_size)
        grid_pos[1] = torch.fmod(grid_pos_ct, float(grid_size))
        grid_pos_ct = (grid_pos_ct - grid_pos[1]) / float(grid_size)
        grid_pos[0] = torch.fmod(grid_pos_ct, float(grid_size))
        grid_pos = torch.transpose(grid_pos, 0, 1)
        grid_center = torch.zeros(3, device=device) + grid_size / 2.0
        grid_pos = (grid_pos - grid_center) * self.grid_resolution

        # compute offsets and scalars for computing the sine and cosine values
        freq_scalar = torch.zeros(self.num_input_filters, device=device)
        freq_offset = torch.zeros(self.num_input_filters, device=device)
        for i in range(self.num_input_filters):
            freq_scalar[i] = self.freq_scalar * pow(
                self.freq_base, float(math.floor(i / 2))
            )
            freq_offset[i] = 0.0

            if i % 2 == 0:  # convert from cosine to sine
                freq_scalar[i] = -freq_scalar[i]
                freq_offset[i] = math.pi / 2.0

        freq_scalar = freq_scalar.view(-1, 1).expand(
            -1, grid_size * grid_size * grid_size
        )
        freq_offset = freq_offset.view(-1, 1).expand(
            -1, grid_size * grid_size * grid_size
        )

        batch_size = len(data.y)
        cell_atom_delta = torch.zeros(0, 3, device=device)
        cell_sum_index = torch.zeros(0, device=device).long()
        channel_atomic_number = torch.zeros(0, device=device).long()
        channel_sum_index = torch.zeros(0, device=device).long()
        num_atom_channels = 0
        atom_count = 0

        for batch_idx in range(batch_size):
            # Get the atomic numbers of all atoms in the batch
            batch_atomic_numbers = torch.masked_select(
                atomic_numbers, data.batch.eq(batch_idx)
            ).long()
            batch_num_atoms = len(batch_atomic_numbers)

            # Mask used to remove cells corresponding to the same atoms
            identy_mask = torch.eye(batch_num_atoms, device=device).eq(0)

            # Compute the pairwise deltas (distance) between atoms
            atom_delta = torch.masked_select(
                atom_pos_perturb,
                data.batch.view(-1, 1).expand(-1, 3).eq(batch_idx),
            ).view(-1, 3)
            atom_delta = atom_delta.view(-1, 1, 3).repeat(
                1, batch_num_atoms, 1
            )
            atom_delta = atom_delta - torch.transpose(atom_delta, 0, 1)
            atom_delta = torch.masked_select(
                atom_delta,
                identy_mask.view(batch_num_atoms, batch_num_atoms, 1).expand(
                    -1, -1, 3
                ),
            )
            atom_delta = atom_delta.view(
                batch_num_atoms * (batch_num_atoms - 1), 3
            )
            cell_atom_delta = torch.cat([cell_atom_delta, atom_delta], dim=0)

            unique_atomic_numbers = torch.unique(batch_atomic_numbers)
            num_unique_atomic_numbers = len(unique_atomic_numbers)
            channel_atomic_number = torch.cat(
                [
                    channel_atomic_number,
                    unique_atomic_numbers.repeat(batch_num_atoms),
                ],
                dim=0,
            )

            # Cumpute the sum index for index_add operation in forward
            batch_channel_sum_index = torch.arange(
                batch_num_atoms, device=device
            )
            batch_channel_sum_index = batch_channel_sum_index.view(
                -1, 1
            ).repeat(1, num_unique_atomic_numbers)
            batch_channel_sum_index = batch_channel_sum_index.view(
                batch_num_atoms * num_unique_atomic_numbers
            )
            batch_channel_sum_index = batch_channel_sum_index + atom_count
            channel_sum_index = torch.cat(
                [channel_sum_index, batch_channel_sum_index], dim=0
            )

            batch_atomic_number_map = torch.zeros(
                self.max_atomic_number, device=device
            ).long()
            for i, atomic_number in enumerate(unique_atomic_numbers):
                batch_atomic_number_map[int(atomic_number)] = i

            batch_sum_index = torch.zeros(
                batch_num_atoms, device=device
            ).long()
            for i, atomic_number in enumerate(batch_atomic_numbers):
                batch_sum_index[i] = batch_atomic_number_map[
                    batch_atomic_numbers[i]
                ]

            batch_sum_index = batch_sum_index + num_atom_channels
            batch_sum_index = batch_sum_index.view(1, -1).repeat(
                batch_num_atoms, 1
            )
            offset = torch.arange(
                0,
                batch_num_atoms * num_unique_atomic_numbers,
                num_unique_atomic_numbers,
                device=device,
            ).long()

            batch_sum_index = batch_sum_index + offset.view(-1, 1).expand(
                -1, batch_num_atoms
            )
            batch_sum_index = torch.masked_select(
                batch_sum_index, identy_mask
            ).view(batch_num_atoms * (batch_num_atoms - 1))

            cell_sum_index = torch.cat(
                [cell_sum_index, batch_sum_index], dim=0
            )
            num_atom_channels += num_unique_atomic_numbers * batch_num_atoms
            atom_count += batch_num_atoms

        num_grid_cells = len(cell_sum_index)

        cell_grids = torch.zeros(
            num_atom_channels,
            self.num_input_filters,
            grid_size * grid_size * grid_size,
            device=device,
        )

        # Compute the grid cell values. Only do cell_batch_size at a time to ensure we don't run out of RAM
        process_idx = 0
        while process_idx < num_grid_cells:
            process_start = process_idx
            process_end = min(
                process_idx + self.cell_batch_size, num_grid_cells
            )
            process_length = process_end - process_start

            process_cell_atom_delta = cell_atom_delta[
                process_start:process_end
            ].clone()
            delta = grid_pos.view(1, -1, 3).repeat(process_length, 1, 1)
            delta += process_cell_atom_delta.view(process_length, 1, 3).expand(
                -1, grid_size * grid_size * grid_size, 3
            )

            distance_sqr = torch.sum((delta) ** 2, dim=2)

            # gaussian weighting of grid points
            gaussian_scalar = -1.0 / (
                2.0 * self.gaussian_std * self.gaussian_std
            )
            dist_gaussian = torch.exp(gaussian_scalar * distance_sqr)

            # cosine / sine weighting of grid points
            distance = torch.sqrt(distance_sqr)
            distance = distance.view(process_length, 1, -1).expand(
                process_length, self.num_input_filters, -1
            )

            distance = (distance / freq_scalar) + freq_offset

            dist_gaussian = dist_gaussian.view(process_length, 1, -1).expand(
                process_length, self.num_input_filters, -1
            )

            process_cell_sum_index = cell_sum_index[
                process_start:process_end
            ].clone()
            cell_grids.index_add_(
                0, process_cell_sum_index, dist_gaussian * torch.cos(distance)
            )

            process_idx = process_end

        cell_grids = cell_grids.view(
            num_atom_channels,
            self.num_input_filters,
            grid_size,
            grid_size,
            grid_size,
        )

        display_example_features = False
        if display_example_features is True:
            Display.display_grid_values(
                cell_grids[0], 1, self.display_base_name
            )
            Display.display_grid_values(
                cell_grids[1], 2, self.display_base_name
            )
            Display.display_grid_values(
                cell_grids[2], 3, self.display_base_name
            )
            Display.display_grid_values(
                cell_grids[3], 4, self.display_base_name
            )

        if self.show_timing_info is True:
            print(
                "Time _create_grid_local: {}".format(time.time() - start_time)
            )

        return cell_grids, channel_sum_index, channel_atomic_number, inv_rot
