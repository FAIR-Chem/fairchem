import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

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
        num_input_filters=10,  # must be multiple of 2
        max_num_elements=3,
        element_embedding_size=92,
        num_conv1_filters=16,
        conv1_kernal_size=5,
        num_conv2_filters=32,
        conv2_kernal_size=3,
        num_conv3_filters=32,
        conv3_kernal_size=1,
        num_conv4_filters=16,
        conv4_kernal_size=3,
        show_timing_info=False,
    ):
        super(CNN3D_LOCAL, self).__init__(
            num_atoms, bond_feat_dim, num_targets
        )

        self.conv1 = nn.Conv3d(
            max_num_elements * num_input_filters,
            num_conv1_filters,
            conv1_kernal_size,
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
        self.fc1_energy = nn.Linear(num_conv4_filters, max_num_elements)
        self.fc1_forces = nn.Linear(num_conv4_filters, max_num_elements * 3)
        self.num_input_filters = num_input_filters
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution
        self.gaussian_std = gaussian_std
        self.max_num_elements = max_num_elements
        self.regress_forces = regress_forces
        self.num_conv4_filters = num_conv4_filters
        self.show_timing_info = show_timing_info
        self.show_timing_info = show_timing_info
        self.element_embedding_size = element_embedding_size

        self.embedding_table = torch.zeros(
            max_num_elements, element_embedding_size
        )
        self.num_embeddings = 0

    def forward(self, data):
        start_time = time.time()
        num_atoms = len(data.x)

        x0, element_gate, inv_rot = self._create_grid_local(data)

        act_function = nn.LeakyReLU(negative_slope=0.01)
        act_function1 = nn.Softplus()

        x1 = self.conv1(x0)
        x1 = act_function1(x1)
        x1 = F.avg_pool3d(x1, 2)

        x2 = self.conv2(x1)
        x2 = act_function(x2)
        # x2 = F.avg_pool3d(x2, 2)

        x3 = self.conv3(x2)
        x3 = act_function(x3)
        # x3 = F.avg_pool3d(x3, 2)

        x4 = self.conv4(x3)
        x4 = act_function(x4)
        x4_max, indices = torch.max(
            x4.view(num_atoms, self.num_conv4_filters, -1), dim=2
        )

        x_energy = self.fc1_energy(x4_max)
        x_energy = x_energy * element_gate
        x_energy = torch.sum(x_energy, dim=1)
        x_energy = global_add_pool(x_energy, data.batch)

        if self.regress_forces is True:
            x4_forces = self.conv4_forces(x3)
            x4_forces = act_function(x4_forces)
            x4_max_forces, indices = torch.max(
                x4_forces.view(num_atoms, self.num_conv4_filters, -1), dim=2
            )

            x_forces = self.fc1_forces(x4_max_forces).view(
                -1, self.max_num_elements, 3
            )
            x_forces = x_forces * element_gate.view(
                -1, self.max_num_elements, 1
            ).expand(-1, self.max_num_elements, 3)
            x_forces = torch.sum(x_forces, dim=1)
            inv_rot = torch.transpose(inv_rot, 0, 1)
            x_forces = torch.mm(x_forces, inv_rot)

            if self.show_timing_info is True:
                print("Time forward: {}".format(time.time() - start_time))
            return x_energy, x_forces

        if self.show_timing_info is True:
            print("Time forward: {}".format(time.time() - start_time))

        # if torch.randint(0, 100, (1,))[0] == 1:
        #    Display.display_grid_values(x1[0], 1)
        #    Display.display_grid_values(x2[0], 2)
        #    Display.display_grid_values(x3[0], 3)
        #    Display.display_grid_values(x4[0], 4)

        return x_energy

    def _create_grid_local(self, data):
        """
        Initializes the input convolutional input features from the atom positions
        """
        grid_size = self.grid_size
        num_atoms = len(data.x)
        device = data.x.device

        # get the position of the atoms
        atom_pos = torch.narrow(data.x, 1, 92, 3)

        # Convert the atom embeddings to one hot embedding.
        # TODO: dataloader should return atomic numbers, will avoid the
        #       following feature-matching-to-embed logic.
        atom_embeddings = torch.narrow(
            data.x, 1, 0, self.element_embedding_size
        )
        atom_feats = torch.zeros(
            num_atoms, self.max_num_elements, device=device
        )
        for i, embedding in enumerate(atom_embeddings):
            found_match = False
            for j in range(self.num_embeddings):
                if torch.all(torch.eq(embedding, self.embedding_table[j])):
                    atom_feats[i][j] = 1.0
                    found_match = True

            if found_match is False:
                self.embedding_table[self.num_embeddings] = embedding
                atom_feats[i][self.num_embeddings] = 1.0
                self.num_embeddings = self.num_embeddings + 1

        # randomly tranlate and rotate the system of atoms
        atom_pos_perturb, inv_rot = Preprocess.perturb_atom_positions(
            atom_pos, self.grid_resolution, randomly_rotate=False
        )
        # print('Num atoms = {}'.format(num_atoms))

        grid = torch.zeros(
            num_atoms,
            self.max_num_elements,
            self.num_input_filters,
            grid_size * grid_size * grid_size,
            device=device,
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

        range_idx = torch.arange(num_atoms, device=device)
        freq_scalar = torch.zeros(self.num_input_filters, device=device)
        freq_offset = torch.zeros(self.num_input_filters, device=device)
        for i in range(self.num_input_filters):
            freq_scalar[i] = 0.015 * pow(1.9, float(math.floor(i / 2)))
            freq_offset[i] = 0.0

            if i % 2 == 0:
                freq_scalar[i] = -freq_scalar[i]
                freq_offset[i] = math.pi / 2.0

        freq_scalar = freq_scalar.view(-1, 1).expand(
            -1, grid_size * grid_size * grid_size
        )
        freq_offset = freq_offset.view(-1, 1).expand(
            -1, grid_size * grid_size * grid_size
        )

        start_time = time.time()
        gaussian_scalar = -1.0 / (2.0 * self.gaussian_std * self.gaussian_std)

        for i in range(num_atoms):
            batch_idx = data.batch[i]

            # Find the list of atoms in the same system
            mask = data.batch.eq(batch_idx)
            idx_list = torch.masked_select(range_idx, mask)
            mask = torch.eq(idx_list, i)
            mask = torch.eq(mask, False)
            idx_list = torch.masked_select(idx_list, mask)
            atom_feats_mask = torch.index_select(atom_feats, 0, idx_list)

            # put the grid positions in the same coordinate frame as the atoms
            grid_pos_trans = grid_pos + atom_pos_perturb[i]

            num_atoms_system = len(idx_list)

            # get the positions of the atoms in the system
            atom_pos_perturb_system = torch.index_select(
                atom_pos_perturb, 0, idx_list
            )
            atom_pos_perturb_system = atom_pos_perturb_system.view(
                num_atoms_system, 1, 3
            ).expand(num_atoms_system, grid_size * grid_size * grid_size, 3)

            # compute the squared distance between the grid points and the atoms
            grid_pos_trans = grid_pos_trans.view(1, -1, 3).expand(
                num_atoms_system, grid_size * grid_size * grid_size, 3
            )
            dist_sqr = torch.sum(
                (grid_pos_trans - atom_pos_perturb_system) ** 2, dim=2
            )

            # gaussian weighting of grid points
            dist_gaussian = torch.exp(gaussian_scalar * dist_sqr)

            # cosine / sine weighting of grid points
            dist = torch.sqrt(dist_sqr)
            dist = dist.view(num_atoms_system, 1, -1).expand(
                num_atoms_system, self.num_input_filters, -1
            )
            dist = (dist / freq_scalar) + freq_offset

            # put it all together
            dist_gaussian = dist_gaussian.view(num_atoms_system, 1, -1).expand(
                num_atoms_system, self.num_input_filters, -1
            )
            dist_gaussian = dist_gaussian * torch.cos(dist)

            atom_feats_expand = torch.transpose(atom_feats_mask, 0, 1)
            atom_feats_expand = atom_feats_expand.view(
                self.max_num_elements, num_atoms_system, 1, 1
            )
            atom_feats_expand = atom_feats_expand.expand(
                self.max_num_elements,
                num_atoms_system,
                self.num_input_filters,
                grid_size * grid_size * grid_size,
            )

            dist_gaussian = dist_gaussian.view(
                1, num_atoms_system, self.num_input_filters, -1
            )
            dist_gaussian = dist_gaussian.expand(
                self.max_num_elements,
                num_atoms_system,
                self.num_input_filters,
                -1,
            )

            grid[i] += torch.sum(dist_gaussian * atom_feats_expand, dim=1)

        # Add a gausian weight to each window - didn't show any improvement
        gaussian_grid_weight = False
        if gaussian_grid_weight is True:
            window_gaussian = grid_pos / self.grid_resolution
            window_gaussian = torch.sum((window_gaussian) ** 2, dim=1)
            window_gaussian_scalar = -1.0 / (
                2.0 * (float(grid_size) / 4.0) * (float(grid_size) / 4.0)
            )
            window_gaussian = torch.exp(
                window_gaussian_scalar * window_gaussian
            )
            window_gaussian = window_gaussian.view(
                1, 1, grid_size * grid_size * grid_size
            )
            window_gaussian = window_gaussian.expand(
                num_atoms, self.max_num_elements * self.num_input_filters, -1
            )

            grid = grid.view(
                num_atoms,
                self.max_num_elements * self.num_input_filters,
                grid_size * grid_size * grid_size,
            )
            grid = grid * window_gaussian

        display_example_features = False
        if display_example_features is True:
            grid = grid.view(
                num_atoms,
                self.max_num_elements * self.num_input_filters,
                grid_size,
                grid_size,
                grid_size,
            )
            Display.display_grid_values(grid[0], 1)
            Display.display_grid_values(grid[28], 2)
            Display.display_grid_values(grid[27], 3)
            Display.display_grid_values(grid[50], 4)
        grid = grid.view(
            num_atoms,
            self.max_num_elements * self.num_input_filters,
            grid_size,
            grid_size,
            grid_size,
        )

        if self.show_timing_info is True:
            print(
                "Time _create_grid_local: {}".format(time.time() - start_time)
            )

        return grid, atom_feats, inv_rot
