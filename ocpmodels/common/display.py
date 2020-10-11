"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import time
from collections import defaultdict, deque

import torch
from PIL import Image


class Display:
    def display_grid_values(grid, grid_idx, base_name):
        start_time = time.time()
        # print('{}'.format(len(grid)))
        num_channels = len(grid)
        num_x = len(grid[0])
        num_y = len(grid[0][0])
        num_z = len(grid[0][0][0])
        print(
            "{} {} {} {} {}".format(
                num_channels, num_x, num_y, num_z, torch.sum(grid)
            )
        )

        border_size = 2

        height_data = num_z * (num_y)
        width_data = num_channels * (num_x)
        height_img = num_z * (num_y + border_size)
        width_img = num_channels * (num_x + border_size)

        img = Image.new("RGB", (width_img, height_img))
        blue_cpu = torch.zeros(height_data, width_data)

        grid_cpu = grid.detach().clone().to("cpu")
        grid_cpu = grid_cpu.view(width_data, height_data)
        grid_cpu = torch.transpose(grid_cpu, 0, 1).contiguous()
        grid_cpu = grid_cpu.view(num_y, num_z, width_data)
        grid_cpu = torch.transpose(grid_cpu, 0, 1).contiguous()
        grid_cpu = grid_cpu.view(width_data, height_data)

        red_cpu = -1.0 * torch.clamp(grid_cpu, max=0.0)
        green_cpu = torch.clamp(grid_cpu, min=0.0)

        # Add lines vertically
        red_cpu = red_cpu.view(height_data, num_channels, num_x)
        red_cpu = torch.cat(
            (red_cpu, torch.zeros(height_data, num_channels, border_size)),
            dim=2,
        )
        red_cpu = red_cpu.view(height_data, width_img)

        green_cpu = green_cpu.view(height_data, num_channels, num_x)
        green_cpu = torch.cat(
            (green_cpu, torch.zeros(height_data, num_channels, border_size)),
            dim=2,
        )
        green_cpu = green_cpu.view(height_data, width_img)

        blue_cpu = blue_cpu.view(height_data, num_channels, num_x)
        blue_cpu = torch.cat(
            (
                blue_cpu,
                torch.zeros(height_data, num_channels, border_size).fill_(100),
            ),
            dim=2,
        )
        blue_cpu = blue_cpu.view(height_data, width_img)

        # Add lines horizontally
        red_cpu = red_cpu.view(num_z, num_y, width_img)
        red_cpu = torch.cat(
            (red_cpu, torch.zeros(num_z, border_size, width_img)), dim=1
        )
        red_cpu = red_cpu.view(height_img, width_img)

        green_cpu = green_cpu.view(num_z, num_y, width_img)
        green_cpu = torch.cat(
            (green_cpu, torch.zeros(num_z, border_size, width_img)), dim=1
        )
        green_cpu = green_cpu.view(height_img, width_img)

        blue_cpu = blue_cpu.view(num_z, num_y, width_img)
        blue_cpu = torch.cat(
            (blue_cpu, torch.zeros(num_z, border_size, width_img).fill_(100)),
            dim=1,
        )
        blue_cpu = blue_cpu.view(height_img, width_img)

        max_val = max(torch.max(red_cpu), torch.max(green_cpu)) + 0.0001
        red_cpu = torch.clamp(255.0 * red_cpu / max_val, max=255.0)
        red_cpu = red_cpu.view(-1)
        green_cpu = torch.clamp(255.0 * green_cpu / max_val, max=255.0)
        green_cpu = green_cpu.view(-1)
        blue_cpu = blue_cpu.view(-1)
        img.putdata(list(zip(red_cpu, green_cpu, blue_cpu)))

        img.save("{}grid_{}.png".format(base_name, grid_idx))
        print("TIME: display_grid_values: {}".format(time.time() - start_time))

    def display_model_conv_weights(weights, conv_layer_idx, base_name):
        start_time = time.time()
        weights = weights.clone().detach().cpu()
        weight_size = weights.size()
        # print('{}'.format(weight_size))
        out_channels = weight_size[0]
        in_channels = weight_size[1]
        num_x = weight_size[2]
        num_y = weight_size[3]
        num_z = weight_size[4]

        border_size = 1

        height_data = in_channels * num_z * num_y
        width_data = out_channels * num_x
        height_img = (
            in_channels * (num_z + border_size) * (num_y + border_size)
        )
        width_img = out_channels * (num_x + border_size)

        # ic, oc, x, y, z
        weights = torch.transpose(weights, 0, 1).contiguous()
        # ic, oc*x, z*y
        weights = (
            torch.transpose(weights, 3, 4)
            .contiguous()
            .view(in_channels, out_channels * num_x, num_z * num_y)
        )
        # ic*z*y, oc*x
        weights = (
            torch.transpose(weights, 1, 2)
            .contiguous()
            .view(height_data, width_data)
        )

        img = Image.new("RGB", (width_img, height_img))
        blue_cpu = torch.zeros(height_data, width_data)
        red_cpu = -1.0 * torch.clamp(weights, max=0.0)
        green_cpu = torch.clamp(weights, min=0.0)

        # Add lines horizontally
        red_cpu = red_cpu.view(height_data, out_channels, num_x)
        red_cpu = torch.cat(
            (red_cpu, torch.zeros(height_data, out_channels, border_size)),
            dim=2,
        )
        red_cpu = red_cpu.view(height_data, width_img)

        green_cpu = green_cpu.view(height_data, out_channels, num_x)
        green_cpu = torch.cat(
            (green_cpu, torch.zeros(height_data, out_channels, border_size)),
            dim=2,
        )
        green_cpu = green_cpu.view(height_data, width_img)

        blue_cpu = blue_cpu.view(height_data, out_channels, num_x)
        blue_cpu = torch.cat(
            (
                blue_cpu,
                torch.zeros(height_data, out_channels, border_size).fill_(100),
            ),
            dim=2,
        )
        blue_cpu = blue_cpu.view(height_data, width_img)

        # Add lines vertically
        red_cpu = red_cpu.view(in_channels, num_z, num_y, width_img)
        red_cpu = torch.cat(
            (red_cpu, torch.zeros(in_channels, border_size, num_y, width_img)),
            dim=1,
        )
        red_cpu = torch.cat(
            (
                red_cpu,
                torch.zeros(
                    in_channels, num_z + border_size, border_size, width_img
                ),
            ),
            dim=2,
        )
        red_cpu = red_cpu.view(height_img, width_img)

        green_cpu = green_cpu.view(in_channels, num_z, num_y, width_img)
        green_cpu = torch.cat(
            (
                green_cpu,
                torch.zeros(in_channels, border_size, num_y, width_img),
            ),
            dim=1,
        )
        green_cpu = torch.cat(
            (
                green_cpu,
                torch.zeros(
                    in_channels, num_z + border_size, border_size, width_img
                ),
            ),
            dim=2,
        )
        green_cpu = green_cpu.view(height_img, width_img)

        blue_cpu = blue_cpu.view(in_channels, num_z, num_y, width_img)
        blue_cpu = torch.cat(
            (
                blue_cpu,
                torch.zeros(in_channels, border_size, num_y, width_img).fill_(
                    100
                ),
            ),
            dim=1,
        )
        blue_cpu = torch.cat(
            (
                blue_cpu,
                torch.zeros(
                    in_channels, num_z + border_size, border_size, width_img
                ).fill_(100),
            ),
            dim=2,
        )
        blue_cpu = blue_cpu.view(height_img, width_img)

        max_val = max(torch.max(red_cpu), torch.max(green_cpu)) + 0.0001
        red_cpu = torch.clamp(320.0 * red_cpu / max_val, max=255.0)
        red_cpu = red_cpu.view(-1)
        green_cpu = torch.clamp(320.0 * green_cpu / max_val, max=255.0)
        green_cpu = green_cpu.view(-1)
        blue_cpu = blue_cpu.view(-1)
        img.putdata(list(zip(red_cpu, green_cpu, blue_cpu)))

        img.save("{}conv{}.png".format(base_name, conv_layer_idx))
        print(
            "TIME: display_model_weights: {}".format(time.time() - start_time)
        )

    def display_model_fc_weights(weights, fc_layer_idx, base_name):
        start_time = time.time()
        weights = weights.clone().detach().cpu()
        weight_size = weights.size()
        # print('{}'.format(weight_size))
        in_channels = weight_size[1]
        out_channels = weight_size[0]

        height = in_channels
        width = out_channels

        img = Image.new("RGB", (width, height))
        img.paste((0, 0, 100), [0, 0, img.size[0], img.size[1]])

        for oc in range(out_channels):
            for ic in range(in_channels):
                val = 1000.0 * weights[oc][ic]
                if val > 0:
                    img.putpixel((oc, ic), (0, val, 0))
                if val < 0:
                    img.putpixel((oc, ic), (-val, 0, 0))

        img.save("{}fc{}.png".format(base_name, fc_layer_idx))
        print(
            "TIME: display_model_weights: {}".format(time.time() - start_time)
        )

    def display_output_tracker(tracker, idx, base_name):
        output = tracker[0]
        target = tracker[1]

        min_val = torch.min(target)
        max_val = torch.max(target)
        delta = max_val - min_val
        min_val = min_val - 0.2 * delta
        max_val = max_val + 0.2 * delta

        width = len(output)
        height = 100

        img = Image.new("RGB", (width, height))
        img.paste((0, 0, 50), [0, 0, img.size[0], img.size[1]])

        for c in range(width):
            r = height * (output[c] - min_val) / (max_val - min_val)
            r = min(height - 1, r)
            r = max(0, r)

            tr = height * (target[c] - min_val) / (max_val - min_val)
            tr = min(height - 1, tr)
            tr = max(0, tr)

            if tr > r:
                for r0 in range(int(r), int(tr)):
                    img.putpixel((c, r0), (128, 150, 0))
            else:
                for r0 in range(int(tr), int(r)):
                    img.putpixel((c, r0), (150, 128, 0))

            img.putpixel((c, r), (255, 255, 0))
            img.putpixel((c, tr), (255, 0, 0))

        img.save("{}tracker{}.png".format(base_name, idx))
