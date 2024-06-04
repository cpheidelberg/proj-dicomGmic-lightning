# Copyright (C) 2020 Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, Kangning Liu,
# Sudarshini Tyagi, Laura Heacock, S. Gene Kim, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of GMIC.
#
# GMIC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# GMIC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GMIC.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import torch
from src.modeling import modules
from src.utilities import tools


class CNN(torch.nn.Module):
    def __init__(self, parameters):
        super(CNN, self).__init__()

        self._cam_size = parameters["cam_size"]
        self._crop_shape = parameters["crop_shape"]
        self._percent_t = parameters["percent_t"]

        self.downsampling_branch = modules.ResNetV2(
            input_channels=1, num_filters=16,
            # first conv layer
            first_layer_kernel_size=(7,7), first_layer_conv_stride=2,
            first_layer_padding=3,
            # first pooling layer
            first_pool_size=3, first_pool_stride=2, first_pool_padding=0,
            # res blocks architecture
            blocks_per_layer_list=[2, 2, 2, 2, 2],
            block_strides_list=[1, 2, 2, 2, 2],
            block_fn=modules.BasicBlockV2,
            growth_factor=2)

        self.postprocess_module = torch.nn.Conv2d(
            in_channels=parameters["post_processing_dim"],
            out_channels=parameters["num_classes"],
            kernel_size=(1, 1),
            bias=False)

        self.retrieve_roi_module = modules.RetrieveROIModule(
            num_crops_per_class = parameters["K"],
            crop_shape = parameters["crop_shape"],
            gpu_number = None if parameters["device_type"] != "gpu" else parameters["gpu_number"])

        self.detection_network = modules.ResNetV1(
            initial_filters=64,
            block=modules.BasicBlockV1,
            layers=[2,2,2,2], input_channels=3)


    def forward(self, x_original):
        """
        :param x_original: N x H x W x C array
        """
        # global network: x_small -> class activation map
        h_g = self.downsampling_branch.forward(x_original)
        self.saliency_map = torch.sigmoid(self.postprocess_module.forward(h_g))

        # calculate y_global
        # note that y_global is not directly used in inference
        self.y_global = modules.top_t_percent(self.saliency_map, self._percent_t)

        # region proposal network
        small_x_locations = self.retrieve_roi_module.forward(x_original, self._cam_size, self.saliency_map)

        # convert crop locations that is on self.cam_size to x_original
        self.patch_locations = tools.scale_crops(small_x_locations, self._cam_size, x_original.size()[-2:])

        # patch retriever
        crops_variable = tools.retrieve_crops(x_original, self.patch_locations, self._crop_shape, self.retrieve_roi_module.crop_method)
        self.patches = crops_variable.data.cpu().numpy()

        # detection network
        batch_size, num_crops, I, J = crops_variable.size()
        crops_var = crops_variable.view(batch_size * num_crops, I, J).unsqueeze(1).expand(-1, 3, -1 , -1)
        h_crops = self.detection_network.forward(crops_var).mean(dim=2).mean(dim=2).view(batch_size, num_crops, -1)

        # use max pooling to collapse the feature map
        g1, _ = torch.max(h_g, dim=2)
        global_vec, _ = torch.max(g1, dim=2)

        return self.y_global, global_vec, h_crops
