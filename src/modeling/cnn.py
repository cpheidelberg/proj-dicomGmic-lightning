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

        self.global_network = modules.GlobalNetwork(parameters)
        self.downsampling_branch = self.global_network.downsampling_branch
        self.postprocess_module = self.global_network.postprocess_module

        self.retrieve_roi_module = modules.RetrieveROIModule(parameters)

        self.detection_network = modules.ResNetV1(64, modules.BasicBlockV1, [2,2,2,2], 3)


    def forward(self, x_original):
        """
        :param x_original: N x H x W x C array
        """
        # global network: x_small -> class activation map
        h_g, self.saliency_map = self.global_network.forward(x_original)

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
