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
import numpy as np
import src.modeling.modules as m
import src.utilities.tools as utils


class Classifier(torch.nn.Module):
    def __init__(self, parameters):
        super(Classifier, self).__init__()

        self._cam_size = parameters["cam_size"]
        self._crop_shape = parameters["crop_shape"]

        self.aggregation_function = m.TopTPercentAggregationFunction(parameters["percent_t"])
        self.retrieve_roi_module = m.RetrieveROIModule(parameters)

        # detection network
        self.local_network = m.LocalNetwork()
        self.dn_resnet = self.local_network.dn_resnet

        # MIL module
        self.attention_module = m.AttentionModule(parameters["num_classes"])
        self.mil_attn_V = self.attention_module.mil_attn_V
        self.mil_attn_U = self.attention_module.mil_attn_U
        self.mil_attn_w = self.attention_module.mil_attn_w

        # classifier
        self.classifier_linear = self.attention_module.classifier_linear

        # fusion branch
        self.fusion_dnn = torch.nn.Linear(parameters["post_processing_dim"] + 512, parameters["num_classes"])


    def _convert_crop_position(self, crops_x_small, cam_size, x_original):
        """
        Function that converts the crop locations from cam_size to x_original
        :param crops_x_small: N, k*c, 2 numpy matrix
        :param cam_size: (h,w)
        :param x_original: N, C, H, W pytorch variable
        :return: N, k*c, 2 numpy matrix
        """
        # retrieve the dimension of both the original image and the small version
        h, w = cam_size
        _, _, H, W = x_original.size()

        # interpolate the 2d index in h_small to index in x_original
        top_k_prop_x = crops_x_small[:, :, 0] / h
        top_k_prop_y = crops_x_small[:, :, 1] / w
        # sanity check
        assert np.max(top_k_prop_x) <= 1.0, "top_k_prop_x >= 1.0"
        assert np.min(top_k_prop_x) >= 0.0, "top_k_prop_x <= 0.0"
        assert np.max(top_k_prop_y) <= 1.0, "top_k_prop_y >= 1.0"
        assert np.min(top_k_prop_y) >= 0.0, "top_k_prop_y <= 0.0"
        # interpolate the crop position from cam_size to x_original
        top_k_interpolate_x = np.expand_dims(np.around(top_k_prop_x * H), -1)
        top_k_interpolate_y = np.expand_dims(np.around(top_k_prop_y * W), -1)
        top_k_interpolate_2d = np.concatenate([top_k_interpolate_x, top_k_interpolate_y], axis=-1)
        return top_k_interpolate_2d


    def _retrieve_crop(self, x_original, crop_positions, crop_method):
        """
        Function that takes in the original image and cropping position and returns the crops
        :param x_original_pytorch: PyTorch Tensor array (N,C,H,W)
        :param crop_positions:
        :return:
        """
        batch_size, num_crops, _ = crop_positions.shape
        crop_h, crop_w = self._crop_shape

        output = torch.ones((batch_size, num_crops, crop_h, crop_w)).type_as(x_original)

        for i in range(batch_size):
            for j in range(num_crops):
                utils.crop_pytorch(x_original[i, 0, :, :], self._crop_shape, crop_positions[i,j,:], output[i,j,:,:], method=crop_method)
        return output


    def forward(self, x_original, h_g, saliency_map):
        # calculate y_global
        # note that y_global is not directly used in inference
        self.y_global = self.aggregation_function.forward(saliency_map)

        # region proposal network
        small_x_locations = self.retrieve_roi_module.forward(x_original, self._cam_size, saliency_map)

        # convert crop locations that is on self.cam_size to x_original
        self.patch_locations = self._convert_crop_position(small_x_locations, self._cam_size, x_original)

        # patch retriever
        crops_variable = self._retrieve_crop(x_original, self.patch_locations, self.retrieve_roi_module.crop_method)
        self.patches = crops_variable.data.cpu().numpy()

        # detection network
        batch_size, num_crops, I, J = crops_variable.size()
        crops_variable = crops_variable.view(batch_size * num_crops, I, J).unsqueeze(1)
        h_crops = self.local_network.forward(crops_variable).view(batch_size, num_crops, -1)

        # MIL module
        # y_local is not directly used during inference
        z, self.patch_attns, self.y_local = self.attention_module.forward(h_crops)

        # fusion branch
        # use max pooling to collapse the feature map
        g1, _ = torch.max(h_g, dim=2)
        global_vec, _ = torch.max(g1, dim=2)
        concat_vec = torch.cat([global_vec, z], dim=1)
        self.y_fusion = torch.sigmoid(self.fusion_dnn(concat_vec))

        return self.y_fusion, self.y_global, self.y_local
