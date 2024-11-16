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

"""
Module that define the core logic of GMIC
"""

import torch
from src.utilities import tools
import src.modeling.modules as m


class GMIC(torch.nn.Module):
    def __init__(self, parameters):
        super(GMIC, self).__init__()

        # save parameters
        self.experiment_parameters = parameters
        self.cam_size = parameters["cam_size"]

        # construct networks
        # global network
        self.global_network = m.GlobalNetwork(self.experiment_parameters, self)
        self.global_network.add_layers()

        # aggregation function
        self.aggregation_function = m.TopTPercentAggregationFunction(self.experiment_parameters, self)

        # detection module
        self.retrieve_roi_crops = m.RetrieveROIModule(self.experiment_parameters, self)

        # detection network
        self.local_network = m.LocalNetwork(self.experiment_parameters, self)
        self.local_network.add_layers()

        # MIL module
        self.attention_module = m.AttentionModule(self.experiment_parameters, self)
        self.attention_module.add_layers()

        # fusion branch
        self.fusion_dnn = torch.nn.Linear(parameters["post_processing_dim"]+512, parameters["num_classes"])


    def forward_cnn(self, x):
        """
        Execute the forward step of the CNN part of the GMIC, converting
        the input image into y_global and the feature vector (h_crops & global_vec).
        - x: (N,H,W,C) Tensor
        """
        # global network: x_small -> class activation map
        h_g, self.saliency_map = self.global_network.forward(x)

        # calculate y_global
        # note that y_global is not directly used in inference
        self.y_global = self.aggregation_function.forward(self.saliency_map)

        # region proposal network
        small_x_locations = self.retrieve_roi_crops.forward(x, self.cam_size, self.saliency_map)

        # convert crop locations that is on self.cam_size to x
        self.patch_locations = tools.scale_crops(small_x_locations, self.cam_size, x.size()[2:])

        # patch retriever
        crops_variable = tools.retrieve_crops(x, self.patch_locations, self.experiment_parameters["crop_shape"], self.retrieve_roi_crops.crop_method)
        self.patches = crops_variable.data.cpu().numpy()

        # detection network
        batch_size, num_crops, I, J = crops_variable.size()
        crops_variable = crops_variable.view(batch_size * num_crops, I, J).unsqueeze(1)
        h_crops = self.local_network.forward(crops_variable).view(batch_size, num_crops, -1)

        # use max pooling to collapse the feature map
        g1, _ = torch.max(h_g, dim=2)
        global_vec, _ = torch.max(g1, dim=2)

        return self.y_global, h_crops, global_vec


    def forward_classifier(self, global_vec, h_crops):
        """
        Execute the forward step of the classification step of the GMIC,
        converting the feature vector (global_vec & h_crops) into y_fusion
        and y_local.
        - global_vec: Tensor
        - h_crops: Tensor
        """

        # MIL module
        # y_local is not directly used during inference
        # print(f'{global_vec.shape=}, {h_crops.shape=}')
        z, self.patch_attns, self.y_local = self.attention_module.forward(h_crops)

        # fusion branch
        self.y_fusion = torch.sigmoid(self.fusion_dnn(torch.cat([global_vec, z], dim=1)))

        return self.y_fusion, self.y_local


    def forward(self, x):
        """
        Execute the forward step for the entire GMIC.
        - x: N,H,W,C Tensor
        """
        y_global, h_crops, global_vec = self.forward_cnn(x)
        y_fusion, y_local = self.forward_classifier(global_vec, h_crops)

        return y_fusion, y_global, y_local


    def cnn_named_parameters(self):
        """
        Filter named parameters and return only those used in the forward_cnn() step.
        """
        classifier = ('fusion_dnn', 'classifier_linear', 'mil_attn_V', 'mil_attn_U', 'mil_attn_w')
        return ((name, param) for name, param in self.named_parameters() if not name.startswith(classifier))
