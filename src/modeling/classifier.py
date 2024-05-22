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
import src.modeling.modules as m


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


    def forward(self, h_g, h_crops):

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
