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

class Classifier(torch.nn.Module):
    """
    The attention module takes multiple hidden representations and compute the attention-weighted average
    Use Gated Attention Mechanism in https://arxiv.org/pdf/1802.04712.pdf
    """
    def __init__(self, parameters):
        super(Classifier, self).__init__()

        # MIL module
        self.mil_attn_V = torch.nn.Linear(512, 128, bias=False)
        self.mil_attn_U = torch.nn.Linear(512, 128, bias=False)
        self.mil_attn_w = torch.nn.Linear(128, 1, bias=False)

        # classifier
        self.classifier_linear = torch.nn.Linear(512, parameters["num_classes"], bias=False)

        # fusion branch
        self.fusion_dnn = torch.nn.Linear(parameters["post_processing_dim"] + 512, parameters["num_classes"])


    def forward(self, global_vec, h_crops):
        """
        Function that takes in the hidden representations of crops and use attention to generate a single hidden vector
        :param h_small:
        :param h_crops:
        :return:
        """
        # MIL module
        batch_size, num_crops, h_dim = h_crops.size()
        h_crops_reshape = h_crops.view(batch_size * num_crops, h_dim)

        # calculate the attn score
        attn_projection = torch.sigmoid(self.mil_attn_U(h_crops_reshape)) * torch.tanh(self.mil_attn_V(h_crops_reshape))
        attn_score = self.mil_attn_w(attn_projection)

        # use softmax to map score to attention
        attn_score_reshape = attn_score.view(batch_size, num_crops)
        self.patch_attns = torch.nn.functional.softmax(attn_score_reshape, dim=1)

        # final hidden vector
        z = torch.sum(self.patch_attns.unsqueeze(-1) * h_crops, 1)

        # map to the final layer
        self.y_local = torch.sigmoid(self.classifier_linear(z))

        # fusion branch
        concat_vec = torch.cat([global_vec, z], dim=1)
        self.y_fusion = torch.sigmoid(self.fusion_dnn(concat_vec))

        return self.y_fusion, self.y_local

    def loss(self, y_hat, y, weight):
        weight = weight if weight is None or len(weight) == len(y) else weight[:len(y)]
        return torch.nn.functional.binary_cross_entropy(y_hat, y, weight=weight, reduction='sum')
