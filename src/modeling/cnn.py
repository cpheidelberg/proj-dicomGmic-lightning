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

class CNN(torch.nn.Module):
    def __init__(self, parameters):
        super(CNN, self).__init__()

        self.global_network = m.GlobalNetwork(parameters)
        self.downsampling_branch = self.global_network.downsampling_branch
        self.postprocess_module = self.global_network.postprocess_module

    def forward(self, x_original):
        """
        :param x_original: N x H x W x C array
        """
        # global network: x_small -> class activation map
        h_g, self.saliency_map = self.global_network.forward(x_original)

        # Collapse the dimensions (except the batch size) into one
        feature_vector = h_g.reshape((h_g.shape[0], h_g.shape[1] * h_g.shape[2] * h_g.shape[3]))

        return self.saliency_map, h_g, feature_vector
