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
from src.modeling import cnn, classifier


class GMIC(torch.nn.Module):
    def __init__(self, parameters):
        super(GMIC, self).__init__()

        self.cnn = cnn.CNN(parameters)
        self.classifier = classifier.Classifier(parameters)


    def forward(self, x_original):
        saliency_map, h_g, feature_vector = self.cnn.forward(x_original)
        y_fusion, y_global, y_local = self.classifier.forward(x_original, saliency_map, h_g)

        return y_fusion, y_global, y_local, feature_vector
