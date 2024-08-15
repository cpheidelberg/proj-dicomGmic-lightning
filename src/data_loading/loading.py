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

import numpy as np
import PIL.Image as pillow
from src.constants import VIEWS
from src.data_loading import augmentations


def flip_image(image, view, horizontal_flip) -> np.ndarray:
    """
    If training mode, makes all images face right direction.
    In medical, keeps the original directions unless horizontal_flip is set.
    """
    flip  = horizontal_flip == 'NO' and VIEWS.is_right(view)
    flip |= horizontal_flip == 'YES' and VIEWS.is_left(view)
    return np.fliplr(image) if flip else image


def read_image(path, dtype) -> np.ndarray:
    return np.array(pillow.open(path), dtype=dtype)


def write_image(path, image):
    pillow.fromarray(np.asarray(image)).save(path, format='PNG')


def flip_and_crop(image, view, horizontal_flip, best_center) -> np.ndarray:
    image = flip_image(image, view, horizontal_flip)
    image, _ = augmentations.random_augmentation_best_center(
        image=image,
        input_size=(2944, 1920),
        random_number_generator=np.random.RandomState(0),
        best_center=best_center,
        view=view
    )
    return image.copy()


def _standardize(image):
    # Standardizes an image in-place 
    image -= np.mean(image)
    image /= np.maximum(np.std(image), 10**(-5))


def adjust_brightness(image: np.ndarray) -> np.ndarray:
    image = image * (2 ** 16 - 1) // image.max()
    if np.argmax(np.bincount(image.flatten())) > 10000:
        return (2 ** 16 - 1) - image
    return image


def process_image(image, view, horizontal_flip, best_center) -> np.ndarray:
    """
    Applies augmentation window with random noise in location and size
    and return normalized cropped image.
    """
    image = flip_and_crop(image, view, horizontal_flip, best_center)
    _standardize(image)
    return image


def read_image_standardized(path) -> np.ndarray:
    image = read_image(path, dtype=np.float32)
    _standardize(image)
    return image
