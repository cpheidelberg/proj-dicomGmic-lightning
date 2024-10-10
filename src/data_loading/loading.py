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
    Flip images to ensure correct horizontal orientation
    """
    flip  = horizontal_flip == 'NO' and VIEWS.is_right(view)
    flip |= horizontal_flip == 'YES' and VIEWS.is_left(view)
    return np.fliplr(image) if flip else image



def read_image(path: str, dtype) -> np.ndarray:
    """
    Open an image and return it as an NumPy array
    - path: the file path from which to read
    - dtype: the type of the resulting array
    """
    return np.array(pillow.open(path), dtype=dtype)


def write_image(path: str, image: np.ndarray):
    """
    Save an image from an NumPy array to a PNG file
    - path: the file path where the image will be saved
    - image: the image to be saved
    """
    pillow.fromarray(np.asarray(image)).save(path, format='PNG')


def crop_image(image, view, best_center) -> np.ndarray:
    """
    Applies augmentation window with random noise in location and size
    and return normalized cropped image.
    """
    img, _ = augmentations.random_augmentation_best_center(image, (2944, 1920), np.random.RandomState(0), best_center=best_center, view=view)
    return img.copy()


def _standardize(image):
    """
    Standardizes the image in-place; insuring that its pixel values are
    distributed with mean = 0 and sd = 1
    """
    # Standardizes an image in-place 
    image -= np.mean(image)
    image /= np.maximum(np.std(image), 10**(-5))


def adjust_brightness(image: np.ndarray) -> np.ndarray:
    """
    Scale brightness to 0 - 65535 range
    If the most common colour (i.e. probably the background colour) is too light,
    we assume that it means that the background is light and tissue is dark
    in this image, so we invert the colours to ensure that all images have light
    tissue on dark background.
    """
    img = image * (2 ** 16 - 1) // image.max()
    most_frequent = np.argmax(np.bincount(img.flatten()))
    return (2 ** 16 - 1) - img if most_frequent > 10000 else img


def process_image(image, view, horizontal_flip, best_center) -> np.ndarray:
    """
    Flip, crop and standardize
    """
    img = crop_image(flip_image(image, view, horizontal_flip), view, best_center)
    _standardize(img)
    return img


def read_image_standardized(image) -> np.ndarray:
    """
    Reads an image as float32 and standardizes it
    - path: the file path from which to read
    """
    img = read_image(image, 'float32')
    _standardize(img)
    return img
