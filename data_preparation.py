# Data pre-processing pipeline, largely based on that of the original authors for accurate comparison of results:
# https://github.com/ARM-software/sesr/blob/master/utils.py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from typing import Tuple, List, Iterable, NamedTuple, Dict

AUTOTUNE = tf.data.AUTOTUNE
DATA_DIR = "data/"


class Batch(NamedTuple):
    hr: np.ndarray
    lr: np.ndarray


#Convert RGB image to YCbCr from https://github.com/ARM-software/sesr/blob/master/utils.p
def rgb_to_ycbcr(rgb: tf.Tensor) -> tf.Tensor:
    ycbcr_from_rgb = tf.constant([[65.481, 128.553, 24.966],
                                  [-37.797, -74.203, 112.0],
                                  [112.0, -93.786, -18.214]])
    rgb = tf.cast(rgb, dtype=tf.dtypes.float32) / 255.
    ycbcr = tf.linalg.matmul(rgb, ycbcr_from_rgb, transpose_b=True)
    return ycbcr + tf.constant([[[16., 128., 128.]]])


#Get the Y-Channel only
def rgb_to_y(example: tfds.features.FeaturesDict) -> Tuple[tf.Tensor, tf.Tensor]:
    lr_ycbcr = rgb_to_ycbcr(example['lr'])
    hr_ycbcr = rgb_to_ycbcr(example['hr'])
    return lr_ycbcr[..., 0:1] / 255., hr_ycbcr[..., 0:1] / 255.


def random_crop(low_res_img, high_res_img, low_res_crop_size, super_res_factor):
    """
    Return random crop from image for data augmentation, as done in original paper.
    Based on example found at: https://keras.io/examples/vision/edsr/
    """
    high_res_crop_size = low_res_crop_size * super_res_factor
    low_res_img_shape = tf.shape(low_res_img)[:2]  # Of shape (height, width)

    low_res_width = tf.random.uniform(shape=(), maxval=low_res_img_shape[1] - low_res_crop_size + 1, dtype=tf.int32)
    low_res_height = tf.random.uniform(shape=(), maxval=low_res_img_shape[0] - low_res_crop_size + 1, dtype=tf.int32)

    high_res_width = low_res_width * super_res_factor
    high_res_height = low_res_height * super_res_factor

    low_res_img_cropped = low_res_img[
                          low_res_height: low_res_height + low_res_crop_size,
                          low_res_width: low_res_width + low_res_crop_size,
                          ]
    high_res_img_cropped = high_res_img[
                           high_res_height: high_res_height + high_res_crop_size,
                           high_res_width: high_res_width + high_res_crop_size,
                           ]
    return low_res_img_cropped, high_res_img_cropped


def get_random_crops(low_res_img: tf.Tensor,
                     high_res_img: tf.Tensor,
                     low_res_crop_size: int,
                     super_res_factor: int,
                     num_crops: int) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    tuples = (random_crop(low_res_img, high_res_img, low_res_crop_size, super_res_factor) for _ in range(num_crops))
    lr, hr = zip(*tuples)
    return list(lr), list(hr)


def get_dataset(dataset_name: str,
                batch_size: int = 32,
                low_res_crop_size: int = 64,
                super_res_factor: int = 2,
                num_crops_per_image: int = 64,
                epochs: int = 300):
    """
    :param dataset_name: name of TensorFlow dataset to use e.g. div2k/bicubic_x2
    :param batch_size: batch size to use for training dataset
    :param low_res_crop_size: size of cropped patches of low res images (e.g. 64 will extract patches of size 64x64)
    :param super_res_factor: factor of super-resolution, for div2k/bicubic_x2 this is 2
    :param num_crops_per_image: number of cropped patches to extract per image
    :param epochs: number of epochs for training data
    :return: training dataset, validation dataset
    """
    dataset_dir = DATA_DIR
    ds_train, ds_val = tfds.load(dataset_name, split=["train", "validation"], shuffle_files=True, data_dir=dataset_dir)
    random_crops_fn = partial(get_random_crops, low_res_crop_size=low_res_crop_size, super_res_factor=super_res_factor,
                              num_crops=num_crops_per_image)
    ds_train = ds_train.map(rgb_to_y).map(random_crops_fn).unbatch().shuffle(buffer_size=10000).batch(
        batch_size).repeat(epochs)
    ds_val = ds_val.map(rgb_to_y).batch(1)

    ds_train = ds_train.map(lambda lr, hr: Batch(lr=lr, hr=hr)).prefetch(tf.data.AUTOTUNE)
    ds_val = ds_val.map(lambda lr, hr: Batch(lr=lr, hr=hr)).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val


