import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from typing import Tuple, List, Iterable, NamedTuple, Dict

import models.model

AUTOTUNE = tf.data.AUTOTUNE
DATA_DIR = "data/"


class Batch(NamedTuple):
    hr: np.ndarray
    lr: np.ndarray


# Convert RGB image to YCbCr from https://github.com/ARM-software/sesr/blob/master/utils.p
def rgb_to_ycbcr(rgb: tf.Tensor) -> tf.Tensor:
    ycbcr_from_rgb = tf.constant([[65.481, 128.553, 24.966],
                                  [-37.797, -74.203, 112.0],
                                  [112.0, -93.786, -18.214]])
    rgb = tf.cast(rgb, dtype=tf.dtypes.float32) / 255.
    ycbcr = tf.linalg.matmul(rgb, ycbcr_from_rgb, transpose_b=True)
    return ycbcr + tf.constant([[[16., 128., 128.]]])


def upscale_image_with_model(low_res_rgb: tf.Tensor, model: models.model.Model, model_weights: Dict) -> tf.Tensor:
    """Takes a low-res rgb image and upsamples it"""
    low_res_ycbcr = rgb_to_ycbcr(low_res_rgb)
    y = low_res_ycbcr[0][..., 0]
    cb = low_res_ycbcr[0][..., 1]
    cr = low_res_ycbcr[0][..., 2]
    # Upsample Cb and Cr channels with bicubic interpolation, as is standard practise
    cb_upsampled = tf.image.resize(cb[..., np.newaxis], (128, 128), method="bicubic")[np.newaxis, ...]
    cr_upsampled = tf.image.resize(cr[..., np.newaxis], (128, 128), method="bicubic")[np.newaxis, ...]
    net_input = y.numpy()[np.newaxis, ...][..., np.newaxis] / 255
    y_upsampled = model.forward(model_weights, net_input)
    # Concatenate upsampled Y, Cb and Cr channels
    full_upsampled_ycbcr = np.concatenate((y_upsampled * 255, cb_upsampled), axis=3)
    full_upsampled_ycbcr = np.concatenate((full_upsampled_ycbcr, cr_upsampled), axis=3)
    rgb_network_upsampled = ycbcr_to_rgb(full_upsampled_ycbcr)
    return rgb_network_upsampled


def ycbcr_to_rgb(ycbcr: tf.Tensor) -> tf.Tensor:
    """Takes non-normalised ycbcr channels and returns non-normalised rgb image"""
    ycbcr = ycbcr - tf.constant([[16., 128., 128.]])
    rgb_from_ycbcr = tf.constant([[255./219., 0., (255./224)*1.402],
                                  [255./219., (255./224.)*1.772*(0.114/0.587), -(255./224.)*1.402*(0.299/0.587)],
                                  [255./219., (255./224.)*1.772, 0.]])
    rgb = tf.linalg.matmul(ycbcr, rgb_from_ycbcr, transpose_b=True)
    return rgb


# Get the Y-Channel only
def rgb_to_y(img: tf.Tensor) -> tf.Tensor:
    lr_ycbcr = rgb_to_ycbcr(img)
    return lr_ycbcr[..., 0:1] / 255.


def remove_dict(example: tfds.features.FeaturesDict) -> Tuple[tf.Tensor, tf.Tensor]:
    lr = example['lr']
    hr = example['hr']
    return lr, hr


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


def get_test_dataset(dataset_name: str,
                           batch_size: int = 32,
                           low_res_crop_size: int = 64,
                           super_res_factor: int = 2):
    """
    :param dataset_name: name of TensorFlow dataset to use e.g. div2k/bicubic_x2
    :param batch_size: batch size to use for training dataset
    :param low_res_crop_size: size of cropped patches of low res images (e.g. 64 will extract patches of size 64x64)
    :param super_res_factor: factor of super-resolution, for div2k/bicubic_x2 this is 2
    :param num_crops_per_image: number of cropped patches to extract per image
    :param epochs: number of epochs for training data
    :return: training dataset, validation dataset
    """
    dataset_dir = DATA_DIR + "div2k_bicubic_x2"
    ds_train, ds_val = tfds.load(dataset_name, split=["train", "validation"], shuffle_files=True, data_dir=dataset_dir)
    val_random_crops_fn = partial(get_random_crops, low_res_crop_size=low_res_crop_size,
                                  super_res_factor=super_res_factor,
                                  num_crops=32)  # Fewer crops to reduce time spent evaluating
    ds_val = ds_val.map(remove_dict).map(val_random_crops_fn).unbatch().batch(batch_size)
    ds_val = ds_val.map(lambda lr, hr: Batch(lr=lr, hr=hr)).prefetch(tf.data.AUTOTUNE)

    return ds_val


if __name__ == "__main__":
    test_ds = tfds.as_numpy(get_test_dataset("div2k", batch_size=1))
    test_ds = iter(test_ds)
    test_im_rgb = next(test_ds).lr
    test_im_ycbcr = rgb_to_ycbcr(test_im_rgb)
    back_to_rgb = ycbcr_to_rgb(test_im_ycbcr)
    print(test_im_rgb.shape)
    print(back_to_rgb.shape)
    print(test_im_rgb - back_to_rgb)

