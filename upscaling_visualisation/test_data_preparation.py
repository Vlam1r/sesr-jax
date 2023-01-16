import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from typing import Tuple, List, Iterable, NamedTuple, Dict
import jax.numpy as jnp
import matplotlib.pyplot as plt

import models.model

AUTOTUNE = tf.data.AUTOTUNE
DATA_DIR = "../data/"


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
    cb_upsampled = tf.image.resize(cb[..., np.newaxis], (2 * cb.shape[0], 2 * cb.shape[1]), method="bicubic")[
        np.newaxis, ...]
    cr_upsampled = tf.image.resize(cr[..., np.newaxis], (2 * cr.shape[0], 2 * cr.shape[1]), method="bicubic")[
        np.newaxis, ...]
    net_input = y.numpy()[np.newaxis, ...][..., np.newaxis] / 255
    y_upsampled = model.apply(model_weights, net_input)
    # Concatenate upsampled Y, Cb and Cr channels
    full_upsampled_ycbcr = np.concatenate((y_upsampled * 255, cb_upsampled), axis=3)
    full_upsampled_ycbcr = np.concatenate((full_upsampled_ycbcr, cr_upsampled), axis=3)
    rgb_network_upsampled = ycbcr_to_rgb(full_upsampled_ycbcr)
    return rgb_network_upsampled


def ycbcr_to_rgb(ycbcr: tf.Tensor) -> tf.Tensor:
    """Takes non-normalised ycbcr channels and returns non-normalised rgb image"""
    ycbcr = ycbcr - tf.constant([[16., 128., 128.]])
    rgb_from_ycbcr = tf.constant([[255. / 219., 0., (255. / 224) * 1.402],
                                  [255. / 219., (255. / 224.) * 1.772 * (0.114 / 0.587),
                                   -(255. / 224.) * 1.402 * (0.299 / 0.587)],
                                  [255. / 219., (255. / 224.) * 1.772, 0.]])
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


def get_test_dataset(dataset_name: str):
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
    ds_val = ds_val.map(remove_dict).batch(1)
    ds_val = ds_val.map(lambda lr, hr: Batch(lr=lr, hr=hr)).prefetch(tf.data.AUTOTUNE)

    return ds_val


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
    return low_res_img_cropped, high_res_img_cropped, low_res_width, low_res_height


def plot_crop_comparisons_and_full(test_im_rgb):
    low_res_crop, high_res_crop, crop_x, crop_y = random_crop(test_im_rgb.lr[0], test_im_rgb.hr[0], 64, 2)
    print(f"crop_x: {crop_x} crop_y: {crop_y}")
    im_width = test_im_rgb.lr[0].shape[1]
    im_height = test_im_rgb.lr[0].shape[0]
    bbox = jnp.array([crop_y / im_height, crop_x / im_width, (crop_y + 64) / im_height, (crop_x + 64) / im_width])
    bbox = bbox[jnp.newaxis, ...][jnp.newaxis, ...]
    low_res_crop = low_res_crop[jnp.newaxis, ...]
    high_res_crop = high_res_crop[jnp.newaxis, ...]

    m3_network = models.model.Model(network="M3", should_collapse=True)
    m3_params = jnp.load("../trained_model_parameters/params_M3_300.npz", allow_pickle=True)
    m3_params = m3_params['arr_0'].item(0)
    m3_model_upscaled_image = upscale_image_with_model(low_res_crop, m3_network, m3_params)[0] / 255

    full_low_res = ycbcr_to_rgb(rgb_to_ycbcr(test_im_rgb.lr[0])) / 255
    full_low_res = \
        tf.image.draw_bounding_boxes(full_low_res[jnp.newaxis, ...], bbox, colors=jnp.array([[1.0, 0.0, 0.0]]))[0]
    low_res_plot = ycbcr_to_rgb(rgb_to_ycbcr(low_res_crop[0])) / 255
    high_res_plot = ycbcr_to_rgb(rgb_to_ycbcr(high_res_crop[0])) / 255
    bicubic = tf.image.resize(low_res_plot, (2 * low_res_plot.shape[0], 2 * low_res_plot.shape[1]), method="bicubic")

    figure_mosaic = """
    GGG
    GGG
    ABC
    DEF
    """

    fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(14, 12))
    axes["G"].imshow(full_low_res)
    axes["G"].set_xlabel("Full Image")
    axes["A"].imshow(low_res_plot)
    axes["A"].set_xlabel("Low Resolution")
    axes["B"].imshow(high_res_plot)
    axes["B"].set_xlabel("High Resolution")
    axes["C"].imshow(bicubic)
    axes["C"].set_xlabel("Bicubic Upscaled")
    axes["D"].imshow(m3_model_upscaled_image)
    axes["D"].set_xlabel("Model Upscaled")
    axes["E"].axis("off")
    axes["F"].axis("off")
    fig.tight_layout()


def plot_crop_comparisons(test_im_rgb):
    low_res_crop, high_res_crop, crop_x, crop_y = random_crop(test_im_rgb.lr[0], test_im_rgb.hr[0], 64, 2)
    print(f"crop_x: {crop_x} crop_y: {crop_y}")
    low_res_crop = low_res_crop[jnp.newaxis, ...]
    high_res_crop = high_res_crop[jnp.newaxis, ...]

    m3_network = models.model.Model(network="M3", should_collapse=True)
    m3_params = jnp.load("../trained_model_parameters/params_M3_300.npz", allow_pickle=True)
    m3_params = m3_params['arr_0'].item(0)
    m3_model_upscaled_image = upscale_image_with_model(low_res_crop, m3_network, m3_params)[0] / 255

    low_res_plot = ycbcr_to_rgb(rgb_to_ycbcr(low_res_crop[0])) / 255
    high_res_plot = ycbcr_to_rgb(rgb_to_ycbcr(high_res_crop[0])) / 255
    bicubic = tf.image.resize(low_res_plot, (2 * low_res_plot.shape[0], 2 * low_res_plot.shape[1]), method="bicubic")

    figure_mosaic = """
    ABC
    DEF
    """

    fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(14, 10))
    axes["A"].imshow(low_res_plot)
    axes["A"].set_xlabel("Low Resolution")
    axes["B"].imshow(high_res_plot)
    axes["B"].set_xlabel("Ground Truth High Resolution")
    axes["C"].imshow(bicubic)
    axes["C"].set_xlabel("Bicubic Upscaled")
    axes["D"].imshow(m3_model_upscaled_image)
    axes["D"].set_xlabel("M3 Model Upscaled")
    axes["E"].axis("off")
    axes["F"].axis("off")


def plot_full_img(test_im_rgb, crop_x, crop_y, crop_size):
    im_width = test_im_rgb.lr[0].shape[1]
    im_height = test_im_rgb.lr[0].shape[0]
    bbox = jnp.array(
        [crop_y / im_height, crop_x / im_width, (crop_y + crop_size) / im_height, (crop_x + crop_size) / im_width])
    bbox = bbox[jnp.newaxis, ...][jnp.newaxis, ...]

    full_low_res = ycbcr_to_rgb(rgb_to_ycbcr(test_im_rgb.lr[0])) / 255
    full_low_res = tf.image.draw_bounding_boxes(full_low_res[jnp.newaxis, ...], bbox, colors=jnp.array([[1.0, 0.0, 0.0]]))[0]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(full_low_res)
    ax.set_xlabel("Full Image")
