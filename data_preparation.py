import tensorflow as tf
import tensorflow_datasets as tfds
from enum import Enum

AUTOTUNE = tf.data.AUTOTUNE


def random_crop(low_res_img, high_res_img, low_res_crop_size, super_res_factor):
    """
    Return random crop from images for data augmentation, as done in original paper.
    Based on example found at: https://keras.io/examples/vision/edsr/
    """
    high_res_crop_size = low_res_crop_size * super_res_factor
    low_res_img_shape = tf.shape(low_res_img)[:2] # Of shape (height, width)

    low_res_width = tf.random.Uniform(shape=(), maxval=low_res_img_shape[1] - low_res_crop_size + 1, dtype=tf.int32)
    low_res_height = tf.random.Uniform(shape=(), maxval=low_res_img_shape[0] - low_res_crop_size + 1, dtype=tf.int32)

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


def dataset_object(dataset_cache, low_res_crop_size, super_res_factor, training=True):
    ds = dataset_cache
    ds = ds.map(
        lambda lowres, highres: random_crop(lowres, highres, low_res_crop_size, super_res_factor),
    )

    # Batching Data
    ds = ds.batch(16)

    if training:
        # Repeating Data, so that cardinality if dataset becomes infinte
        ds = ds.repeat()
    # prefetching allows later images to be prepared while the current image is being processed
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
