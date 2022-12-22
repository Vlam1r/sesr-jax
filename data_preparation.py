import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from typing import Tuple, List, Iterable, NamedTuple

AUTOTUNE = tf.data.AUTOTUNE
DATA_DIR = "data/"


class Batch(NamedTuple):
    hr: np.ndarray
    lr: np.ndarray


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


def get_dataset(dataset_name: str,
                batch_size: int = 32,
                low_res_crop_size: int = 64,
                super_res_factor: int = 2,
                num_crops_per_image: int = 64) -> Tuple[Iterable[np.ndarray], Iterable[np.ndarray]]:
    """
    :param dataset_name: name of TensorFlow dataset to use e.g. div2k/bicubic_x2
    :param batch_size: batch size to use for training dataset
    :param low_res_crop_size: size of cropped patches of low res images (e.g. 64 will extract patches of size 64x64)
    :param super_res_factor: factor of super-resolution, for div2k/bicubic_x2 this is 2
    :param num_crops_per_image: number of cropped patches to extract per image
    :return: training dataset, validation dataset
    """
    dataset_dir = DATA_DIR
    ds_train, ds_val = tfds.load(dataset_name, split=["train", "validation"], shuffle_files=True, data_dir=dataset_dir)
    random_crops_fn = partial(get_random_crops, low_res_crop_size=low_res_crop_size, super_res_factor=super_res_factor,
                              num_crops=num_crops_per_image)
    ds_train = ds_train.map(remove_dict).map(random_crops_fn).unbatch().shuffle(buffer_size=10000).batch(batch_size) # TODO: Prefetch?
    ds_val = ds_val.map(remove_dict).map(random_crops_fn).unbatch().batch(10)


    # TODO this should be YCbCr -> Y, not RGB -> R
    ds_train = ds_train.map(lambda lr, hr: (lr[:,:,:,0], hr[:,:,:,0]))
    ds_val = ds_val.map(lambda lr, hr: (lr[:,:,:,0], hr[:,:,:,0]))

    ds_train = ds_train.map(lambda lr, hr: Batch(lr=lr, hr=hr))
    ds_val = ds_val.map(lambda lr, hr: Batch(lr=lr, hr=hr))

    return iter(tfds.as_numpy(ds_train)), iter(tfds.as_numpy(ds_val))


if __name__ == "__main__":
    a, b = get_dataset("div2k/bicubic_x2")

