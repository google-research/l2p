# coding=utf-8
# Copyright 2020 The Learning-to-Prompt Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific Learning-to-Prompt governing permissions and
# limitations under the License.
# ==============================================================================
"""Input preprocesses."""

from typing import Any, Callable, Dict, Optional

import ml_collections
from augment import augment_utils
import tensorflow as tf

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
# Constants for configuring config.<name>
RANDOM_ERASING = "randerasing"
AUGMENT = "augment"
MIX = "mix"
COLORJITTER = "colorjitter"

create_mix_augment = augment_utils.create_mix_augment


def resize_small(image: tf.Tensor,
                 size: int,
                 *,
                 antialias: bool = False) -> tf.Tensor:
  """Resizes the smaller side to `size` keeping aspect ratio.

  Args:
    image: Single image as a float32 tensor.
    size: an integer, that represents a new size of the smaller side of an input
      image.
    antialias: Whether to use an anti-aliasing filter when downsampling an
      image.

  Returns:
    A function, that resizes an image and preserves its aspect ratio.
  """
  h, w = tf.shape(image)[0], tf.shape(image)[1]

  # Figure out the necessary h/w.
  ratio = (tf.cast(size, tf.float32) / tf.cast(tf.minimum(h, w), tf.float32))
  h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
  w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)
  image = tf.image.resize(image, [h, w], antialias=antialias)
  return image


def central_crop(image: tf.Tensor, size: int) -> tf.Tensor:
  """Makes central crop of a given size."""
  h, w = size, size
  top = (tf.shape(image)[0] - h) // 2
  left = (tf.shape(image)[1] - w) // 2
  image = tf.image.crop_to_bounding_box(image, top, left, h, w)
  return image


def decode_and_random_resized_crop(image: tf.Tensor, rng,
                                   resize_size: int) -> tf.Tensor:
  """Decodes the images and extracts a random crop."""
  shape = tf.io.extract_jpeg_shape(image)
  begin, size, _ = tf.image.stateless_sample_distorted_bounding_box(
      shape,
      tf.zeros([0, 0, 4], tf.float32),
      seed=rng,
      area_range=(0.05, 1.0),
      min_object_covered=0,  # Don't enforce a minimum area.
      use_image_if_no_bounding_boxes=True)
  top, left, _ = tf.unstack(begin)
  h, w, _ = tf.unstack(size)
  image = tf.image.decode_and_crop_jpeg(image, [top, left, h, w], channels=3)
  image = tf.cast(image, tf.float32) / 255.0
  image = tf.image.resize(image, (resize_size, resize_size))
  return image


def train_preprocess(features: Dict[str, tf.Tensor],
                     crop_size: int = 224) -> Dict[str, tf.Tensor]:
  """Processes a single example for training."""
  image = features["image"]
  # This PRNGKey is unique to this example. We can use it with the stateless
  # random ops in TF.
  rng = features.pop("rng")
  rng, rng_crop, rng_flip = tf.unstack(
      tf.random.experimental.stateless_split(rng, 3))
  image = decode_and_random_resized_crop(image, rng_crop, resize_size=crop_size)
  image = tf.image.stateless_random_flip_left_right(image, rng_flip)
  return {"image": image, "label": features["label"]}


def train_cifar_preprocess(features: Dict[str, tf.Tensor]):
  """Augmentation function for cifar dataset."""
  image = tf.io.decode_jpeg(features["image"])
  image = tf.image.resize_with_crop_or_pad(image, 32 + 4, 32 + 4)
  rng = features.pop("rng")
  rng, rng_crop, rng_flip = tf.unstack(
      tf.random.experimental.stateless_split(rng, 3))
  # Randomly crop a [HEIGHT, WIDTH] section of the image.
  image = tf.image.stateless_random_crop(image, [32, 32, 3], rng_crop)
  # Randomly flip the image horizontally
  image = tf.image.stateless_random_flip_left_right(image, rng_flip)
  image = tf.cast(image, tf.float32) / 255.0
  return {"image": image, "label": features["label"]}


def _check_valid_mean_std(mean, std):
  expected_shape = (1, 1, 3)
  message = "%s shape invalid."
  assert all([a == b for a, b in zip(expected_shape, mean.shape)
             ]), message % "mean"
  assert all([a == b for a, b in zip(expected_shape, std.shape)
             ]), message % "std"


def get_augment_preprocess(
    augment_params: ml_collections.ConfigDict,
    *,
    colorjitter_params: Optional[ml_collections.ConfigDict] = None,
    randerasing_params: Optional[ml_collections.ConfigDict] = None,
    mean: Optional[tf.Tensor] = None,
    std: Optional[tf.Tensor] = None,
    basic_process: Callable[[Dict[str, tf.Tensor]],
                            Dict[str, tf.Tensor]] = train_preprocess,
) -> Callable[[Dict[str, tf.Tensor]], Dict[str, tf.Tensor]]:
  """Creates a custom augmented image preprocess."""
  augmentor = None
  # If augment_params.type is noop/default, we skip.
  if augment_params and augment_params.get(
      "type") and augment_params.type not in ("default", "noop"):
    augmentor = augment_utils.create_augmenter(**augment_params.to_dict())

  jitter = None
  if colorjitter_params and colorjitter_params.type not in ("default", "noop"):
    jitter = augment_utils.create_augmenter(**colorjitter_params.to_dict())

  def train_custom_augment_preprocess(features):

    rng = features.pop("rng")
    rng, rng_aa, rng_re, rng_jt = tf.unstack(
        tf.random.experimental.stateless_split(rng, 4))
    features["rng"] = rng
    outputs = basic_process(features)
    image = outputs["image"]
    # image after basic_process has been normalized to [0,1]
    image = tf.saturate_cast(image * 255.0, tf.uint8)
    if augmentor is not None:
      image = augmentor(rng_aa, image)["image"]
    if jitter is not None:
      image = jitter(rng_jt, image)["image"]
    image = tf.cast(image, tf.float32) / 255.0
    if mean is not None:
      _check_valid_mean_std(mean, std)
      image = (image - mean) / std
    if randerasing_params:
      assert mean is not None, "Random erasing requires normalized images"
      # Perform random erasing after mean/std normalization
      image = augment_utils.create_random_erasing(
          **randerasing_params.to_dict())(rng_re, image)
    outputs["image"] = image
    return outputs

  return train_custom_augment_preprocess


def eval_preprocess(features: Dict[str, tf.Tensor],
                    mean: Optional[tf.Tensor] = None,
                    std: Optional[tf.Tensor] = None,
                    input_size: int = 256,
                    crop_size: int = 224) -> Dict[str, tf.Tensor]:
  """Process a single example for evaluation."""
  image = features["image"]
  assert image.dtype == tf.uint8
  image = tf.cast(image, tf.float32) / 255.0
  # image = resize_small(image, size=int(256 / 224 * input_size))
  # image = central_crop(image, size=input_size)
  image = resize_small(image, size=input_size)  # e.g. 256, 448
  image = central_crop(image, size=crop_size)  # e.g. 224, 384
  if mean is not None:
    _check_valid_mean_std(mean, std)
    image = (image - mean) / std
  return {"image": image, "label": features["label"]}


def cifar_eval_preprocess(
    features: Dict[str, tf.Tensor],
    mean: Optional[tf.Tensor] = None,
    std: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
  """Processes a single example for evaluation for cifar."""
  image = features["image"]
  assert image.dtype == tf.uint8
  image = tf.cast(image, tf.float32) / 255.0
  if mean is not None:
    _check_valid_mean_std(mean, std)
    image = (image - mean) / std
  return {"image": image, "label": features["label"]}


