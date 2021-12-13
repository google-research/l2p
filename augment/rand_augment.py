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
# Lint as: python3
"""Random augment."""

from augment import augment_ops
import tensorflow as tf


IMAGENET_AUG_OPS = [
    'AutoContrast',
    'Equalize',
    'Rotate',
    'Posterize',
    'Solarize',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateX',
    'TranslateY',
    'Identity',
    'SolarizeAdd',
    'Invert'  # This two are not common
]

# Levels in this file are assumed to be floats in [0, 1] range
# If you need quantization or integer levels, this should be controlled
# in client code.
MAX_LEVEL = 1.

# Constant which is used when computing translation argument from level
# Defined below
TRANSLATE_CONST = None


def _randomly_negate_tensor(tensor):
  """With 50% prob turn the tensor negative."""
  should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
  final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
  return final_tensor


def _rotate_level_to_arg(level):
  level = (level / MAX_LEVEL) * 30.
  level = _randomly_negate_tensor(level)
  return (level,)


def _enhance_level_to_arg(level):
  return ((level / MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level):
  level = (level / MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance
  level = _randomly_negate_tensor(level)
  return (level,)


def _translate_level_to_arg(level):
  level = (level / MAX_LEVEL) * TRANSLATE_CONST
  # Flip level to negative with 50% chance
  level = _randomly_negate_tensor(level)
  return (level,)


def _posterize_level_to_arg(level):
  return (tf.cast((level / MAX_LEVEL) * 4, tf.int32),)


def _solarize_level_to_arg(level):
  return (tf.cast((level / MAX_LEVEL) * 256, tf.int32),)


def _solarize_add_level_to_arg(level):
  return (tf.cast((level / MAX_LEVEL) * 110, tf.int32),)


def _ignore_level_to_arg(level):
  del level
  return ()


def _divide_level_by_max_level_arg(level):
  return (level / MAX_LEVEL,)


LEVEL_TO_ARG = {
    'AutoContrast': _ignore_level_to_arg,
    'Equalize': _ignore_level_to_arg,
    'Invert': _ignore_level_to_arg,
    'Rotate': _rotate_level_to_arg,
    'Posterize': _posterize_level_to_arg,
    'Solarize': _solarize_level_to_arg,
    'SolarizeAdd': _solarize_add_level_to_arg,
    'Color': _enhance_level_to_arg,
    'Contrast': _enhance_level_to_arg,
    'Brightness': _enhance_level_to_arg,
    'Sharpness': _enhance_level_to_arg,
    'ShearX': _shear_level_to_arg,
    'ShearY': _shear_level_to_arg,
    'TranslateX': _translate_level_to_arg,
    'TranslateY': _translate_level_to_arg,
    'Identity': _ignore_level_to_arg,
    'Blur': _divide_level_by_max_level_arg,
    'Smooth': _divide_level_by_max_level_arg,
    'Rescale': _divide_level_by_max_level_arg,
}


class RandAugment(object):
  """Random augment with fixed magnitude."""

  def __init__(self,
               num_layers=2,
               prob_to_apply=None,
               magnitude=None,
               num_levels=10,
               cutout=True,
               magstd=None,
               size=224):
    """Initialized rand augment.

    Args:
      num_layers: number of augmentation layers, i.e. how many times to do
        augmentation.
      prob_to_apply: probability to apply on each layer. If None then always
        apply.
      magnitude: default magnitude in range [0, 1], if None then magnitude will
        be chosen randomly.
      num_levels: number of levels for quantization of the magnitude.
      cutout: whether to use cutout afterthen.
      magstd: Use normal distribution to sample magnitude. magnitude must be not
        None when magstd is given.
      size: size of image.
    """
    self.num_layers = num_layers
    self.prob_to_apply = (
        float(prob_to_apply) if prob_to_apply is not None else None)
    self.num_levels = int(num_levels) if num_levels else None
    self.level = float(magnitude) if magnitude is not None else None
    self.cutout = cutout
    self.size = size
    self.magstd = magstd

    global TRANSLATE_CONST
    if size == 224:
      TRANSLATE_CONST = 100
    elif size == 128:
      TRANSLATE_CONST = 40
    elif size == 96:
      TRANSLATE_CONST = 30
    elif size == 32:
      TRANSLATE_CONST = 10
    else:
      TRANSLATE_CONST = int(0.3 * size)

    assert TRANSLATE_CONST is not None

  def _get_level(self, rng):
    """Samples magnitude."""
    if self.level is not None and not self.magstd:
      return tf.convert_to_tensor(self.level)

    if self.magstd:
      # Sample level from normal distribution.
      assert self.level, 'When magstd is provided, level has to be given.'
      assert self.num_levels, ('When magstd is provided, num_levels has to be '
                               'given.')
      assert self.num_levels >= self.level
      level = tf.random.stateless_normal(
          shape=[],
          seed=rng,
          mean=self.level,
          stddev=self.magstd,
          dtype=tf.float32)
      # Clip level
      level = tf.math.minimum(
          tf.constant(self.num_levels, level.dtype),
          tf.math.maximum(tf.constant(0, level.dtype), level))
      return tf.cast(level, tf.float32) / self.num_levels

    if self.num_levels is None:
      return tf.random.uniform(shape=[], dtype=tf.float32)
    else:
      level = tf.random.stateless_uniform(
          shape=[], seed=rng, maxval=self.num_levels + 1, dtype=tf.int32)
      return tf.cast(level, tf.float32) / self.num_levels

  def _apply_one_layer(self, rng, image):
    """Applies one level of augmentation to the image."""
    rng_b, rng_c, rng_l = tf.unstack(
        tf.random.experimental.stateless_split(rng, 3))
    level = self._get_level(rng_l)
    branch_fns = []

    for augment_op_name in IMAGENET_AUG_OPS:
      augment_fn = augment_ops.NAME_TO_FUNC[augment_op_name]
      level_to_args_fn = LEVEL_TO_ARG[augment_op_name]

      def _branch_fn(image=image,
                     augment_fn=augment_fn,
                     level_to_args_fn=level_to_args_fn):
        args = [image] + list(level_to_args_fn(level))
        return augment_fn(*args)

      branch_fns.append(_branch_fn)

    branch_index = tf.random.stateless_uniform(
        shape=[], seed=rng_b, maxval=len(branch_fns), dtype=tf.int32)
    aug_image = tf.switch_case(branch_index, branch_fns, default=lambda: image)
    if self.prob_to_apply is not None:
      return tf.cond(
          tf.random.stateless_uniform(shape=[], seed=rng_c, dtype=tf.float32) <
          self.prob_to_apply, lambda: aug_image, lambda: image)
    else:
      return aug_image

  def __call__(self, rng, image, aug_image_key='image'):
    """Calls tf version of randaugment."""
    output_dict = {}
    rngs = tf.unstack(
        tf.random.experimental.stateless_split(rng, self.num_layers + 1))
    rng_cut, rngs = rngs[0], rngs[1:]
    if aug_image_key is not None:
      aug_image = image
      for i in range(self.num_layers):
        aug_image = self._apply_one_layer(rngs[i], aug_image)

      if self.cutout:
        # followd by cutout
        aug_image = augment_ops.cutout(rng_cut, aug_image)
      output_dict[aug_image_key] = aug_image

    if aug_image_key != 'image':
      output_dict['image'] = image

    return output_dict
