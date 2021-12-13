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
"""Helper code which creates augmentations."""

import functools

from absl import logging
from augment import augment_ops
from augment.rand_augment import RandAugment
import tensorflow as tf


def _get_augmenter_type_and_args(**kwargs):
  """Extracts augmenter type and args from **kwargs dict."""
  augment_type = kwargs['type'].lower()
  augment_args = {}
  for k, v in kwargs.items():
    if k.startswith(augment_type + '_'):
      augment_args[k[len(augment_type) + 1:]] = v
    if k == 'size':
      augment_args[k] = v
  logging.info('Using augmentation %s with parameters %s', augment_type,
               augment_args)
  return augment_type, augment_args


def create_random_erasing(erase_prob):
  """Creates random erasing function."""
  return functools.partial(augment_ops.random_erasing, erase_prob=erase_prob)


def create_augmenter(**kwargs):
  """Creates augmenter for supervised task based on hyperparameters dict.

  Args:
    **kwargs: dictionary augment_type and augmenter arguments.

  Returns:
    augmenter_state: class representing augmenter state or None for stateless
      augmnenter
    sup_augmenter: callable which performs augmentation of the data
  """
  augment_type, augment_args = _get_augmenter_type_and_args(**kwargs)
  if augment_type == 'randaugment':
    augmenter = RandAugment(**augment_args)
    return augmenter
  elif augment_type == 'colorjitter':

    def base_augmenter(rng, x):
      del rng
      return {'image': augment_ops.color_map_fn(x, **augment_args)}

    return base_augmenter
  else:
    raise ValueError('Invalid augmentation type {0}'.format(augment_type))


def create_mix_augment(num_classes,
                       smoothing=0.,
                       mixup_alpha=0.8,
                       cutmix_alpha=1.0,
                       prob_to_apply=1.0):
  """Creates mix style augmentations."""

  def augment_fn(features):
    images, labels = features['image'], features['label']
    assert len(images.shape) == 4, 'Input must be batched'
    oh_labels = tf.cast(tf.one_hot(labels, num_classes), tf.float32)
    rng = features.pop('rng')
    cutmix_rng, mixup_rng, branch_rng, apply_rng = tf.unstack(
        tf.random.experimental.stateless_split(rng, 4))

    branch_fns = []
    # Add mixup function
    if mixup_alpha:

      def _mixup():
        return augment_ops.batch_mixup(mixup_rng, images, oh_labels,
                                       mixup_alpha, smoothing)

      branch_fns.append(_mixup)

    # Add cutmix function
    if cutmix_alpha:

      def _cutmix():
        return augment_ops.batch_cutmix(cutmix_rng, images, oh_labels,
                                        cutmix_alpha, smoothing)

      branch_fns.append(_cutmix)
    branch_index = tf.random.stateless_uniform(
        shape=[], seed=branch_rng, maxval=len(branch_fns), dtype=tf.int32)
    aug_image, aug_labels = tf.switch_case(branch_index, branch_fns)
    augmented_outputs = {'image': aug_image, 'label': aug_labels}
    origin_outputs = {'image': images, 'label': oh_labels}

    if prob_to_apply == 0:
      return origin_outputs
    elif prob_to_apply < 1.0:
      return tf.cond(
          tf.random.stateless_uniform(
              shape=[], seed=apply_rng, dtype=tf.float32) < prob_to_apply,
          lambda: augmented_outputs, lambda: origin_outputs)
    else:
      return augmented_outputs

  return augment_fn
