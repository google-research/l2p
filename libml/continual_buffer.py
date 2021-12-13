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
"""Replay buffer for continual learning."""

from typing import Sequence

import jax
import ml_collections
import numpy as np


class ReplayBuffer:
  """Replay buffer for continual learning."""

  def __init__(self, continual_config: ml_collections.ConfigDict,
               input_shape: Sequence[int]):
    """Initializes the replay buffer.

    Args:
      continual_config: Configuration for continual learning.
      input_shape: Input shape of the data, e.g., (32, 32, 3) for cifar10
    """
    self.num_tasks = continual_config.num_tasks
    self.num_classes_per_task = continual_config.num_classes_per_task
    self.num_samples_per_class = continual_config.replay.num_samples_per_class
    self.num_samples_per_task = self.num_classes_per_task * self.num_samples_per_class
    self.input_shape = input_shape
    self._buffer_size = np.product(
        [self.num_tasks, self.num_classes_per_task, self.num_samples_per_class])
    # place holder for all images
    self.data = {
        'image': np.zeros((self._buffer_size, *input_shape)),
        'label': np.zeros((self._buffer_size,))
    }
    # specify valid data scope, updated when data goes in!
    self.cursor = 0
    # specify data scope for the old data
    self.old_task_boundary = 0
    # specify index dictionaries for each task
    self.index_dict_list = []

  def gen_batch_index(self, num_total_samples: int, per_device_bs: int):
    """Generates batch indices to save into the buffer.

    Args:
      num_total_samples: Total number of training samples for this task, could
        be obtained by, e.g., class_stats_list[task_id][0].
      per_device_bs: Batch size per devices, could be obtained by
        config.per_device_batch_size

    Returns:
      List of batch id's to obtain
    """
    num_devices = jax.device_count()

    def translator(x):
      x1 = x % num_devices
      x2 = x // num_devices
      return x1, x2

    global_bs = per_device_bs * num_devices
    steps = num_total_samples // global_bs
    valid_range = steps * global_bs
    num_samples_per_task = self.num_samples_per_class * self.num_classes_per_task
    indices = np.random.choice(
        np.arange(valid_range), size=num_samples_per_task, replace=False)
    # construct index dictionary
    index_dict = {i: [] for i in range(steps)}
    for index in indices:
      batch_key = index // global_bs
      batch_res = index % global_bs
      device_id, device_batch_id = translator(batch_res)
      index_dict[batch_key].append((device_id, device_batch_id))
    self.index_dict_list.append(index_dict)
    # make sure that old task boundary is saved
    self.old_task_boundary = self.cursor

  @property
  def cur_size(self):
    return self.cursor

  def gen_class_dict(self):
    self.cur_task_class_dict = {}
    self.full_flag = False
    self.old_task_boundary = self.cursor

  def add_example(self, task_id, batch_id, batch):
    """Adds examples in this batch to the buffer, according to the index dict."""
    index_dict = self.index_dict_list[task_id]
    indices = index_dict[batch_id]
    if indices:
      for index in indices:
        self.data['image'][self.cursor] = batch['image'][index[0], index[1]]
        self.data['label'][self.cursor] = batch['label'][index[0], index[1]]
        self.cursor += 1

  def get_random_batch(self, per_device_bs, include_new_task=True):
    """Returns a random batch according to current valid size."""
    num_devices = jax.device_count()
    global_bs = per_device_bs * num_devices
    # if global batch size > current valid size, we just sample with replacement
    replace = False if self.cursor >= global_bs else True
    if include_new_task:
      range_limit = self.cursor
    else:
      range_limit = self.old_task_boundary
    random_indices = np.random.choice(
        np.arange(range_limit), size=global_bs, replace=replace)
    image = self.data['image'][random_indices]
    label = self.data['label'][random_indices].astype(np.int32)
    image = image.reshape(num_devices, per_device_bs, *self.input_shape)
    label = label.reshape(num_devices, per_device_bs)
    return {'image': image, 'label': label}
