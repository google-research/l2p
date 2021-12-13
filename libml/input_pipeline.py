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
"""Deterministic input pipeline for various continual learning datasets."""

import functools
import os
from typing import Callable, Dict, Tuple, Union

from absl import logging
from clu import deterministic_data
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from helper.class_stats import get_number_filtered_examples
from libml import preprocess
import tensorflow as tf
import tensorflow_datasets as tfds

Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
Features = Dict[str, Tensor]
RANDOM_ERASING = preprocess.RANDOM_ERASING
AUGMENT = preprocess.AUGMENT
MIX = preprocess.MIX
COLORJITTER = preprocess.COLORJITTER


def preprocess_with_per_batch_rng(ds: tf.data.Dataset,
                                  preprocess_fn: Callable[[Features], Features],
                                  *, rng: jnp.ndarray) -> tf.data.Dataset:
  """Maps batched `ds` using the preprocess_fn and a deterministic RNG per batch.

  This preprocess_fn usually contains data preprcess needs a batch of data, like
  Mixup.

  Args:
    ds: Dataset containing Python dictionary with the features. The 'rng'
      feature should not exist.
    preprocess_fn: Preprocessing function that takes a Python dictionary of
      tensors and returns a Python dictionary of tensors. The function should be
      convertible into a TF graph.
    rng: Base RNG to use. Per example RNGs will be derived from this by folding
      in the example index.

  Returns:
    The dataset mapped by the `preprocess_fn`.
  """
  rng = list(jax.random.split(rng, 1)).pop()

  def _fn(example_index: int, features: Features) -> Features:
    example_index = tf.cast(example_index, tf.int32)
    features["rng"] = tf.random.experimental.stateless_fold_in(
        tf.cast(rng, tf.int64), example_index)
    processed = preprocess_fn(features)
    if isinstance(processed, dict) and "rng" in processed:
      del processed["rng"]
    return processed

  return ds.enumerate().map(
      _fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_dataset_fns(
    config: ml_collections.ConfigDict,
    label_offset: int = 0,
    subsample_rate: int = -1
) -> Tuple[tfds.core.DatasetBuilder, tfds.core.ReadInstruction, Callable[
    [Features], Features], Callable[[Features], Features], str, Union[Callable[
        [Features], Features], None]]:
  """Gets dataset specific functions."""
  # Use config.augment.type to control custom aug vs default aug, it makes sweep
  # parameter setting easier.
  use_custom_process = (
      config.get(AUGMENT) or config.get(RANDOM_ERASING) or
      config.get(COLORJITTER))
  use_batch_process = config.get(MIX)

  label_key = "label"
  image_key = "image"
  if config.dataset.startswith("imagenet"):
    dataset_builder = tfds.builder("imagenet2012")
    train_split_name = f"train[:{subsample_rate}%]" if subsample_rate > 0 else "train"
    train_split = deterministic_data.get_read_instruction_for_host(
        train_split_name, dataset_info=dataset_builder.info)
    test_split_name = f"validation[:{subsample_rate}%]" if subsample_rate > 0 else "validation"

    # If there is resource error during preparation, checkout
    # https://github.com/tensorflow/datasets/issues/1441#issuecomment-581660890
    dataset_builder.download_and_prepare()

    # Default image size is 224, one can use a different one by setting
    # config.input_size. Note that some augmentation also requires specifying
    # input_size through respective config.
    # input_size = config.get("input_size", 224)
    input_size = config.resize_size
    crop_size = config.input_size
    # Create augmentaton fn.
    if use_custom_process:
      # When using custom augmentation, we use mean/std normalization.
      logging.info("Configure augmentation type %s", config.augment.type)
      mean = tf.constant(
          preprocess.IMAGENET_DEFAULT_MEAN, dtype=tf.float32, shape=[1, 1, 3])
      std = tf.constant(
          preprocess.IMAGENET_DEFAULT_STD, dtype=tf.float32, shape=[1, 1, 3])
      basic_preprocess_fn = functools.partial(
          preprocess.train_preprocess, crop_size=crop_size)
      preprocess_fn = preprocess.get_augment_preprocess(
          config.get(AUGMENT),
          colorjitter_params=config.get(COLORJITTER),
          randerasing_params=config.get(RANDOM_ERASING),
          mean=mean,
          std=std,
          basic_process=basic_preprocess_fn)
      eval_preprocess_fn = functools.partial(
          preprocess.eval_preprocess,
          mean=mean,
          std=std,
          input_size=input_size,
          crop_size=crop_size)
    else:
      # we use 0-1 normalization when specified.
      if config.norm_01:
        preprocess_fn = functools.partial(
            preprocess.train_preprocess, crop_size=crop_size)
        eval_preprocess_fn = functools.partial(
            preprocess.eval_preprocess,
            input_size=input_size,
            crop_size=crop_size)
      else:
        basic_preprocess_fn = functools.partial(
            preprocess.train_preprocess, crop_size=crop_size)
        preprocess_fn = preprocess.get_augment_preprocess(
            None,
            None,
            None,
            mean=mean,
            std=std,
            basic_process=basic_preprocess_fn)
        eval_preprocess_fn = functools.partial(
            preprocess.eval_preprocess,
            mean=mean,
            std=std,
            input_size=input_size,
            crop_size=crop_size)
  elif config.dataset.startswith("cifar") or config.dataset.startswith(
      "svhn") or config.dataset.startswith("domainnet"):
    assert config.dataset in ("cifar10", "cifar100", "svhn_cropped",
                              "domainnet/real", "domainnet/painting",
                              "domainnet/clipart", "domainnet/quickdraw",
                              "domainnet/infograph", "domainnet/sketch")

    dataset_builder = tfds.builder(config.dataset)
    dataset_builder.download_and_prepare()

    train_split_name = f"train[:{subsample_rate}%]" if subsample_rate > 0 else "train"
    train_split = deterministic_data.get_read_instruction_for_host(
        train_split_name, dataset_info=dataset_builder.info)
    test_split_name = f"test[:{subsample_rate}%]" if subsample_rate > 0 else "test"

    # When using custom augmentation, we use mean/std normalization.
    if config.dataset == "cifar10":
      mean, std = preprocess.CIFAR10_MEAN, preprocess.CIFAR10_STD
    elif config.dataset == "cifar100":
      mean, std = preprocess.CIFAR100_MEAN, preprocess.CIFAR100_STD
    elif config.dataset == "svhn_cropped" or config.dataset.startswith(
        "domainnet"):
      mean, std = preprocess.CIFAR10_MEAN, preprocess.CIFAR10_STD
    mean = tf.constant(mean, dtype=tf.float32, shape=[1, 1, 3])
    std = tf.constant(std, dtype=tf.float32, shape=[1, 1, 3])
    input_size = config.get("input_size", 32)

    if input_size != 32:
      # Finetune cifar from imagenet pretrained models.
      logging.info("Use %s input size for cifar.", input_size)

      input_size = config.resize_size  # e.g. 448
      crop_size = config.input_size  # e.g. 384

      # Resize small images by a factor of ratio.
      def train_preprocess(features):
        image = tf.io.decode_jpeg(features["image"])
        image = tf.image.resize(image, (input_size, input_size))
        features["image"] = tf.io.encode_jpeg(tf.cast(image, tf.uint8))
        if label_offset > 0:
          features["label"] = features["label"] + int(label_offset)
        return preprocess.train_preprocess(features, crop_size)

      def eval_preprocess(features, mean, std):
        features["image"] = tf.cast(
            tf.image.resize(features["image"], (input_size, input_size)),
            tf.uint8)
        if label_offset > 0:
          features["label"] = features["label"] + int(label_offset)
        return preprocess.eval_preprocess(features, mean, std, input_size,
                                          crop_size)
    else:
      train_preprocess = preprocess.train_cifar_preprocess
      eval_preprocess = preprocess.cifar_eval_preprocess

    if use_custom_process:
      logging.info("Configure augmentation type %s", config.augment.type)
      # The augmentor util uses augment.size to size specific augmentation.
      config.augment.size = config.input_size
      preprocess_fn = preprocess.get_augment_preprocess(
          config.get(AUGMENT),
          colorjitter_params=config.get(COLORJITTER),
          randerasing_params=config.get(RANDOM_ERASING),
          mean=mean,
          std=std,
          basic_process=train_preprocess)
      eval_preprocess_fn = functools.partial(
          eval_preprocess, mean=mean, std=std)
    else:

      if config.norm_01:
        preprocess_fn = train_preprocess
        eval_preprocess_fn = functools.partial(
            eval_preprocess, mean=None, std=None)
      else:
        preprocess_fn = preprocess_fn = preprocess.get_augment_preprocess(
            None,
            colorjitter_params=None,
            randerasing_params=None,
            mean=mean,
            std=std,
            basic_process=train_preprocess)
        eval_preprocess_fn = functools.partial(
            eval_preprocess, mean=mean, std=std)
  else:
    raise ValueError(f"Dataset {config.dataset} does not exist.")

  if use_batch_process:
    logging.info("Configure mix augmentation type %s", config.mix)
    # When config.mix batch augmentation is enabled.
    batch_preprocess_fn = preprocess.create_mix_augment(
        num_classes=dataset_builder.info.features[label_key].num_classes,
        **config.mix.to_dict())
  else:
    batch_preprocess_fn = None


  return (dataset_builder, train_split, preprocess_fn, eval_preprocess_fn,
          test_split_name, batch_preprocess_fn)


def create_datasets(
    config: ml_collections.ConfigDict,
    data_rng,
    filter_fn=None,
    label_offset: int = -1,
    subsample_rate: int = -1
) -> Tuple[tfds.core.DatasetInfo, tf.data.Dataset, tf.data.Dataset]:
  """Creates datasets for training and evaluation.

  Args:
    config: Configuration to use.
    data_rng: PRNGKey for seeding operations in the training dataset.
    filter_fn: Filter function to filter out dataset.
    label_offset: Used to offset the label for each tasks.
    subsample_rate: Used to subsample every task in percentage.

  Returns:
    A tuple with the dataset info, the training dataset and the evaluation
    dataset.
  """
  (dataset_builder, train_split, preprocess_fn, eval_preprocess_fn,
   test_split_name, batch_preprocess_fn) = get_dataset_fns(
       config, label_offset=label_offset, subsample_rate=subsample_rate)
  data_rng1, data_rng2 = jax.random.split(data_rng, 2)
  skip_batching = batch_preprocess_fn is not None
  batch_dims = [jax.local_device_count(), config.per_device_batch_size]
  train_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=train_split,
      rng=data_rng1,
      preprocess_fn=preprocess_fn,
      cache=False,
      decoders={"image": tfds.decode.SkipDecoding()},
      shuffle_buffer_size=config.shuffle_buffer_size,
      batch_dims=batch_dims if not skip_batching else None,
      num_epochs=config.num_epochs,
      shuffle=True,
      filter_fn=filter_fn)

  if batch_preprocess_fn:
    # Perform batch augmentation on each device and them batch devices.
    train_ds = train_ds.batch(batch_dims[-1], drop_remainder=True)
    train_ds = preprocess_with_per_batch_rng(
        train_ds, batch_preprocess_fn, rng=data_rng2)
    for batch_size in reversed(batch_dims[:-1]):
      train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(4)

  options = tf.data.Options()
  options.experimental_external_state_policy = (
      tf.data.experimental.ExternalStatePolicy.WARN)
  train_ds = train_ds.with_options(options)

  if test_split_name:
    eval_split = deterministic_data.get_read_instruction_for_host(
        test_split_name,
        dataset_info=dataset_builder.info,
        drop_remainder=False)
    num_validation_examples = (
        dataset_builder.info.splits[test_split_name].num_examples)
    eval_num_batches = None
    if config.eval_pad_last_batch:
      eval_batch_size = jax.local_device_count() * config.per_device_batch_size
      eval_num_batches = int(
          np.ceil(num_validation_examples / eval_batch_size /
                  jax.process_count()))
    eval_ds = deterministic_data.create_dataset(
        dataset_builder,
        split=eval_split,
        preprocess_fn=eval_preprocess_fn,
        # Only cache dataset in distributed setup to avoid consuming a lot of
        # memory in Colab and unit tests.
        cache=False,
        batch_dims=[jax.local_device_count(), config.per_device_batch_size],
        num_epochs=1,
        shuffle=False,
        pad_up_to_batches=eval_num_batches,
        filter_fn=filter_fn)
  else:
    eval_ds = None
  return dataset_builder.info, train_ds, eval_ds


def class_filter(x, allowed_labels):
  """Filter function to keep only specific classes in a dataset.

  Args:
    x: A batch of data. Has keys "image" and "label".
    allowed_labels: Specific classes to keep. In the format of tf.constant.

  Returns:
    A boolean tf vector indication which samples to keep.
  """
  label = x["label"]
  isallowed = tf.equal(allowed_labels, tf.cast(label, allowed_labels.dtype))
  reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
  return tf.greater(reduced, tf.constant(0.))


def create_continual_datasets(config: ml_collections.ConfigDict, data_rng,
                              task_id: int):
  """Creates a single task of the split-style continual learning tasks, e.g., split CIFAR-100.

  Args:
    config: Configuration to use.
    data_rng: Random seed to generate the dataset.
    task_id: ID of the task to generate

  Returns:
    A tuple with the dataset info, the training dataset, the evaluation
    dataset, class statistics and the class masks.
  """
  if config.reverse_task:
    task_id = config.continual.num_tasks - task_id - 1
  num_tasks = config.continual.num_tasks
  num_classes_per_task = config.continual.num_classes_per_task
  rand_seed = config.continual.rand_seed
  # determine the class order
  class_order = np.arange(num_tasks * num_classes_per_task)
  if rand_seed >= 0:
    class_order = np.random.permutation(class_order)
  allowed_classes = np.arange(task_id * num_classes_per_task,
                              (task_id + 1) * num_classes_per_task)
  class_mask = class_order[allowed_classes]
  allowed_labels = tf.constant(class_order[allowed_classes])
  filter_fn = functools.partial(class_filter, allowed_labels=allowed_labels)
  # get number of examples per dataset (train, test)
  dataset_list = [
      "svhn_cropped", "mnist", "cifar10", "not_mnist", "fashion_mnist"
  ]
  info, train_ds, eval_ds = create_datasets(config, data_rng, filter_fn)
  if config.dataset in dataset_list and config.continual.num_tasks == 1:
    class_stats = [
        info.splits["train"].num_examples, info.splits["test"].num_examples
    ]
  else:
    class_stats = get_number_filtered_examples(config.dataset,
                                               class_order[allowed_classes])
  return info, train_ds, eval_ds, class_stats, class_mask






def gaussian(peak, width, x):
  """Gaussian schedule, credit to: https://arxiv.org/abs/2105.13327."""
  out = jnp.exp(-((x - peak)**2 / (2 * width**2)))
  return out


def gaussian_schedule(rng,
                      num_classes=100,
                      num_tasks=200,
                      step_per_task=5,
                      random_label=False):
  """Returns a schedule where one task blends smoothly into the next."""
  schedule_length = num_tasks * step_per_task  # schedule length in batches
  episode_length = step_per_task  # episode length in batches

  # Each class label appears according to a Gaussian probability distribution
  # with peaks spread evenly over the schedule
  peak_every = schedule_length // num_classes
  width = 50  # width of Gaussian
  peaks = range(peak_every // 2, schedule_length, peak_every)

  schedule = []
  labels = jnp.array(list(range(num_classes)))
  if random_label:
    labels = jax.random.permutation(rng, labels)  # labels in random order

  for ep_no in range(0, schedule_length // episode_length):

    lbls = []
    while not lbls:  # make sure lbls isn't empty
      for j in range(len(peaks)):
        peak = peaks[j]
        # Sample from a Gaussian with peak in the right place
        p = gaussian(peak, width, ep_no * episode_length)
        (rng2, rng) = jax.random.split(rng)
        add = jax.random.bernoulli(rng2, p=p)
        if add:
          lbls.append(int(labels[j]))

    episode = {"label_set": np.array(lbls), "n_batches": episode_length}
    # episode = {'label_set': lbls}
    schedule.append(episode)

  return schedule


def create_gaussian_cifar100(config: ml_collections.ConfigDict, rng):
  """Create sub-dataset as a task."""
  train_ds_list, eval_ds_list, class_stats_list, class_mask_list = [], [], [], []
  # create schedule first
  rng, gaussian_rng = jax.random.split(rng)
  # assert config.continual.num_train_steps_per_task > 0
  schedule = gaussian_schedule(
      gaussian_rng,
      num_classes=100,
      num_tasks=config.continual.num_tasks,
      step_per_task=config.continual.num_train_steps_per_task,
      random_label=False)
  # create the single test set first
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.process_index())
  info, train_ds, eval_ds = create_datasets(config, data_rng)
  eval_ds_list = [eval_ds]
  eval_stats = info.splits["test"].num_examples
  for episode in schedule:
    rng, data_rng = jax.random.split(rng)
    data_rng = jax.random.fold_in(data_rng, jax.process_index())
    allowed_labels = episode["label_set"]
    filter_fn = functools.partial(class_filter, allowed_labels=allowed_labels)

    info, train_ds, eval_ds = create_datasets(
        config,
        data_rng,
        filter_fn=filter_fn,
        subsample_rate=config.subsample_rate)
    # No test set!
    if config.continual.num_train_steps_per_task > 0:
      class_stats = [
          config.continual.num_train_steps_per_task *
          config.per_device_batch_size * jax.device_count(), eval_stats
      ]
    else:
      class_stats = get_number_filtered_examples(config.dataset, allowed_labels)
    class_mask = allowed_labels
    train_ds_list.append(train_ds)
    # eval_ds_list.append(eval_ds)
    class_stats_list.append(class_stats)
    class_mask_list.append(class_mask)
  return rng, train_ds_list, eval_ds_list, class_stats_list, class_mask_list
