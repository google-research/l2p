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
"""General utility functions."""

import functools
import os
import time
from typing import Any, Dict, Sequence

from absl import logging
from clu import checkpoint
from clu import platform
import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from libml import utils_vit
import scipy


# get ViT specific loading

POS_EMBED = "PositionEmbedding"  # Match the class name of PositionEmbedding
HEAD = "Dense"




def compute_flops(model_cls: Any,
                  variables: Dict[str, Any],
                  input_shape: Sequence[int],
                  fuse_multiply_add: bool = True) -> str:
  """Performs static analysis of the graph to compute theoretical FLOPs."""

  if input_shape[0] != 1:
    raise ValueError("FLOP test requires batch size dim is 1.")
  model = model_cls(train=False)

  def apply_fn(x):
    return model.apply(variables, x, mutable=False)

  model_input = jnp.ones(input_shape, dtype=jnp.float32)
  # jax.xla_computation must accept a function that takes input argument only.
  m = jax.xla_computation(apply_fn)(model_input).as_hlo_module()
  client = jax.lib.xla_bridge.get_backend()
  analysis = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, m)  # pylint: disable=protected-access
  flops = analysis["flops"]
  if fuse_multiply_add:
    flops = flops / 2
  gflops = flops / (10**9)
  logging.info("Module: GFLOPs %0.3f for input shape %s", gflops, input_shape)
  message = "GFLOPS: %0.3f" % gflops
  return message


def log_throughput(model_cls: Any,
                   variables: Dict[str, Any],
                   input_shape: Sequence[int],
                   iterations: int = 500) -> str:
  """Log throughput of models."""

  model = model_cls(train=False)

  inputs = jnp.ones(input_shape, jnp.float32)
  batch_size = inputs.shape[0]
  logging.info("Start to compute throughput for input %s...", input_shape)

  apply_fn = jax.jit(functools.partial(model.apply, mutable=False))
  # Let it compile first with zombie runs.
  for _ in range(10):
    y = apply_fn(variables, inputs)

  start = time.time()
  for _ in range(iterations):
    y = apply_fn(variables, inputs)
  y.block_until_ready()
  total_time = time.time() - start

  logging.info("Cuda time cost per iteration %.3f", total_time / iterations)
  message = "Throughput: %.3f image/s" % (iterations * batch_size / total_time)
  logging.info(message)
  return message


def cosine_decay(lr: float, step: float, total_steps: int):
  ratio = jnp.maximum(0., step / total_steps)
  mult = 0.5 * (1. + jnp.cos(jnp.pi * ratio))
  return mult * lr


def linear_decay(lr: float, step: float, total_steps: int):
  ratio = jnp.maximum(0., step / total_steps)
  return lr * (1 - ratio)


def get_learning_rate(step: int,
                      *,
                      base_learning_rate: float,
                      steps_per_epoch: int,
                      num_epochs: int,
                      schedule: str = "cosine",
                      warmup_epochs: int = 5,
                      min_learning_rate: float = 0.):
  """Cosine learning rate schedule."""

  logging.info(
      "get_learning_rate(step=%s, base_learning_rate=%s, steps_per_epoch=%s, num_epochs=%s",
      step, base_learning_rate, steps_per_epoch, num_epochs)
  if steps_per_epoch <= 0:
    raise ValueError(f"steps_per_epoch should be a positive integer but was "
                     f"{steps_per_epoch}.")
  if warmup_epochs >= num_epochs:
    raise ValueError(f"warmup_epochs should be smaller than num_epochs. "
                     f"Currently warmup_epochs is {warmup_epochs}, "
                     f"and num_epochs is {num_epochs}.")
  epoch = step / steps_per_epoch
  if schedule == "cosine":
    lr = cosine_decay(base_learning_rate, epoch - warmup_epochs,
                      num_epochs - warmup_epochs)
  elif schedule == "linear":
    lr = linear_decay(base_learning_rate, epoch - warmup_epochs,
                      num_epochs - warmup_epochs)
  elif schedule == "constant":
    lr = jnp.array(base_learning_rate)
  warmup = jnp.minimum(1., epoch / warmup_epochs)
  return jnp.where(warmup < 1, lr * warmup,
                   jnp.maximum(lr * warmup, min_learning_rate))


def _reshape_position_embeddings(pa: jnp.ndarray, ratio: float) -> jnp.ndarray:
  """Resizes position embeddings with scipy zoom like ViT."""

  b, n, s, d = pa.shape
  h = w = int(np.sqrt(s))
  # Two dimension spline interpolation.
  pa = jnp.reshape(pa, (b, n, h, w, d))
  newh = neww = int(jnp.ceil(h * ratio))
  pa_new_numpy = scipy.ndimage.zoom(
      np.array(pa), (1, 1, newh / h, neww / w, 1), order=1)
  pa_new = jax.numpy.asarray(pa_new_numpy)
  pa_new = jnp.reshape(pa_new, (b, n, newh * neww, d))
  return pa_new


def _load_and_custom_init_checkpoint(init_state: Any,
                                     checkpoint_path: str,
                                     *,
                                     resize_posembed: float = 1.0,
                                     reinit_head: str = None) -> Any:
  """Load checkpoint for finetuing task."""

  def _find_var_names(s):
    return [i for i in s.keys()]

  logging.info("Load finetune checkpoint from %s", checkpoint_path)
  # 1) Copy model params init_param_dict.
  state = checkpoint.load_state_dict(os.path.split(checkpoint_path)[0])
  init_param_dict = state["optimizer"]["target"]
  state_params = flax.core.freeze(init_param_dict)

  if resize_posembed != 1:
    # resize_posembed represents the image size ratio (new size / orignal
    # size in the checkpoint).
    # 2) Resize POS_EMBED variables and update to init_param_dict
    for pkey in init_param_dict.keys():
      # POS_EMBED is assumed to exist in the first level of init_param_dict.
      if POS_EMBED in pkey:
        # Find variable name for POS_EMBED.
        var_names = _find_var_names(init_param_dict[pkey])
        assert len(var_names) == 1
        var_name = var_names[0]
        pa = state_params[pkey][var_name]
        pa_new = _reshape_position_embeddings(pa, resize_posembed)
        init_param_dict[pkey][var_name] = pa_new
        pa_expected_shape = init_state.optimizer.target[pkey][var_name].shape
        assert jnp.array_equal(pa_expected_shape, pa_new.shape)
        logging.info("Reshape %s.%s from %s to %s", pkey, var_name, pa.shape,
                     pa_new.shape)
  if reinit_head:
    count = 1
    # 3) Re-init classification head parameters.
    for pkey in init_param_dict.keys():
      # kernel/bias are assumed to exist in the first level of init_param_dict.
      if HEAD in pkey:
        var_names = _find_var_names(init_param_dict[pkey])
        for var_name in var_names:
          count += 1
          pa = state_params[pkey][var_name]
          if reinit_head == "zero_all":
            pa_new = jnp.zeros_like(init_state.optimizer.target[pkey][var_name])
          else:
            raise NotImplementedError(
                f"reinit_head mode {reinit_head} not found.")
          init_param_dict[pkey][var_name] = pa_new
          logging.info("Zero init %s.%s (%s)", pkey, var_name, pa_new.shape)
    assert count, "Does not found head parameters"

  # 4): Copy model params to init_state.
  optimizer = init_state.optimizer.replace(
      target=flax.core.freeze(init_param_dict))
  init_state = init_state.replace(optimizer=optimizer)
  return init_state


def load_and_custom_init_checkpoint(*, config: ml_collections.ConfigDict,
                                    init_state: Any) -> Any:
  """Load checkpoint for continual learning."""
  model_type = config.model_name
  checkpoint_path = config.init_checkpoint
  if model_type.startswith("ViT"):
    restored_params = utils_vit.load_pretrained(
        pretrained_path=checkpoint_path,
        init_params=init_state.optimizer.target,
        model_config=config.model_config)
    # Copy model params to init_state.
    optimizer = init_state.optimizer.replace(target=restored_params)
    init_state = init_state.replace(optimizer=optimizer)
    return init_state
  elif model_type.startswith("resnet"):
    state = checkpoint.load_state_dict(config.init_checkpoint)
    loaded_param_dict = state["optimizer"]["target"]
    loaded_model_state = state["model_state"]
    # we should always change the classification head
    loaded_param_dict["head"]["kernel"] = init_state.optimizer.target["head"][
        "kernel"]
    loaded_param_dict["head"]["bias"] = init_state.optimizer.target["head"][
        "bias"]
    optimizer = init_state.optimizer.replace(
        target=flax.core.freeze(loaded_param_dict))
    init_state = init_state.replace(
        optimizer=optimizer, model_state=loaded_model_state)
    return init_state


def transfer_weights(config, param_dict, task_id, kernel_only=True):
  """Initialize new task classificatier with average of old tasks."""

  param_dict = flax.core.unfreeze(param_dict)
  # feature dim * num_classes
  num_classes_per_task = config.continual.num_classes_per_task
  kernel_old_tasks = param_dict["head"]["kernel"][:, :task_id *
                                                  num_classes_per_task]
  kernel_old_tasks = jnp.reshape(
      kernel_old_tasks, (kernel_old_tasks.shape[0], -1, num_classes_per_task))
  mean_kernel_old_tasks = jnp.mean(kernel_old_tasks, axis=-2)
  param_dict["head"]["kernel"] = (
      param_dict["head"]["kernel"]
      .at[:, task_id * num_classes_per_task:(task_id + 1) *
          num_classes_per_task]
      .set(mean_kernel_old_tasks))
  if not kernel_only:
    bias_old_tasks = param_dict["head"]["bias"][:task_id * num_classes_per_task]
    bias_old_tasks = jnp.reshape(bias_old_tasks, (-1, num_classes_per_task))
    mean_bias_old_tasks = jnp.mean(bias_old_tasks, axis=-2)
    param_dict["head"]["bias"] = (
        param_dict["head"]["bias"]
        .at[task_id * num_classes_per_task:(task_id + 1) *
            num_classes_per_task]
        .set(mean_bias_old_tasks))
  return flax.core.freeze(param_dict)


def weight_norm(param_dict, eps=1e-10):
  """Apply weight normalization to the last linear layer."""

  param_dict = flax.core.unfreeze(param_dict)
  kernel = param_dict["head"]["kernel"]
  kernel_norm = jnp.linalg.norm(kernel, ord=2, axis=0, keepdims=True)
  kernel = kernel / (eps + kernel_norm)
  param_dict["head"]["kernel"] = kernel
  return flax.core.freeze(param_dict)


def replace_cls(param_dict, cls):
  """Replace class token."""

  param_dict = flax.core.unfreeze(param_dict)
  old_cls = param_dict["cls"]
  param_dict["cls"] = cls
  return flax.core.freeze(param_dict), old_cls


def replace_prompt(param_dict, prompt_para):
  """Replace task-specific prompt."""

  param_dict = flax.core.unfreeze(param_dict)
  old_prompt_para = param_dict["task_specific_prompt"]["prompt"]
  param_dict["task_specific_prompt"]["prompt"] = prompt_para
  return flax.core.freeze(param_dict), old_prompt_para


def replace_prompt_weight(param_dict, prompt_weight_para):
  """Replace class token."""

  param_dict = flax.core.unfreeze(param_dict)
  old_prompt_weight_para = param_dict["reweight"]
  param_dict["reweight"] = prompt_weight_para
  return flax.core.freeze(param_dict), old_prompt_weight_para


def replace_prompt_pool(param_dict, prompt_pool_para):
  """Replace class token."""

  param_dict = flax.core.unfreeze(param_dict)
  old_prompt_pool_para = param_dict["prompt_pool"]["prompt"]
  param_dict["prompt_pool"]["prompt"] = prompt_pool_para
  return flax.core.freeze(param_dict), old_prompt_pool_para


def replace_prompt_key(param_dict, prompt_key_para):
  """Replace class token."""

  param_dict = flax.core.unfreeze(param_dict)
  old_prompt_key_para = param_dict["prompt_pool"]["key"]
  param_dict["prompt_pool"]["key"] = prompt_key_para
  return flax.core.freeze(param_dict), old_prompt_key_para


def state_with_new_param(state, param_dict):
  """Return a new state with new parameters."""

  optimizer = state.optimizer.replace(target=param_dict)
  state = state.replace(optimizer=optimizer)
  return state


def get_embedding_params(param_dict):
  """Get the parameters of an embedding layer."""

  embedding = param_dict["embedding"]
  embedding_params = {"embedding": embedding}
  return flax.core.freeze(embedding_params)

