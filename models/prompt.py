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
"""Query/Prompt module for selecting task-specific modules.

"""

import logging
from typing import Any, Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jnp.ndarray
Initializer = Callable[[Array, Sequence[int]], Array]


def l2_normalize(x, axis=None, epsilon=1e-12):
  """l2 normalizes a tensor on an axis with numerical stability."""
  square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
  x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
  return x * x_inv_norm


def prepend_prompt(prompt: Array, x_embed: Array) -> Array:
  """Concatenates `prompt` to the beginning of `x_embed`.

  Args:
    prompt: [B, P, H] The prompt.
    x_embed: [B, T, H] The embedded input.

  Returns:
    The input with the prompt concatenated to the front. [B, P + T, H]
  """
  return jnp.concatenate([prompt, x_embed], axis=1)


def reinit_from_sample_of_embeddings(rng: Array, shape: Sequence[int],
                                     embeddings: Array):
  """Initializes prompt by drawing vectors from the embeddings.

  Args:
    rng: The rng seed used in our sampling.
    shape: [P, H] The shape of the prompt variable. shape[0] tells us how many
      vectors to sample.
    embeddings: [V, H] The embeddings to draw vectors from, should be unbatched.
      Returns A sample of the embedding table as a jax array, used for replace
      the original prompt.

  Returns:
    The reinitialized prompt.
  """
  assert len(shape) == 2
  # make sure the embeddings are unbatched
  assert len(embeddings.shape) == 2
  population_size = embeddings.shape[0]
  if embeddings.shape[-1] != shape[-1]:
    raise ValueError(
        "Shape mismatch between the number of features in the "
        f"embeddings: {embeddings.shape[-1]} and the requested prompt shape "
        f"{shape[-1]}.")
  replace = False
  if shape[0] > population_size:
    logging.warning(
        "Prompt Length: %d is larger than the number of vectors "
        "to draw from: %d. Switching to draws with replacement.", shape[0],
        population_size)
    replace = True
  index = jax.random.choice(rng, population_size, (shape[0],), replace)
  prompt = embeddings[index]
  return prompt


def expand_to_batch(x: Array, batch_size: int, axis=0):
  """Expands unbatched `x` to the specified batch_size`."""
  shape = [1 for _ in x.shape]
  shape.insert(axis, batch_size)
  return jnp.tile(jnp.expand_dims(x, axis=axis), shape)


class Projection(nn.Module):
  """Projection layer for prompt features."""
  feature_list: Any
  activation: Any = nn.tanh

  @nn.compact
  def __call__(self, x):
    for features in self.feature_list:
      x = nn.Dense(
          features=features, kernel_init=nn.initializers.xavier_uniform())(
              x)
      x = self.activation(x)
    return x


class Prompt(nn.Module):
  """Promp module including prompt pool and prompt selection mechanism.

  This is the training time version of prompting a model. Calling the injected
  `prompt` module will generate your unbatched prompt. This model then
  replicates it for the batched input and concatenates them together.

  Attributes:
    length: Length of each prompt.
    embedding_key: Way of calculating the key feature. Choose from "mean",
      "max", "mean_max", "cls".
    prompt_init: Initilaizer of the prompt parameters.
    prompt_pool: If use prompt pool or not.
    prompt_key: If use separate prompt key parameters or not.
    pool_size: Size of the prompt pool (number of prompts).
    top_k: Top k prompt to prehend to the input.
    batchwise_prompt: If use the same set or prompts for the same batch,
      following majority vote rule.
    prompt_key_init: Initialization ways of the prompt key parameter.
    num_classes_per_task: Num of classes per task.
    num_layers: int = 1 Number of layers to add prompts
    use_prefix_tune_for_e_prompt: If use prefix-tuning for E-Prompt
    num_heads: Number of heads for MSA
    same_key_value: If use the same key and value for prefix-tuning
    num_tasks: Total number of tasks in the continual learning setting.
  """
  length: int
  embedding_key: str = "mean"
  prompt_init: Initializer = nn.initializers.uniform()
  prompt_pool: bool = False
  prompt_key: bool = False
  pool_size: int = None
  top_k: int = None
  batchwise_prompt: bool = False
  prompt_key_init: str = "zero"
  num_classes_per_task: int = -1
  num_layers: int = 1
  use_prefix_tune_for_e_prompt: bool = False
  num_heads: int = -1
  same_key_value: bool = False
  num_tasks: int = 5

  @nn.compact
  def __call__(self,
               x_embed,
               prompt_mask=None,
               task_id=-1,
               cls_features=None,
               label=None):
    res = dict()
    embed_dim = x_embed.shape[-1]
    if self.prompt_pool:
      if self.use_prefix_tune_for_e_prompt:  # use prefix style
        assert embed_dim % self.num_heads == 0
        if self.same_key_value:
          prompt = self.param("prompt", self.prompt_init,
                              (self.num_layers, 1, self.pool_size, self.length,
                               self.num_heads, embed_dim // self.num_heads))
          prompt = jnp.tile(prompt, (1, 2, 1, 1, 1, 1))
        else:
          prompt = self.param("prompt", self.prompt_init,
                              (self.num_layers, 2, self.pool_size, self.length,
                               self.num_heads, embed_dim // self.num_heads))
      else:
        prompt = self.param(
            "prompt", self.prompt_init,
            (self.num_layers, self.pool_size, self.length, embed_dim))

      # now get key matching part.
      if self.embedding_key == "mean":
        x_embed_mean = jnp.mean(x_embed, axis=1)  # bs, emb
      elif self.embedding_key == "max":
        x_embed_mean = jnp.max(x_embed, axis=1)
      elif self.embedding_key == "mean_max":
        x_embed_mean = jnp.max(x_embed, axis=1) + 2 * jnp.mean(x_embed, axis=1)
      elif self.embedding_key == "cls":
        if cls_features is None:  # just for init
          x_embed_mean = jnp.max(x_embed, axis=1)
        else:
          x_embed_mean = cls_features
      else:
        raise NotImplementedError(
            "Not supported way of calculating embedding keys!")
      # if using learnable prompt keys
      if self.prompt_key:
        key_shape = (self.pool_size, embed_dim)
        if self.prompt_key_init == "zero":
          prompt_key = self.param("key", nn.initializers.zeros, key_shape)
        elif self.prompt_key_init == "uniform":
          prompt_key = self.param("key", nn.initializers.uniform(), key_shape)
      else:
        # only compatible with prompt, not prefix
        prompt_mean = jnp.mean(prompt, axis=[0, 2])  # pool_size, emb
        prompt_key = prompt_mean

      prompt_key_norm = l2_normalize(prompt_key, axis=-1)
      x_embed_norm = l2_normalize(x_embed_mean, axis=-1)

      sim = jnp.matmul(prompt_key_norm, jnp.transpose(
          x_embed_norm))  # pool_size, bs or pool_size, #class, bs
      sim = jnp.transpose(sim)  # bs, pool_size
      (sim_top_k, idx) = jax.lax.top_k(sim, self.top_k)
      if self.batchwise_prompt:
        prompt_id, id_counts = jnp.unique(
            idx, return_counts=True, size=self.pool_size)
        _, major_idx = jax.lax.top_k(id_counts, self.top_k)
        major_prompt_id = prompt_id[major_idx]
        idx = expand_to_batch(major_prompt_id, x_embed.shape[0])

      if prompt_mask is not None:
        idx = prompt_mask  # bs, allowed_size
      res["prompt_idx"] = idx
      batched_key_norm = jnp.take(
          prompt_key_norm, idx, axis=0)  # bs, top_k, embed_dim
      res["selected_key"] = batched_key_norm

      if self.use_prefix_tune_for_e_prompt:
        batched_prompt_raw = jnp.take(
            prompt, idx,
            axis=2)  # num_layers, bs, allowed_size, prompt_len, embed_dim
        num_layers, bs, dual, allowed_size, prompt_len, num_heads, heads_embed_dim = batched_prompt_raw.shape
        batched_prompt = jnp.reshape(batched_prompt_raw,
                                     (num_layers, bs, dual, allowed_size *
                                      prompt_len, num_heads, heads_embed_dim))
      else:
        batched_prompt_raw = jnp.take(
            prompt, idx,
            axis=1)  # num_layers, bs, allowed_size, prompt_len, embed_dim
        num_layers, bs, allowed_size, prompt_len, embed_dim = batched_prompt_raw.shape
        batched_prompt = jnp.reshape(
            batched_prompt_raw,
            (num_layers, bs, allowed_size * prompt_len, embed_dim))
      res["batched_prompt"] = batched_prompt
      res["prompt_key_norm"] = prompt_key_norm
      res["x_embed_norm"] = x_embed_norm
      res["sim"] = sim

      # Put pull_constraint loss calculation inside
      x_embed_norm = x_embed_norm[:, jnp.newaxis, :]  # bs, 1, embed_dim
      sim_pull = batched_key_norm * x_embed_norm
      reduce_sim = jnp.sum(sim_pull) / x_embed.shape[0]
      res["reduce_sim"] = reduce_sim

    else:
      if self.use_prefix_tune_for_e_prompt:  # use prefix style
        assert embed_dim % self.num_heads == 0
        if self.same_key_value:
          prompt = self.param("prompt", self.prompt_init,
                              (self.num_layers, 1, self.length, self.num_heads,
                               embed_dim // self.num_heads))
          prompt = jnp.tile(prompt, (1, 2, 1, 1, 1))
        else:
          prompt = self.param("prompt", self.prompt_init,
                              (self.num_layers, 2, self.length, self.num_heads,
                               embed_dim // self.num_heads))
        batched_prompt = expand_to_batch(prompt, x_embed.shape[0], axis=2)
      else:
        prompt = self.param("prompt", self.prompt_init,
                            (self.num_layers, self.length, embed_dim))
        batched_prompt = expand_to_batch(prompt, x_embed.shape[0], axis=1)

    res["batched_prompt"] = batched_prompt
    return res
