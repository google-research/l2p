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
"""Implementation of Vision Transformer in JAX."""

import functools
from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
import ml_collections
from models import prefix_attention
from models import prompt

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x):
    return x


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    posemb_init: positional embedding initializer.
  """

  posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]

  @nn.compact
  def __call__(self, inputs):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      inputs: Inputs to the layer.

    Returns:
      Output tensor with shape `(bs, timesteps, in_dim)`.
    """
    # inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape)
    return inputs + pe


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  dtype: Dtype = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        features=self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            inputs)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        features=actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            x)
    output = nn.Dropout(rate=self.dropout_rate)(
        output, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    inputs: input data.
    mlp_dim: dimension of the mlp on top of attention block.
    dtype: the dtype of the computation (default: float32).
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout for attention heads.
    deterministic: bool, deterministic or not (to apply dropout).
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
  """

  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  prefix_layer: Any = None

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Encoder1DBlock module.

    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.

    Returns:
      output after transformer encoder block.
    """

    # Attention block.
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = prefix_attention.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads,
        prefix=self.prefix_layer)(x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic)

    return x + y


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    num_layers: number of layers
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate in self attention.
  """

  num_layers: int
  mlp_dim: int
  num_heads: int
  prefix: Any
  g_prompt_layer_idx: Any
  prompt: Any
  e_prompt_layer_idx: Any
  use_prefix_tune_for_e_prompt: bool = True
  use_prefix_tune_for_g_prompt: bool = True
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, inputs, *, train):
    """Applies Transformer model on the inputs.

    Args:
      inputs: Inputs to the layer.
      train: Set to `True` when training.

    Returns:
      output of a transformer encoder.
    """
    assert inputs.ndim == 3  # (batch, len, emb)

    x = AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
        name='posembed_input')(
            inputs)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input Encoder
    # counter for prompts
    prompt_counter = -1
    prefix_layer = None
    batched_prompt = None
    for lyr in range(self.num_layers):
      prefix_layer = None
      batched_prompt = None
      if (self.prefix is not None) and (lyr in self.g_prompt_layer_idx):
        if not self.use_prefix_tune_for_g_prompt:
          batched_prompt = self.prefix[lyr]  # len * hiddensize
          batched_prompt = prompt.expand_to_batch(
              batched_prompt, batch_size=x.shape[0])
        else:
          prefix_layer = self.prefix[lyr]
          # batch it here
          prefix_layer = prompt.expand_to_batch(
              prefix_layer, batch_size=x.shape[0], axis=1)
      else:
        prefix_layer = None
      if self.prompt is not None:
        if isinstance(self.e_prompt_layer_idx, int):
          self.e_prompt_layer_idx = [self.e_prompt_layer_idx]
        if lyr in self.e_prompt_layer_idx:
          prompt_counter += 1
          if self.use_prefix_tune_for_e_prompt:
            if prefix_layer is not None:
              # concatenate shared prompt/prefix with this prefix
              prefix_layer = jnp.concatenate(
                  [prefix_layer, self.prompt[prompt_counter]], axis=-3)
            else:
              prefix_layer = self.prompt[prompt_counter]
          else:
            # do the concatenation here
            if batched_prompt is not None:
              batched_prompt = jnp.concatenate(
                  [batched_prompt, self.prompt[prompt_counter]], axis=-2)
              x = prompt.prepend_prompt(batched_prompt, x)
            else:
              batched_prompt = self.prompt[prompt_counter]
              x = prompt.prepend_prompt(batched_prompt, x)

      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          num_heads=self.num_heads,
          prefix_layer=prefix_layer)(
              x, deterministic=not train)
    encoded = nn.LayerNorm(name='encoder_norm')(x)

    return encoded


class VisionTransformer(nn.Module):
  """VisionTransformer with prompts.

  Attributes:
    num_classes: number of total classes.
    patches: A ConfigDict containing patch size.
    transformer: A ConfigDict containing transformer parameters.
    hidden_size: Size of the input embedding feature.
    train: If in training mode or not.
    norm_pre_logits: If normalizing pre-logits or not.
    temperature: Temparature parameter of cosine normalization.
    representation_size: Representation size of the penaltimate layer. Default
      to be None (so we don't have this extra layer).
    classifier: Use which part of the output feature to do classification.
      Choose from 'token', 'gap', 'prompt', 'token+prompt'.
    use_cls_token: If use class token or not.
    prompt_params: Dictionary containing prompt parameters
    reweight_prompt: If add a reweighting layer after prompts. Default to be
      None and deprecated for now.
    num_tasks: Number of tasks in continual learning.
  """

  num_classes: int
  patches: ml_collections.ConfigDict
  transformer: ml_collections.ConfigDict
  hidden_size: int
  train: bool = False
  norm_pre_logits: bool = False
  temperature: float = 1.0
  representation_size: Optional[int] = None
  classifier: str = 'token'
  use_cls_token: bool = True
  prompt_params: Any = None
  reweight_prompt: bool = False
  num_tasks: int = -1
  prefix_params: Any = None
  prompt_contrastive_temp: float = -1.0
  num_classes_per_task: int = -1

  @nn.compact
  def __call__(self,
               inputs,
               prompt_mask=None,
               task_id=-1,
               cls_features=None,
               label=None):
    res_vit = dict()
    x = inputs
    n, h, w, c = x.shape

    # We can merge s2d+emb into a single conv; it's the same.
    x = nn.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding')(
            x)

    # 12.12: init prefix
    use_prefix_tune_for_g_prompt = True
    if self.prefix_params:
      n_layers = self.transformer.num_layers
      n_heads = self.transformer.num_heads
      # prefix_len,
      g_prompt_length = self.prefix_params['g_prompt_length']
      g_prompt_layer_idx = self.prefix_params['g_prompt_layer_idx']
      embedding_size = self.hidden_size // n_heads
      if not self.prefix_params['use_prefix_tune_for_g_prompt']:
        use_prefix_tune_for_g_prompt = False
        prefix = self.param('prefix', nn.initializers.uniform(),
                            (n_layers, g_prompt_length, self.hidden_size))
      else:
        # 1.4: added for the same key and value
        if self.prefix_params['same_key_value']:
          prefix = self.param(
              'prefix', nn.initializers.uniform(),
              (n_layers, 1, g_prompt_length, n_heads, embedding_size))
          prefix = jnp.tile(prefix, (1, 2, 1, 1, 1))
        else:
          prefix = self.param(
              'prefix', nn.initializers.uniform(),
              (n_layers, 2, g_prompt_length, n_heads, embedding_size))
    else:
      prefix = None
      g_prompt_layer_idx = []

    if self.prefix_params:
      if not self.prefix_params['use_prefix_tune_for_g_prompt']:
        total_prompt_len = self.prefix_params['g_prompt_length'] * len(
            self.prefix_params['g_prompt_layer_idx'])

    # Here, x is a grid of embeddings.
    batched_prompt = None
    use_prefix_tune_for_e_prompt = False
    same_key_value_for_pool = False
    e_prompt_layer_idx = []
    if self.transformer is not None:
      n, h, w, c = x.shape
      x = jnp.reshape(x, [n, h * w, c])
      # res_vit["embedding"] = x
      # put it after class token for now
      if self.prompt_params is not None:
        # set up number of layers
        if isinstance(self.prompt_params['e_prompt_layer_idx'], int):
          num_prompted_layers = 1
        else:
          num_prompted_layers = len(self.prompt_params['e_prompt_layer_idx'])
        # set up if using prefix-style prompts or not
        use_prefix_tune_for_e_prompt = self.prompt_params[
            'use_prefix_tune_for_e_prompt']
        if use_prefix_tune_for_e_prompt:
          same_key_value_for_pool = self.prompt_params['same_key_value']
        e_prompt_layer_idx = self.prompt_params['e_prompt_layer_idx']
        # set up number of heads for prefix
        num_heads = self.transformer.num_heads
        if 'prompt_pool' in self.prompt_params:  # pylint: disable=unsupported-membership-test
          prompt_pool_params = self.prompt_params['prompt_pool']
          if prompt_pool_params.initializer == 'normal':
            initializer = nn.initializers.normal()
          # for now we don't have other initilizers besides uniform and normal
          else:
            initializer = nn.initializers.uniform()
          prompt_pool_module = prompt.Prompt(
              length=prompt_pool_params.length,
              embedding_key=prompt_pool_params.embedding_key,
              prompt_init=initializer,
              name='prompt_pool',
              prompt_pool=True,
              prompt_key=prompt_pool_params.prompt_key,
              pool_size=prompt_pool_params.pool_size,
              top_k=prompt_pool_params.top_k,
              batchwise_prompt=prompt_pool_params.batchwise_prompt,
              prompt_key_init=prompt_pool_params.prompt_key_init,
              num_classes_per_task=self
              .num_classes_per_task,
              num_layers=num_prompted_layers,
              use_prefix_tune_for_e_prompt=use_prefix_tune_for_e_prompt,
              num_heads=num_heads,
              num_tasks=self.num_tasks,
          )
          res_vit = prompt_pool_module(
              x,
              prompt_mask,
              task_id=task_id,
              cls_features=cls_features,
              label=label)
        batched_prompt = res_vit['batched_prompt']
        total_prompt_len = 0
        if self.prefix_params:
          if not self.prefix_params['use_prefix_tune_for_g_prompt']:
            total_prompt_len += self.prefix_params['g_prompt_length'] * len(
                self.prefix_params['g_prompt_layer_idx'])
        for key in self.prompt_params:  # pylint: disable=not-an-iterable
          if not use_prefix_tune_for_e_prompt:
            if key == 'prompt_pool':
              # make it multi-layered prompts
              total_prompt_len += self.prompt_params[
                  key].length * self.prompt_params[
                      key].top_k * num_prompted_layers
            elif key == 'shared_prompt' or key == 'task_specific_prompt':
              total_prompt_len += self.prompt_params[
                  key].length * num_prompted_layers

      # If we want to add a class token, add it here.
      if self.use_cls_token:
        cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
        cls = jnp.tile(cls, [n, 1, 1])
        x = jnp.concatenate([cls, x], axis=1)

      x = Encoder(
          name='Transformer',
          prefix=prefix,
          g_prompt_layer_idx=g_prompt_layer_idx,
          prompt=batched_prompt,
          e_prompt_layer_idx=e_prompt_layer_idx,
          use_prefix_tune_for_e_prompt=use_prefix_tune_for_e_prompt,
          use_prefix_tune_for_g_prompt=use_prefix_tune_for_g_prompt,
          **self.transformer)(
              x, train=self.train)

    if self.use_cls_token and self.classifier == 'token':
      if self.prompt_params:
        x = x[:, total_prompt_len]
      else:
        x = x[:, 0]
    elif self.classifier == 'gap':
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
    elif self.classifier == 'prompt':
      x = x[:, 0:total_prompt_len]
      if self.reweight_prompt:
        reweight = self.param('reweight', nn.initializers.uniform(),
                              (total_prompt_len,))
        reweight = nn.softmax(reweight)
        x = jnp.average(x, axis=1, weights=reweight)
      else:
        x = jnp.mean(x, axis=1)
    elif self.use_cls_token and self.prompt_params and self.classifier == 'token+prompt':
      x = x[:, 0:total_prompt_len + 1]
      x = jnp.mean(x, axis=1)
    else:
      raise ValueError(f'Invalid classifier={self.classifier}')
    # Added for utilizing pretrained features
    res_vit['pre_logits'] = x
    if self.representation_size is not None:
      x = nn.Dense(features=self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = IdentityLayer(name='pre_logits')(x)
    if self.norm_pre_logits:
      eps = 1e-10
      x_norm = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)
      x = x / (x_norm + eps)
    x = nn.Dense(
        features=self.num_classes,
        name='head',
        kernel_init=nn.initializers.zeros)(
            x)
    x = x / self.temperature
    res_vit['logits'] = x
    return res_vit


# Mapping model.name -> config.
MODEL_CONFIGS = {}


def _register(get_config):
  config = get_config()
  MODEL_CONFIGS[config.name] = config
  return get_config


@_register
def get_testing_config():
  """Returns the ViT-B/16 configuration."""
  config = ml_collections.ConfigDict()
  # Only used for testing.
  config.name = 'testing'
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_size = 10
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 10
  config.transformer.num_heads = 2
  config.transformer.num_layers = 1
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.1
  config.classifier = 'token'
  config.representation_size = None
  return config


@_register
def get_b16_config():
  """Returns the ViT-B/16 configuration."""
  config = ml_collections.ConfigDict()
  # Name refers to basename in the directory of pretrained models:
  # https://console.cloud.google.com/storage/vit_models/
  config.name = 'ViT-B_16'
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_size = 768
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 3072
  config.transformer.num_heads = 12
  config.transformer.num_layers = 12
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = 'token'
  config.representation_size = None
  return config


@_register
def get_b16_2X2_config():  # pylint: disable=invalid-name
  """Returns the ViT-B/16 configuration."""
  config = ml_collections.ConfigDict()
  # Name refers to basename in the directory of pretrained models:
  # https://console.cloud.google.com/storage/vit_models/
  config.name = 'ViT-B_16_2X2'
  config.patches = ml_collections.ConfigDict({'size': (2, 2)})
  config.hidden_size = 768
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 3072
  config.transformer.num_heads = 12
  config.transformer.num_layers = 12
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = 'token'
  config.representation_size = None
  return config


@_register
def get_b16_4X4_config():  # pylint: disable=invalid-name
  """Returns the ViT-B/16 configuration."""
  config = ml_collections.ConfigDict()
  # Name refers to basename in the directory of pretrained models:
  # https://console.cloud.google.com/storage/vit_models/
  config.name = 'ViT-B_16_4X4'
  config.patches = ml_collections.ConfigDict({'size': (4, 4)})
  config.hidden_size = 768
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 3072
  config.transformer.num_heads = 12
  config.transformer.num_layers = 12
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = 'token'
  config.representation_size = None
  return config


@_register
def get_b32_config():
  """Returns the ViT-B/32 configuration."""
  config = get_b16_config()
  config.name = 'ViT-B_32'
  config.patches.size = (32, 32)
  return config


@_register
def get_l16_config():
  """Returns the ViT-L/16 configuration."""
  config = ml_collections.ConfigDict()
  # Name refers to basename in the directory of pretrained models:
  # https://console.cloud.google.com/storage/vit_models/
  config.name = 'ViT-L_16'
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_size = 1024
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 4096
  config.transformer.num_heads = 16
  config.transformer.num_layers = 24
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.1
  config.classifier = 'token'
  config.representation_size = None
  return config


@_register
def get_l32_config():
  """Returns the ViT-L/32 configuration."""
  config = get_l16_config()
  config.transformer.dropout_rate = 0.0
  # Name refers to basename in the directory of pretrained models:
  # https://console.cloud.google.com/storage/vit_models/
  config.name = 'ViT-L_32'
  config.patches.size = (32, 32)
  return config


@_register
def get_h14_config():
  """Returns the ViT-H/14 configuration."""
  config = ml_collections.ConfigDict()
  # Name refers to basename in the directory of pretrained models:
  # https://console.cloud.google.com/storage/vit_models/
  config.name = 'ViT-H_14'
  config.patches = ml_collections.ConfigDict({'size': (14, 14)})
  config.hidden_size = 1280
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 5120
  config.transformer.num_heads = 16
  config.transformer.num_layers = 32
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.1
  config.classifier = 'token'
  config.representation_size = None
  return config


@_register
def get_s16_config():
  """Returns the ViT-S/16 configuration."""
  config = ml_collections.ConfigDict()
  config.name = 'ViT-S_16'
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_size = 384
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 1536
  config.transformer.num_heads = 6
  config.transformer.num_layers = 12
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = 'token'
  config.representation_size = None
  return config


def create_model(name, config):
  """Creates model partial function."""
  # del config
  if name not in MODEL_CONFIGS:
    raise ValueError(f'Model {name} does not exist.')
  model_config = MODEL_CONFIGS[name]
  model_config = dict(model_config)
  model_config.pop('name')
  # add pre logits normalization or not
  if config.get('norm_pre_logits'):
    model_config['norm_pre_logits'] = config.norm_pre_logits
  if config.get('temperature'):
    model_config['temperature'] = config.temperature
  if config.get('use_e_prompt'):
    prompt_params = {}
    # Specify which layer the prompt should be add on
    prompt_params['e_prompt_layer_idx'] = config.get('e_prompt_layer_idx')
    # Using prefix-tuning for E-Prompt
    prompt_params['use_prefix_tune_for_e_prompt'] = config.get('use_prefix_tune_for_e_prompt')
    # If using the same key and value in prefix
    prompt_params['same_key_value'] = config.get('same_key_value_for_pool')
    if config.prompt_pool:
      prompt_params['prompt_pool'] = config.prompt_pool_param
    model_config['prompt_params'] = prompt_params

  if config.get('use_g_prompt'):
    prefix_params = {}
    prefix_params['g_prompt_length'] = config.g_prompt_length
    prefix_params['g_prompt_layer_idx'] = config.g_prompt_layer_idx
    prefix_params['same_key_value'] = config.get('same_key_value_for_shared')
    prefix_params['use_prefix_tune_for_g_prompt'] = config.get(
        'use_prefix_tune_for_g_prompt')
    model_config['prefix_params'] = prefix_params

  model_config['use_cls_token'] = config.get('use_cls_token')
  if config.get('vit_classifier'):
    model_config['classifier'] = config.vit_classifier
  if config.get('reweight_prompt'):
    model_config['reweight_prompt'] = config.reweight_prompt
  if config.get('continual'):
    model_config['num_tasks'] = config.continual.num_tasks
    model_config['num_classes_per_task'] = config.continual.num_classes_per_task
  model_config = ml_collections.ConfigDict(model_config)
  return functools.partial(VisionTransformer, **model_config), model_config


def create_original_vit(name):
  if name not in MODEL_CONFIGS:
    raise ValueError(f'Model {name} does not exist.')
  model_config = MODEL_CONFIGS[name]
  model_config = dict(model_config)
  model_config.pop('name')
  model_config = ml_collections.ConfigDict(model_config)
  return functools.partial(VisionTransformer, **model_config), model_config
