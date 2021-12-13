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
from typing import Any, Callable, Optional, Tuple, Dict

import flax.linen as nn
import jax.numpy as jnp
import ml_collections

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
    x = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads)(x, x)
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
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          num_heads=self.num_heads)(
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
  prompt_params: Dict[str, Any] = None
  reweight_prompt: bool = False
  num_tasks: int = -1

  @nn.compact
  def __call__(self, inputs, prompt_mask=None, task_id=-1, cls_features=None):
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

    # Here, x is a grid of embeddings.
    if self.transformer is not None:
      n, h, w, c = x.shape
      x = jnp.reshape(x, [n, h * w, c])
      # res_vit["embedding"] = x
      # put it after class token for now
      if self.prompt_params is not None:
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
              num_tasks=self.num_tasks)  # 9.6: added for getting cls features
          res_prompt = prompt_pool_module(
              x, prompt_mask, task_id=task_id, cls_features=cls_features)
          x = res_prompt['prompted_embedding']
          # For debugging purpose
          res_vit['sim'] = res_prompt['sim']
          res_vit['prompt_norm'] = res_prompt['prompt_norm']
          res_vit['x_embed_norm'] = res_prompt['x_embed_norm']
          res_vit['prompted_embedding'] = res_prompt['prompted_embedding']
          res_vit['prompt_idx'] = res_prompt['prompt_idx']
          res_vit['selected_key'] = res_prompt['selected_key']
          res_vit['reduce_sim'] = res_prompt['reduce_sim']
        # calculate the length of all prompts
        total_prompt_len = 0
        for key in self.prompt_params:  # pylint: disable=not-an-iterable
          if key == 'prompt_pool':
            total_prompt_len += (
                self.prompt_params[key].length * self.prompt_params[key].top_k)
          else:
            total_prompt_len += self.prompt_params[key].length

      # If we want to add a class token, add it here.
      if self.use_cls_token:
        cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
        cls = jnp.tile(cls, [n, 1, 1])
        x = jnp.concatenate([cls, x], axis=1)

      x = Encoder(name='Transformer', **self.transformer)(x, train=self.train)

    if self.use_cls_token and self.classifier == 'token':
      x = x[:, 0]
    elif self.classifier == 'gap':
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
    elif self.prompt_params and self.classifier == 'prompt':
      x = x[:, 1:(
          1 +
          total_prompt_len)] if self.use_cls_token else x[:, 0:total_prompt_len]
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
    # 9.6 added for utilizing pretrained features
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
  if config.use_prompt:
    # 8.3: refactored the prompt parts to be shared and task-specific
    prompt_params = {}
    if config.prompt_pool:
      prompt_params['prompt_pool'] = config.prompt_pool_param
    model_config['prompt_params'] = prompt_params
  if config.use_cls_token:
    model_config['use_cls_token'] = config.use_cls_token
  if config.vit_classifier:
    model_config['classifier'] = config.vit_classifier
  if config.get('reweight_prompt'):
    model_config['reweight_prompt'] = config.reweight_prompt
  model_config['num_tasks'] = config.continual.num_tasks
  model_config = ml_collections.ConfigDict(model_config)
  return functools.partial(VisionTransformer, **model_config), model_config


def create_original_vit(name):
  """Creates original ViT for key feature generation."""
  if name not in MODEL_CONFIGS:
    raise ValueError(f'Model {name} does not exist.')
  model_config = MODEL_CONFIGS[name]
  model_config = dict(model_config)
  model_config.pop('name')
  model_config = ml_collections.ConfigDict(model_config)
  return functools.partial(VisionTransformer, **model_config), model_config
