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
"""Model training loss utilities."""

import flax.linen as nn
import jax
import jax.numpy as jnp


def cross_entropy_loss(*, logits, labels):
  logp = jax.nn.log_softmax(logits)
  loglik = jnp.take_along_axis(logp, labels[:, None], axis=1)
  return -loglik


def apply_label_smoothing(one_hot_targets, label_smoothing):
  """Apply label smoothing to the one-hot targets.

  Applies label smoothing such that the on-values are transformed from 1.0 to
  `1.0 - label_smoothing + label_smoothing / num_classes`, and the off-values
  are transformed from 0.0 to `label_smoothing / num_classes`.
  https://arxiv.org/abs/1512.00567

  Note that another way of performing label smoothing (which we don't use here)
  is to take `label_smoothing` mass from the on-values and distribute it to the
  off-values; in other words, transform the on-values to `1.0 - label_smoothing`
  and the  off-values to `label_smoothing / (num_classes - 1)`.
  http://jmlr.org/papers/v20/18-789.html


  Args:
    one_hot_targets: One-hot targets for an example, a [batch, ..., num_classes]
      float array.
    label_smoothing: float; A scalar in [0, 1] used to smooth the labels.

  Returns:
    A float array of the same shape as `one_hot_targets` with smoothed label
    values.
  """
  on_value = 1.0 - label_smoothing
  num_classes = one_hot_targets.shape[-1]
  off_value = label_smoothing / num_classes
  one_hot_targets = one_hot_targets * on_value + off_value
  return one_hot_targets


def apply_weights(output, weights):
  """Applies given weights of the inputs in the minibatch to outputs.

  Note that weights can be per example (i.e. of shape `[batch_size,]`) or per
  pixel/token (i.e. of shape `[batch_size, height, width]` or
  `[batch_size, len]`) so we need to broadcast it to the output shape.

  Args:
    output: nd-array; Computed output, which can be loss or the correctly
      classified examples, etc.
    weights: nd-array; Weights of inputs in the batch, which can be None or
      array of shape [batch, ...].

  Returns:

  """
  desired_weights_shape = weights.shape + (1,) * (output.ndim - weights.ndim)
  weights = jax.lax.broadcast_in_dim(
      weights,
      shape=desired_weights_shape,
      broadcast_dimensions=tuple(range(weights.ndim)))
  # scale the outputs with weights
  return output * weights


def weighted_unnormalized_softmax_cross_entropy(logits,
                                                one_hot_targets,
                                                weights=None,
                                                label_smoothing=None,
                                                label_weights=None,
                                                logits_normalized=False):
  """Computes weighted softmax cross entropy give logits and targets.

  This computes sum_(x,y) softmax-ce(x, y) for a single, potentially padded
  minibatch. If the minibatch is padded (that is it contains null examples)
  it is assumed that weights is a binary mask where 0 indicates that the
  example is null.

  Args:
    logits: float array; Output of model in shape [batch, ..., num_classes].
    one_hot_targets: One hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).
    label_smoothing: float scalar to use to smooth the one-hot labels.
    label_weights: float 1d-array; Weight per label of shape [num_classes].
    logits_normalized: bool; if True, the logits are assumed to already be
      normalized.

  Returns:
    The softmax cross entropy of the examples in the given batch.
  """
  if logits.ndim != one_hot_targets.ndim:
    raise ValueError(
        "Incorrect shapes. Got shape %s logits and %s one_hot_targets" %
        (str(logits.shape), str(one_hot_targets.shape)))

  # optionally apply label smoothing
  if label_smoothing is not None:
    one_hot_targets = apply_label_smoothing(one_hot_targets, label_smoothing)

  # optionally apply label weights
  if label_weights is not None:
    one_hot_targets *= label_weights

  if not logits_normalized:
    logits = nn.log_softmax(logits)
  loss = -jnp.einsum("...k,...k->...", one_hot_targets, logits)
  if weights is not None:
    loss = apply_weights(loss, weights)

  return loss


def softmax_cross_entropy_loss(*, logits, labels):
  if len(labels.shape) > 1:
    # labels are one-hot encoded.
    return weighted_unnormalized_softmax_cross_entropy(
        logits=logits, one_hot_targets=labels)
  else:
    return cross_entropy_loss(logits=logits, labels=labels)
