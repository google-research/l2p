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
"""Main framework for continual learning."""

import functools
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from absl import logging
from clu import checkpoint
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from libml import input_pipeline
from libml import losses
from libml import utils
from libml import utils_vit
from libml.continual_buffer import ReplayBuffer
from libml.eval_metrics import EvalMetrics_list
from models import resnet_v1
from models import vit
import tensorflow as tf

# global variable for maintaining summary steps
summary_step = 0


@flax.struct.dataclass
class TrainState:
  step: int
  optimizer: flax.optim.Optimizer
  model_state: Any


def create_optimizer(config: ml_collections.ConfigDict, params: Any):
  """Optionally creates the optimizer to use for every task.

  Args:
    config: Configuration to use.
    params: Parameters associated with the optimizer.

  Returns:
    The newly created optimizer.
  """
  if config.optim in ("adamw", "adam"):
    if config.get("optim_wd_ignore"):
      # Allow zero weight decay for certain parameters listed in optim_wd_ignore
      igns = config.optim_wd_ignore
      p = flax.optim.ModelParamTraversal(
          lambda path, _: not any([i in path for i in igns]))
      p_nowd = flax.optim.ModelParamTraversal(
          lambda path, _: any([i in path for i in igns]))
      p_opt = flax.optim.Adam(weight_decay=config.weight_decay)
      p_nowd_opt = flax.optim.Adam(weight_decay=0)
      opt_def = flax.optim.MultiOptimizer((p, p_opt), (p_nowd, p_nowd_opt))
    else:
      opt_def = flax.optim.Adam(weight_decay=config.weight_decay)
  elif config.optim == "sgd":
    opt_def = flax.optim.Momentum(beta=config.sgd_momentum, nesterov=True)
  else:
    raise NotImplementedError(f"{config.optim} does not exist.")

  if (not config.get("freeze_part")) or config.get("optim_wd_ignore"):
    optimizer = opt_def.create(params)
  else:
    # freeze part of the parameters according to specification
    freeze_part = config.freeze_part
    p_normal = flax.optim.ModelParamTraversal(
        lambda path, _: not any([i in path for i in freeze_part]))
    p_freeze = flax.optim.ModelParamTraversal(
        lambda path, _: any([i in path for i in freeze_part]))
    if config.optim == "adam":
      p_normal_opt = flax.optim.Adam(weight_decay=config.weight_decay)
      p_freeze_opt = flax.optim.Adam(weight_decay=0)
    elif config.optim == "sgd":
      p_normal_opt = flax.optim.Momentum(
          beta=config.sgd_momentum, nesterov=True)
      p_freeze_opt = flax.optim.Momentum(beta=0, nesterov=False)
    opt_def = flax.optim.MultiOptimizer((p_normal, p_normal_opt),
                                        (p_freeze, p_freeze_opt))
    optimizer = opt_def.create(params)
  return optimizer


def create_train_state(config: ml_collections.ConfigDict, rng: np.ndarray,
                       input_shape: Sequence[int],
                       num_classes: int) -> Tuple[Any, TrainState]:
  """Creates and initializes the model.

  Args:
    config: Configuration for model.
    rng: JAX PRNG Key.
    input_shape: Shape of the inputs fed into the model.
    num_classes: Number of classes in the output layer.

  Returns:
    The initialized TrainState with the optimizer.
  """
  # Create model function.
  if config.model_name.startswith("resnet"):
    model_cls = resnet_v1.create_model(config.model_name, config)
  elif config.model_name.startswith("ViT"):
    model_cls, model_config = vit.create_model(config.model_name, config)
    config.model_config = model_config
  else:
    raise ValueError(f"Model {config.model_name} not supported.")
  model = functools.partial(model_cls, num_classes=num_classes)
  variables = model(train=False).init(rng, jnp.ones(input_shape))
  model_state = dict(variables)
  params = model_state.pop("params")
  parameter_overview.log_parameter_overview(params)
  if config.get("log_model_profile"):  # Be True or [1, 2]
    message_1 = utils.log_throughput(model, variables, input_shape)
    message_2 = utils.compute_flops(model, variables,
                                    [1] + list(input_shape[1:]))
    count = parameter_overview.count_parameters(params)
    message_3 = "Params: {:,}".format(count)
    message = ", ".join([message_1, message_2, message_3])
    logging.info("Profile results %s", message)
    if (isinstance(config.log_model_profile, (int,)) and
        config.log_model_profile >= 2):
      sys.exit(0)
  optimizer = create_optimizer(config, params)
  return model, TrainState(step=0, optimizer=optimizer, model_state=model_state)


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):

  accuracy: metrics.Accuracy
  eval_loss: metrics.Average.from_output("loss")


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):

  train_accuracy: metrics.Accuracy
  learning_rate: metrics.LastValue.from_output("learning_rate")
  loss: metrics.Average.from_output("loss")
  loss_std: metrics.Std.from_output("loss")
  l2_grads: metrics.Average.from_output("l2_grads")


def train_step(
    model: Any,
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    rng: np.ndarray,
    learning_rate_fn: Callable[[int], float],
    weight_decay: float,
    grad_clip_max_norm: Optional[float] = None,
    initial_step: int = -1,
    freeze: bool = False,
    freeze_bn_stats: bool = False,
    num_total_class: int = -1,
    train_mask: bool = False,
    class_mask=None,
    cur_task_id: int = -1,
    use_prompt_mask=False,
    original_vit_model=None,  # 9.6: added for cls feature
    original_vit_params=None,  # 9.6: added for cls feature
    config=None
) -> Tuple[TrainState, metrics.Collection]:
  """Performs a single training step.

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    state: State of the model (optimizer and state).
    batch: Training inputs for this step.
    rng: Random seed.
    learning_rate_fn: Function that computes the learning rate given the step
      number.
    weight_decay: Weighs L2 regularization term.
    grad_clip_max_norm: Gradient norm max value. Default to be None.
    initial_step: Initial step number of current task. Used for calculating the
      relateive step in the current task.
    freeze: If freeze part of the model parameters according to
      config.freeze_parts.
    freeze_bn_stats: If freeze parameters of BatchNorm layers (if have).
    num_total_class: Total number of classes for all tasks.
    train_mask: If using the class mask at training.
    class_mask: 0-1 vectors, for blocking out gradients of classes from
      non-current tasks.
    cur_task_id: ID of the current tasks, starting from 0.
    use_prompt_mask: If mask the prompts at training time, equivalent to
      diversifying penalty.
    original_vit_model: Original vit model definition. Use for calculating cls
      token feature used as key in prompt selection.
    original_vit_params: Pretrained vit model weights. Use for calculating cls
      token feature used as key in prompt selection.
    config: Configuration for model.

  Returns:
    The new model state and dictionary with metrics.
  """
  logging.info("train_step(batch=%s)", batch)

  step = state.step + 1
  if initial_step > 0:
    lr = learning_rate_fn(step - initial_step + 1)
  else:
    lr = learning_rate_fn(step)
  # Convert one-hot labels to single values if appliable.
  u_labels = (
      jnp.argmax(batch["label"], 1)
      if len(batch["label"].shape) > 1 else batch["label"])

  def loss_fn(params):
    variables = {"params": params}
    # save here for later replacement
    old_model_state = state.model_state
    variables.update(state.model_state)
    if use_prompt_mask:
      start = cur_task_id * config.prompt_pool_param.top_k
      end = (cur_task_id + 1) * config.prompt_pool_param.top_k
      single_prompt_mask = jnp.arange(start, end)
      single_prompt_mask = single_prompt_mask[jnp.newaxis, :]
      prompt_mask = jnp.repeat(
          single_prompt_mask, batch["image"].shape[0], axis=0)
      if end > config.prompt_pool_param.pool_size:
        prompt_mask = None
    else:
      prompt_mask = None

    # calculating cls feature
    if original_vit_model is not None:
      original_vit_variables = {
          "params": original_vit_params,
      }
      original_vit_res = original_vit_model(train=False).apply(
          original_vit_variables, batch["image"], mutable=False)
      cls_features = original_vit_res["pre_logits"]
    else:
      cls_features = None

    task_id = cur_task_id
    res, new_model_state = model(train=True).apply(
        variables,
        batch["image"],
        prompt_mask,
        task_id,
        cls_features,
        batch["label"],
        mutable=["batch_stats"],
        rngs={"dropout": rng})
    logits = res["logits"]
    # here is the trick to mask out classes of non-current tasks
    if train_mask:
      not_mask = np.setdiff1d(np.arange(num_total_class), class_mask)
      if config.continual.get("replay_no_mask"):
        # dev, bs, #class, where dev is implicit here
        logits = jax.ops.index_update(
            logits, jax.ops.index[:config.per_device_batch_size, not_mask],
            -jnp.inf)
        if config.continual.get("replay_reverse_mask"):
          logits = logits.at[config.per_device_batch_size:,
                             class_mask].set(-jnp.inf)
      else:
        logits = logits.at[..., not_mask].set(-jnp.inf)
    # end of trick
    loss = jnp.mean(
        losses.softmax_cross_entropy_loss(logits=logits, labels=batch["label"]))
    if weight_decay > 0:
      weight_penalty_params = jax.tree_leaves(variables["params"])
      weight_l2 = sum(
          [jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
      weight_penalty = weight_decay * 0.5 * weight_l2
      loss = loss + weight_penalty

    if config.get("pull_constraint"):
      loss = loss - config.pull_constraint_coeff * res["reduce_sim"]

    new_model_state = dict(new_model_state)
    if freeze_bn_stats:
      new_model_state = old_model_state
    return loss, (new_model_state, logits)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (new_model_state, logits)), grad = grad_fn(state.optimizer.target)

  # Compute average gradient across multiple workers.
  grad = jax.lax.pmean(grad, axis_name="batch")

  # Compute l2 grad always for training debugging.
  grads, _ = jax.tree_flatten(grad)
  l2_g = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads]))
  if grad_clip_max_norm:
    g_factor = jnp.minimum(1.0, grad_clip_max_norm / (l2_g + 1e-6))
    grad = jax.tree_map(lambda p: g_factor * p, grad)
  if freeze:
    hparams = state.optimizer.optimizer_def.hyper_params
    new_optimizer = state.optimizer.apply_gradient(
        grad,
        hyper_params=[
            hparams[0].replace(learning_rate=lr),
            hparams[1].replace(learning_rate=0),
        ])
  else:
    new_optimizer = state.optimizer.apply_gradient(grad, learning_rate=lr)
  new_state = state.replace(  # pytype: disable=attribute-error
      step=step,
      optimizer=new_optimizer,
      model_state=new_model_state)

  metrics_update = TrainMetrics.gather_from_model_output(
      loss=loss,
      logits=logits,
      labels=u_labels,
      learning_rate=lr,
      l2_grads=l2_g)
  return new_state, metrics_update


def eval_step(model: Any,
              state: TrainState,
              batch: Dict[str, jnp.ndarray],
              task_id: int = -1,
              task_inc: bool = False,
              class_mask=None,
              return_prompt_id=False,
              original_vit_model=None,
              original_vit_params=None) -> metrics.Collection:
  """Computes the metrics for the given model in inference mode.

  The model is applied to the inputs with train=False using all devices on the
  host. Afterwards metrics are averaged across *all* devices (of all hosts).

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    state: Replicate model state.
    batch: Inputs that should be evaluated.
    task_id: Current task to be evaluated.
    task_inc: Specify if doing task incremental, if so, adding corresponding
      mask.
    class_mask: 0-1 vectors, for blocking out gradients of classes from
      non-current tasks.
    return_prompt_id: If return selected prompt ids for each example, used for
      plotting the prompt selection histrogram.
    original_vit_model: Original vit model definition. Use for calculating cls
      token feature used as key in prompt selection.
    original_vit_params: Pretrained vit model weights. Use for calculating cls
      token feature used as key in prompt selection.

  Returns:
    Dictionary of the replicated metrics, and optionally the selected prompt
    ids, if return_prompt_id is specified as True
  """

  if original_vit_model is not None:
    original_vit_variables = {
        "params": original_vit_params,
    }
    original_vit_res = original_vit_model(train=False).apply(
        original_vit_variables, batch["image"], mutable=False)
    cls_features = original_vit_res["pre_logits"]
  else:
    cls_features = None

  logging.info("eval_step(batch=%s)", batch)
  variables = {
      "params": state.optimizer.target,
  }
  variables.update(state.model_state)
  res = model(train=False).apply(
      variables, batch["image"], cls_features=cls_features, mutable=False)
  logits = res["logits"]
  if task_inc:
    # adding mask to output logits
    logits_mask = jnp.ones_like(logits) * (-jnp.inf)
    logits_mask = logits_mask.at[..., class_mask].set(0)
    logits = logits + logits_mask
  loss = jnp.mean(
      losses.cross_entropy_loss(logits=logits, labels=batch["label"]))
  if task_id < 0:
    return EvalMetrics.gather_from_model_output(
        logits=logits,
        labels=batch["label"],
        loss=loss,
        mask=batch.get("mask"),
    )
  else:
    metrics_update = EvalMetrics_list[task_id].gather_from_model_output(
        logits=logits,
        labels=batch["label"],
        loss=loss,
        mask=batch.get("mask"),
    )
    if return_prompt_id:
      return metrics_update, res["prompt_idx"]
    else:
      return metrics_update


def evaluate_tasks_till_now(cur_task_id: int,
                            model: nn.Module,
                            state: TrainState,
                            eval_ds_list: List[tf.data.Dataset],
                            class_mask_list: Any,
                            num_eval_steps: int = -1,
                            task_inc: bool = False,
                            return_prompt_id: bool = False,
                            original_vit_model=None,
                            original_vit_params=None) -> Union[None, List[Any]]:
  """Evaluates for all tasks till the current one.

  Args:
    cur_task_id: Current task id.
    model: Flax module for the model.
    state: Replicate model state.
    eval_ds_list: List of the evaluation datasets.
    class_mask_list: List of class masks used for each task.
    num_eval_steps: Number of evaluation steps. Default to be -1, meaning use
      all batches in the dataset to do evaluation.
    task_inc: Specify if doing task incremental, if so, adding corresponding
      mask.
    return_prompt_id: If return selected prompt ids for each example, used for
      plotting the prompt selection histrogram.
    original_vit_model: Original vit model definition. Use for calculating cls
      token feature used as key in prompt selection.
    original_vit_params: Pretrained vit model weights. Use for calculating cls
      token feature used as key in prompt selection.

  Returns:
    List of the replicated metrics, and the list of selected prompt
    ids. Note that if return_prompt_id is specified as False, the list of
    selected prompt ids will be an empty list.
  """
  # logging.info(f"Starting evaluation for task 0 to {cur_task_id}.")
  eval_metrics_list = []
  prompt_idx_list = []

  for task_id in range(cur_task_id + 1):
    prompt_idx_cur_task = []
    eval_metrics = None
    eval_ds = eval_ds_list[task_id]
    b_pmap = functools.partial(
        jax.pmap, axis_name="batch", static_broadcasted_argnums=0)
    eval_func = b_pmap(
        functools.partial(
            eval_step,
            task_id=task_id,
            task_inc=task_inc,
            class_mask=class_mask_list[task_id],
            return_prompt_id=return_prompt_id,
            original_vit_model=original_vit_model,
            original_vit_params=original_vit_params))
    for step, batch in enumerate(eval_ds):  # pytype: disable=wrong-arg-types
      batch = jax.tree_map(np.asarray, batch)
      res = eval_func(model, state, batch)
      if return_prompt_id:
        metrics_update = flax_utils.unreplicate(res[0])
        prompt_idx = res[1]
        prompt_idx_cur_task.append(prompt_idx)
      else:
        metrics_update = flax_utils.unreplicate(res)
      eval_metrics = (
          metrics_update
          if eval_metrics is None else eval_metrics.merge(metrics_update))
      del batch
      if num_eval_steps > 0 and step + 1 == num_eval_steps:
        break

    eval_metrics_list.append(eval_metrics)
    if return_prompt_id:
      prompt_idx_list.append(jnp.concatenate(prompt_idx_cur_task))

    if len(eval_ds_list) == 1:
      break
  return eval_metrics_list, prompt_idx_list


def train_and_evaluate_per_task(task_id: int, config: ml_collections.ConfigDict,
                                workdir: str, *, model, state,
                                original_vit_model, original_vit_params,
                                num_total_class, train_ds_list, eval_ds_list,
                                class_stats_list, class_mask_list, acc_matrix,
                                writer, replay_buffer, rng):
  """Runs a training and evaluation loop for a single task.

  Args:
    task_id: The id of the current task we are training on.
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
    model: Input model.
    state: State of the input model, unreplicated.
    original_vit_model: Original vit model definition. Use for calculating cls
      token feature used as key in prompt selection.
    original_vit_params: Pretrained vit model weights. Use for calculating cls
      token feature used as key in prompt selection.
    num_total_class: Total number of classes for all tasks.
    train_ds_list: The list of training datasets.
    eval_ds_list: The list of evaluation datasets.
    class_stats_list: The list of class statistics (number of training and test
      examples) for each task.
    class_mask_list: The list of class masks.
    acc_matrix: Global matrix to save end-of-task accuracies for calculate
      forgetting and learning accuracy.
    writer: Default metrics writer.
    replay_buffer: The replay buffer to use. Default to be None.
    rng: Random seed.

  Returns:
    The unreplicated state and the random seed.
  """
  # logging.info(f"Working on task {task_id}.")
  review_trick = config.continual.get("review_trick")
  global summary_step

  # Create new optimizer for each task to clear optimizer status
  if task_id > 0 and config.reinit_optimizer:
    optimizer = create_optimizer(config, state.optimizer.target)
    state = state.replace(optimizer=optimizer)
    if config.continual.get("weights_transfer"):
      param_dict = state.optimizer.target
      transferred_weights = utils.transfer_weights(
          config,
          param_dict,
          task_id,
          kernel_only=config.continual.get("kernel_only"))
      optimizer = optimizer.replace(target=transferred_weights)
      state = state.replace(optimizer=optimizer)

  # Transfer previous learned prompt params to the new prompt
  if config.prompt_pool and config.prompt_pool_param.shared_prompt_pool:
    if task_id > 0:
      prev_start = (task_id - 1) * config.prompt_pool_param.top_k
      prev_end = task_id * config.prompt_pool_param.top_k
      cur_start = prev_end
      cur_end = (task_id + 1) * config.prompt_pool_param.top_k
      if (prev_end > config.prompt_pool_param.pool_size) or (
          cur_end > config.prompt_pool_param.pool_size):
        pass
      else:
        param_dict = state.optimizer.target
        prompt_pool_para = param_dict["prompt_pool"]["prompt"]
        if config.use_prefix_tune_for_e_prompt:
          prompt_pool_para = prompt_pool_para.at[:, :, cur_start:cur_end].set(
              prompt_pool_para[:, :, prev_start:prev_end])
        else:
          prompt_pool_para = prompt_pool_para.at[:, cur_start:cur_end].set(
              prompt_pool_para[:, prev_start:prev_end])
        param_dict, _ = utils.replace_prompt_pool(param_dict, prompt_pool_para)
        state = utils.state_with_new_param(state, param_dict)

  # Transfer previous learned prompt param keys to the new prompt
  if config.prompt_pool and config.prompt_pool_param.prompt_key and config.prompt_pool_param.shared_prompt_key:
    if task_id > 0:
      prev_start = (task_id - 1) * config.prompt_pool_param.top_k
      prev_end = task_id * config.prompt_pool_param.top_k
      cur_start = prev_end
      cur_end = (task_id + 1) * config.prompt_pool_param.top_k
      param_dict = state.optimizer.target
      prompt_key_para = param_dict["prompt_pool"]["key"]
      prompt_key_para = prompt_key_para.at[cur_start:cur_end].set(
          prompt_key_para[prev_start:prev_end])
      param_dict, _ = utils.replace_prompt_key(param_dict, prompt_key_para)
      state = utils.state_with_new_param(state, param_dict)

  # Build input pipeline.
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.process_index())

  train_ds = train_ds_list[task_id]
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types

  # Learning rate schedule.
  global_batch_size = config.per_device_batch_size * jax.device_count()
  # num_train_steps = config.continual.num_train_steps_per_task
  # Hacky operation, currently disable setting the number of training steps.
  num_train_steps = -1
  # specify number of total train steps

  if num_train_steps == -1:
    # num_train_steps = train_ds.cardinality().numpy()
    # 0 represents # training examples here
    num_train_steps = int(
        class_stats_list[task_id][0] * config.num_epochs) // global_batch_size
    assert num_train_steps > 0
  steps_per_epoch = num_train_steps // config.num_epochs
  num_train_steps = steps_per_epoch * config.num_epochs
  if config.eval_every_steps == -1 or config.get("eval_per_epochs"):
    # Show plots in the epoch view (x-axis).
    eval_every_steps = steps_per_epoch * config.get("eval_per_epochs", 1)
    summary_step_div = steps_per_epoch
  else:
    eval_every_steps = config.eval_every_steps
    summary_step_div = 1

  if review_trick:
    num_review_steps = config.continual.num_review_steps
    num_review_epochs = config.continual.num_review_epochs
    if num_review_steps == -1:
      num_review_steps = replay_buffer.num_samples_per_task * num_review_epochs // global_batch_size
      assert num_review_steps > 0
    steps_per_review_epoch = num_review_steps // num_review_epochs
    num_review_steps = steps_per_review_epoch * num_review_epochs
    # no matter how many epochs we train, we only eval five times,
    # so the epoch should be > 5
    eval_every_review_steps = steps_per_review_epoch * (num_review_epochs // 5)
    summary_review_step_div = eval_every_review_steps
  else:
    num_review_steps = 0

  global_num_steps = (num_review_steps +
                      num_train_steps) * config.continual.num_tasks

  logging.info(
      "global_batch_size=%d, num_train_steps=%d, steps_per_epoch=%d, eval_every_steps=%d",
      global_batch_size, num_train_steps, steps_per_epoch, eval_every_steps)
  # We treat the learning rate in the config as the learning rate for batch size
  # 256 but scale it according to our batch size.
  base_learning_rate = config.learning_rate * global_batch_size / 256.0
  learning_rate_fn = functools.partial(
      utils.get_learning_rate,
      base_learning_rate=base_learning_rate,
      steps_per_epoch=steps_per_epoch,
      num_epochs=config.num_epochs,
      warmup_epochs=config.warmup_epochs,
      schedule=config.learning_rate_schedule,
      min_learning_rate=config.get("min_learning_rate", 0.) *
      global_batch_size / 256.0)

  # Set up checkpointing of the model and the input pipeline.
  checkpoint_dir = os.path.join(workdir, f"checkpoints_task{task_id}")

  if (not config.save_last_ckpt_only) or (
      config.save_last_ckpt_only and
      (task_id == (config.continual.num_tasks - 1))):
    ckpt = checkpoint.MultihostCheckpoint(checkpoint_dir, max_to_keep=2)
    state = ckpt.restore_or_initialize(state)
  initial_step = int(state.step) + 1

  disable_l2_wd = config.optim == "adamw"
  # Distribute training.
  state = flax_utils.replicate(state)

  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          model=model,
          learning_rate_fn=learning_rate_fn,
          weight_decay=0 if disable_l2_wd else config.weight_decay,
          grad_clip_max_norm=config.get("grad_clip_max_norm"),
          initial_step=initial_step,
          freeze=bool(config.freeze_part),
          freeze_bn_stats=config.freeze_bn_stats,
          num_total_class=num_total_class,
          train_mask=config.continual.train_mask,
          class_mask=class_mask_list[task_id],
          cur_task_id=task_id,
          use_prompt_mask=config.prompt_pool_param.use_prompt_mask,
          original_vit_model=original_vit_model,
          original_vit_params=original_vit_params,
          config=config),
      axis_name="batch")

  p_train_step_flag = False

  if initial_step == 1:
    writer.write_hparams(dict(config))

  logging.info("Starting training loop at step %d.", initial_step)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=global_num_steps, writer=writer)
  train_metrics = None
  rng, drop_out_rng = jax.random.split(rng, 2)
  drop_out_rng = jax.random.fold_in(drop_out_rng, jax.process_index())

  if replay_buffer:
    replay_buffer.gen_batch_index(
        num_total_samples=class_stats_list[task_id][0],
        per_device_bs=config.per_device_batch_size)
    num_savable_steps = class_stats_list[task_id][0] // global_batch_size
  with metric_writers.ensure_flushes(writer):

    for step in range(initial_step,
                      initial_step + num_train_steps + num_review_steps):
      # `step` is a Python integer. `state.step` is JAX integer on the GPU/TPU
      # devices.
      # this relative step is the step inside each task, used for replay
      relative_step = step - initial_step
      is_last_step = ((relative_step + 1) == num_train_steps + num_review_steps)

      in_train_session = (step < initial_step + num_train_steps)
      in_review_session = (step >= initial_step + num_train_steps)

      if in_review_session and (not p_train_step_flag):
        # do not use mask in the review_session!
        p_train_step = jax.pmap(
            functools.partial(
                train_step,
                model=model,
                learning_rate_fn=learning_rate_fn,
                weight_decay=0 if disable_l2_wd else config.weight_decay,
                grad_clip_max_norm=config.get("grad_clip_max_norm"),
                initial_step=initial_step,
                freeze=bool(config.freeze_part),
                freeze_bn_stats=config.freeze_bn_stats,
                num_total_class=num_total_class,
                train_mask=False,
                class_mask=None,
                config=config),
            axis_name="batch")
        p_train_step_flag = True

      if config.get("no_train") and in_train_session:
        # we just update the replay buffer, but do not train
        if replay_buffer and (relative_step < num_savable_steps):
          batch = jax.tree_map(np.asarray, next(train_iter))
          # add this line to distinguish from logits reply
          # if not config.continual.replay.logits_replay:
          replay_buffer.add_example(task_id, relative_step, batch)
      else:
        if config.get("review_last_only") and in_review_session and task_id < (
            config.continual.num_tasks - 1):
          pass
        else:
          if in_train_session:
            batch = jax.tree_map(np.asarray, next(train_iter))
            # replay starts
            # if replay, we should save it into the buffer
            if replay_buffer and (relative_step < num_savable_steps):
              replay_buffer.add_example(task_id, relative_step, batch)
            # if in 2nd or later task, we also sample from the buffer
            if replay_buffer and (task_id > 0) and (not review_trick):
              replay_batch = replay_buffer.get_random_batch(
                  config.per_device_batch_size,
                  config.continual.replay.include_new_task)
              # concatenate them through the batch_size axis
              image_concat = np.concatenate(
                  [batch["image"], replay_batch["image"]], axis=1)
              label_concat = np.concatenate(
                  [batch["label"], replay_batch["label"]], axis=1)
              label_concat = label_concat.astype(np.int32)
              batch = {"image": image_concat, "label": label_concat}
          else:
            batch = replay_buffer.get_random_batch(config.per_device_batch_size,
                                                   True)

          drop_out_rng_step = jax.random.fold_in(drop_out_rng, step)
          drop_out_rng_step_all = jax.random.split(drop_out_rng_step,
                                                   jax.local_device_count())
          if config.get("weight_norm"):
            state = flax_utils.unreplicate(state)
            param_dict = state.optimizer.target
            param_dict = utils.weight_norm(param_dict)
            optimizer = state.optimizer.replace(target=param_dict)
            state = state.replace(optimizer=optimizer)
            state = flax_utils.replicate(state)

          state, metrics_update = p_train_step(
              state=state, batch=batch, rng=drop_out_rng_step_all)
          metric_update = flax_utils.unreplicate(metrics_update)
          train_metrics = (
              metric_update
              if train_metrics is None else train_metrics.merge(metric_update))
          # Quick indication that training is happening.
          logging.log_first_n(logging.INFO, "Finished training step %d.", 5,
                              step)
          # align logging parameters in review session:

          if (in_train_session and
              (step % eval_every_steps == 0)) or (in_review_session and (
                  (step - initial_step - num_train_steps + 1) %
                  eval_every_review_steps == 0)):
            summary_step += 1
            write_flag = True
          else:
            write_flag = False

          if (relative_step % summary_step_div == 0) or is_last_step:
            train_summary_step = relative_step // summary_step_div + task_id * config.num_epochs
            writer.write_scalars(train_summary_step, train_metrics.compute())
            writer.write_scalars(train_summary_step, {"task_id": task_id})
            train_metrics = None

          # add this setting for gaussian schedule
          if config.get("gaussian_schedule"):
            eval_interval = 10
            write_flag = False
          else:
            eval_interval = 1

          if config.eval_last_only:
            eval_interval = config.continual.num_tasks

          if (write_flag or is_last_step) and ((task_id + 1) % eval_interval
                                               == 0):
            with report_progress.timed("eval"):
              eval_metrics_list, prompt_idx_list = evaluate_tasks_till_now(
                  task_id,
                  model,
                  state,
                  eval_ds_list,
                  class_mask_list,
                  config.num_eval_steps,
                  config.continual.eval_task_inc,
                  config.get("prompt_histogram"),
                  original_vit_model=original_vit_model,
                  original_vit_params=original_vit_params)

            for i, prompt_idx in enumerate(prompt_idx_list):
              writer.write_histograms(
                  summary_step, {f"histogram_{i}": prompt_idx},
                  {f"histogram_{i}": config.prompt_pool_param.pool_size})

            avg_acc, count = 0, 0
            for i, eval_metrics in enumerate(eval_metrics_list):
              res_to_write = eval_metrics.compute()
              writer.write_scalars(summary_step, res_to_write)
              count += 1
              avg_acc += res_to_write[f"accuracy_{i}"]
            writer.write_scalars(summary_step, {"avg_acc": avg_acc / count})

          if is_last_step and ((task_id + 1) % eval_interval == 0):
            for i, eval_metrics in enumerate(eval_metrics_list):
              res_to_write = eval_metrics.compute()
              # row -> task_id; col -> current task
              acc_matrix[i, task_id] = res_to_write[f"accuracy_{i}"]
            diagonal = np.diag(acc_matrix)
            if task_id > 0:
              forgetting = np.mean((np.max(acc_matrix, axis=1) -
                                    acc_matrix[:, task_id])[:task_id])
              backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])
              writer.write_scalars(summary_step, {
                  "forgetting": forgetting,
                  "backward": backward
              })
            learning_acc = np.mean(diagonal[:(task_id + 1)])
            writer.write_scalars(summary_step, {"learning_acc": learning_acc})

          if (not config.save_last_ckpt_only) or (
              config.save_last_ckpt_only and
              (task_id == (config.continual.num_tasks - 1))):
            if (step % config.checkpoint_every_steps) == 0 or is_last_step and (
                (task_id + 1) % eval_interval == 0):
              with report_progress.timed("checkpoint"):
                ckpt.save(flax_utils.unreplicate(state))

  state = flax_utils.unreplicate(state)
  logging.info("Finishing training at step %d", int(state.step))

  return state, rng


def get_train_eval_components(config: ml_collections.ConfigDict,
                              rng: jax.random.PRNGKey):
  """Helper function for generating train and evaluation datasets."""
  if config.dataset == "5datasets":
    rng, train_ds_list, eval_ds_list, class_stats_list, class_mask_list = input_pipeline.create_5datasets(
        config, rng)
    train_ds = train_ds_list[0]
  elif config.dataset == "core50":
    rng, train_ds_list, eval_ds_list, class_stats_list, class_mask_list = input_pipeline.create_core50(
        config, rng)
    train_ds = train_ds_list[0]
  elif config.dataset == "imagenet_r":
    if config.get("imr_eval"):
      rng, train_ds_list, eval_ds_list, class_stats_list, class_mask_list = input_pipeline.create_split_imagenet_r_eval(
          config, rng)
    else:
      rng, train_ds_list, eval_ds_list, class_stats_list, class_mask_list = input_pipeline.create_split_imagenet_r(
          config, rng)
    train_ds = train_ds_list[0]
  elif config.dataset == "cifar100" and config.get("gaussian_schedule"):
    create_gaussian_cifar100 = input_pipeline.create_gaussian_cifar100
    rng, train_ds_list, eval_ds_list, class_stats_list, class_mask_list = create_gaussian_cifar100(
        config, rng)
    train_ds = train_ds_list[0]
  else:
    rng, data_rng = jax.random.split(rng)
    data_rng = jax.random.fold_in(data_rng, jax.process_index())
    _, train_ds, _ = input_pipeline.create_datasets(config, data_rng)

    # Build input pipeline, for creating tasks of datasets
    task_range = config.continual.num_tasks
    train_ds_list, eval_ds_list, class_stats_list, class_mask_list = [], [], [], []
    for task_id in range(task_range):
      rng, data_rng = jax.random.split(rng)
      data_rng = jax.random.fold_in(data_rng, jax.process_index())
      _, train_ds_sub, eval_ds_sub, class_stats, class_mask = input_pipeline.create_continual_datasets(
          config, data_rng, task_id)
      train_ds_list.append(train_ds_sub)
      eval_ds_list.append(eval_ds_sub)
      class_stats_list.append(class_stats)
      class_mask_list.append(class_mask)
  if ("5datasets" in config.dataset) or ("core50" in config.dataset):
    num_total_class = 50
  elif "imagenet_r" in config.dataset:
    num_total_class = 200
  else:
    num_total_class = 100  # ds_info.features["label"].num_classes
  return rng, train_ds_list, eval_ds_list, class_stats_list, class_mask_list, train_ds, num_total_class


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
  """Runs a training and evaluation loop for sequentially arriving tasks.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  # create parent workdir
  if not tf.io.gfile.exists(workdir):
    tf.io.gfile.makedirs(workdir)

  # generate random seed
  rng = jax.random.PRNGKey(config.seed)

  rng, train_ds_list, eval_ds_list, class_stats_list, class_mask_list, train_ds, num_total_class = get_train_eval_components(
      config, rng)

  rng, model_rng = jax.random.split(rng)
  model, state = create_train_state(
      config,
      model_rng,
      input_shape=train_ds.element_spec["image"].shape[1:],
      num_classes=num_total_class)

  if config.get("init_checkpoint"):
    state = utils.load_and_custom_init_checkpoint(
        config=config, init_state=state)

  # create default writer
  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0)

  # create matrix to save end-of-task accuracies for calculate F and LA
  acc_matrix = np.zeros(
      (config.continual.num_tasks, config.continual.num_tasks))

  # if doing replay strategy or not
  if config.continual.get("replay"):
    # initialize replay buffer
    replay_buffer = ReplayBuffer(
        continual_config=config.continual,
        input_shape=train_ds.element_spec["image"].shape[2:])
  else:
    replay_buffer = None

  # Load original ViT for feature extraction
  original_vit_model = None
  original_vit_params = None
  if config.get("prompt_pool_param"):
    if config.prompt_pool_param.embedding_key == "cls":
      original_model_cls, original_model_config = vit.create_original_vit(
          config.model_name)
      original_vit_model = functools.partial(
          original_model_cls, num_classes=num_total_class)
      rng, model_rng = jax.random.split(rng)
      original_vit_init_param = original_vit_model(train=False).init(
          model_rng, jnp.ones(train_ds.element_spec["image"].shape[1:]))
      original_vit_params = utils_vit.load_pretrained(
          pretrained_path=config.init_checkpoint,
          init_params=original_vit_init_param["params"],
          model_config=original_model_config)

  task_range = config.continual.num_tasks
  for task_id in range(task_range):
    kwargs = {
        "model": model,
        "state": state,
        "original_vit_model": original_vit_model,
        "original_vit_params": original_vit_params,
        "num_total_class": num_total_class,
        "train_ds_list": train_ds_list,
        "eval_ds_list": eval_ds_list,
        "class_stats_list": class_stats_list,
        "class_mask_list": class_mask_list,
        "acc_matrix": acc_matrix,
        "writer": writer,
        "replay_buffer": replay_buffer,
        "rng": rng
    }
    state, rng = train_and_evaluate_per_task(task_id, config, workdir, **kwargs)
