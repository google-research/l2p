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
"""A config for L2P on Gaussian scheduled cifar100."""

import ml_collections

def get_config():
  """Return config files for Gaussian scheduled cifar100."""
  config = ml_collections.ConfigDict()
  config.model_name = "ViT-B_16"
  config.per_device_batch_size = 16

  config.dataset = "cifar100"

  config.gaussian_schedule = True

  config.reinit_optimizer = False

  config.eval_last_only = False
  config.save_last_ckpt_only = True

  config.learning_rate = 0.03
  config.optim = "adam"
  config.sgd_momentum = 0.9
  config.grad_clip_max_norm = 1.0
  config.learning_rate_schedule = "constant"
  config.warmup_epochs = 0
  config.weight_decay = 0
  config.num_epochs = 5
  config.num_eval_steps = -1
  config.eval_pad_last_batch = False
  config.log_loss_every_steps = 3
  config.eval_every_steps = -1

  config.checkpoint_every_steps = 5000
  config.shuffle_buffer_size = 10000

  config.seed = 42
  config.trial = 0

  config.input_size = 224
  config.resize_size = 256

  config.model_config = None
  # load pretrained model
  config.init_checkpoint = ml_collections.FieldReference(None, field_type=str)

  # continual learning
  config.continual = ml_collections.ConfigDict()
  config.continual.num_tasks = 200
  config.continual.num_classes_per_task = 100
  config.continual.rand_seed = -1
  config.continual.num_train_steps_per_task = -1
  config.continual.train_mask = True
  config.continual.eval_task_inc = False

  # if normalizing pre-logits, optional
  config.norm_pre_logits = False
  config.weight_norm = False
  config.temperature = 1

  # important! if using 0-1 normalization
  config.norm_01 = True
  config.reverse_task = False

  # Parameters for using class token or not
  config.use_cls_token = True

  # if using task-specific class token
  config.task_specific_cls_token = False

  # Parameters for using prompt tuning
  config.use_prompt = True

  # Parameters for ViT classifier using
  # choose from "token", "gap", "prompt" and "token+prompt"
  config.vit_classifier = "prompt"
  # Parameters for prompt pool
  config.prompt_pool = True
  config.prompt_pool_param = ml_collections.ConfigDict()
  config.prompt_pool_param.pool_size = 10
  config.prompt_pool_param.length = 5
  config.prompt_pool_param.top_k = 5
  config.prompt_pool_param.initializer = "uniform"
  config.prompt_pool_param.prompt_key = True
  config.prompt_pool_param.use_prompt_mask = False
  config.prompt_pool_param.batchwise_prompt = True

  config.prompt_pool_param.prompt_key_init = "uniform"

  config.prompt_pool_param.embedding_key = "cls"

  config.predefined_key_path = ""

  # freeze
  config.freeze_part = ["encoder", "embedding", "cls"]
  config.freeze_bn_stats = False

  config.subsample_rate = -1

  config.prompt_histogram = True

  config.pull_constraint = True
  config.pull_constraint_coeff = 0.1

  return config

