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
"""A config for continual learning."""

import ml_collections


def get_config():
  """Return config files for L2P on CORe50."""
  config = ml_collections.ConfigDict()
  config.model_name = "ViT-B_16"  # support various sized ViT models
  config.per_device_batch_size = 16

  config.dataset = "core50"
  # do not use split version, use domain incremental version
  config.split_core50 = False
  # Gaussian schedule for cifar100
  config.gaussian_schedule = False
  config.gaussian_mode = ""

  config.offline_eval = False
  config.recreate_eval = False
  config.reinit_optimizer = True
  config.eval_last_only = False
  config.save_last_ckpt_only = True

  config.learning_rate = 0.03
  config.optim = "sgd"
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
  config.eval_per_epochs = 5
  config.checkpoint_every_steps = 5000
  config.shuffle_buffer_size = 10000

  config.seed = 42
  config.trial = 0

  # resize cifar as imagenet input
  config.input_size = 224
  config.resize_size = 256
  config.model_config = None
  # load pretrained model
  config.init_checkpoint = ml_collections.FieldReference(None, field_type=str)

  # continual learning
  config.continual = ml_collections.ConfigDict()
  config.continual.num_tasks = 8
  config.continual.num_classes_per_task = 50
  config.continual.rand_seed = -1
  config.continual.num_train_steps_per_task = -1
  config.continual.train_mask = False
  # if doing task incremental
  config.continual.eval_task_inc = False

  # if normalizing pre-logits
  config.norm_pre_logits = False
  config.weight_norm = False
  config.temperature = 1
  # if using 0-1 normalization for input image
  config.norm_01 = True
  config.reverse_task = False

  # configuration for [cls] token
  config.use_cls_token = True
  config.task_specific_cls_token = False

  # classification option for ViT
  config.vit_classifier = "prompt"

  # do not use G-Prompt in L2P
  config.use_g_prompt = False

  # use basic position and prompt-tuning of E-Prompt for L2P
  config.use_e_prompt = True  # Use E-Prompt
  config.e_prompt_layer_idx = [0]
  config.use_prefix_tune_for_e_prompt = False

  # configuration for L2P
  config.prompt_pool = True
  config.prompt_pool_param = ml_collections.ConfigDict()
  config.prompt_pool_param.pool_size = 10
  config.prompt_pool_param.length = 5
  config.prompt_pool_param.top_k = 4
  config.prompt_pool_param.initializer = "uniform"
  config.prompt_pool_param.prompt_key = True
  config.prompt_pool_param.use_prompt_mask = False
  config.prompt_pool_param.mask_first_epoch = False

  config.prompt_pool_param.shared_prompt_pool = False
  config.prompt_pool_param.shared_prompt_key = False
  config.prompt_pool_param.batchwise_prompt = True
  config.prompt_pool_param.prompt_key_init = "uniform"
  config.prompt_pool_param.embedding_key = "cls"
  config.predefined_key_path = ""

  # non freeze model parts
  config.freeze_part = []
  config.freeze_bn_stats = False

  # subsample dataset or not
  config.subsample_rate = -1
  # key loss
  config.pull_constraint = True
  config.pull_constraint_coeff = 1.0

  # prompt utils
  config.prompt_histogram = False
  config.prompt_mask_mode = None
  config.save_prompts = False


  # if doing replay trick, not that if using replay with L2P, make sure
  # to set config.freeze_part = [], ie, not freezing for better adaptation
  # config.continual.replay = ml_collections.ConfigDict()
  # config.continual.replay.num_samples_per_task = 100
  # config.continual.replay.include_new_task = True
  # config.continual.replay_no_mask = True
  # config.continual.replay_reverse_mask = False

  return config
