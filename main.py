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
"""Main file for running the example."""

import os

from absl import app
from absl import flags
from absl import logging
import jax
from ml_collections import config_flags
import train_continual
import tensorflow as tf


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "my_config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.DEFINE_string("exp_id", None, "Work id.")
# flags.mark_flags_as_required(["config", "workdir"])
# Flags --jax_backend_target and --jax_xla_backend are available through JAX.


def main(argv):
  del argv

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")
  if FLAGS.exp_id:
    FLAGS.workdir = os.path.join(FLAGS.workdir, FLAGS.exp_id)

  if FLAGS.jax_backend_target:
    logging.info("Using JAX backend target %s", FLAGS.jax_backend_target)
    jax_xla_backend = ("None" if FLAGS.jax_xla_backend is None else
                       FLAGS.jax_xla_backend)
    logging.info("Using JAX XLA backend %s", jax_xla_backend)

  logging.info("JAX host: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX devices: %r", jax.devices())


  train_continual.train_and_evaluate(FLAGS.my_config, FLAGS.workdir)


if __name__ == "__main__":
  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  app.run(main)
