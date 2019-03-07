# coding=utf-8
# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs asynchronous training."""

from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains.run_experiment import run_experiment
from dopamine.utils import threading_utils
import tensorflow as tf


@threading_utils.local_attributes(['_environment'])
class AsyncRunner(run_experiment.Runner):
  """Defines a train runner for asynchronous training."""

  def __init__(
      self, base_dir, create_agent_fn,
      create_environment_fn=atari_lib.create_atari_environment, **kwargs):
    """Creates an asynchronous runner.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
      **kwargs: Additional positional arguments.
    """
    threading_utils.initialize_local_attributes(
        self, _environment=create_environment_fn)
    super(AsyncRunner, self).__init__(
        base_dir=base_dir, create_agent_fn=create_agent_fn,
        create_environment_fn=create_environment_fn, **kwargs)

  def _initialize_session(self):
    """Creates a tf.Session that supports GPU usage in multiple threads."""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    self._sess = tf.Session('', config=config)
