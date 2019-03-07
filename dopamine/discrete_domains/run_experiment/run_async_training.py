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

from dopamine.discrete_domains.run_experiment import run_experiment
import tensorflow as tf


class AsyncRunner(run_experiment.Runner):
  """Defines a train runner for asynchronous training."""

  def _initialize_session(self):
    """Creates a tf.Session that supports GPU usage in multiple threads."""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    self._sess = tf.Session('', config=config)
