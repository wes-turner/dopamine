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

import threading
import time

from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains.run_experiment import run_experiment
from dopamine.utils import threading_utils
import tensorflow as tf

_SLEEP_SECONDS = 0.01


def async_method(method):
  def _method(*args):
    thread = threading.Thread(target=method, args=args)
    thread.start()
    thread.join()
  return _method


class _ThreadSafeQueue(object):
  """Implements a thread safe queue than can be filled and free-up."""

  def __init__(self, size):
    self._size = size
    self._lock = threading.Lock()
    self.reset()

  def reset(self):
    """Empty the queue."""
    with self._lock:
      self._value = 0
      self._total = 0

  def fill(self, maximum=None):
    """Fill the queue."""
    maximum = maximum or self._size
    with self._lock:
      delta = min(self._size - self._value, maximum)
      self._value += delta
    return delta

  def pop(self):
    """Removes one element from the queue."""
    with self._lock:
      self._value -= 1
      self._total += 1
      if self._value < 0:
        raise ValueError('Queue is already empty.')

  def total(self):
    return self._total


@threading_utils.local_attributes(['_environment'])
class AsyncRunner(run_experiment.Runner):
  """Defines a train runner for asynchronous training."""

  def __init__(
      self, base_dir, create_agent_fn,
      create_environment_fn=atari_lib.create_atari_environment,
      max_simultaneous_iterations=1, **kwargs):
    """Creates an asynchronous runner.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
      max_simultaneous_iterations: int, maximum number of iterations running
        simultaneously in separate threads.
      **kwargs: Additional positional arguments.
    """
    threading_utils.initialize_local_attributes(
        self, _environment=create_environment_fn)
    self._running_iterations = _ThreadSafeQueue(max_simultaneous_iterations)
    self._output_lock = threading.Lock()
    super(AsyncRunner, self).__init__(
        base_dir=base_dir, create_agent_fn=create_agent_fn,
        create_environment_fn=create_environment_fn, **kwargs)

  def _initialize_session(self):
    """Creates a tf.Session that supports GPU usage in multiple threads."""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    self._sess = tf.Session('', config=config)

  # TODO(aarg): Decouple experience generation from training.
  def _run_experiment_loop(self):
    """Runs iterations in multiple threads until `num_iterations` is reached."""
    self._running_iterations.reset()
    started_iterations = 0
    while self._running_iterations.total() < self._num_iterations:
      iterations_to_start = self._running_iterations.fill(
          self._num_iterations - started_iterations)
      for i in range(iterations_to_start):
        self._run_one_iteration(started_iterations + i)
      started_iterations += iterations_to_start
      if not started_iterations:
        time.sleep(_SLEEP_SECONDS)

  @async_method
  def _run_one_iteration(self, iteration):
    statistics = super(AsyncRunner, self)._run_one_iteration(iteration)
    with self._output_lock:
      self._log_experiment(iteration, statistics)
      self._checkpoint_experiment(iteration)
    self._running_iterations.pop()
