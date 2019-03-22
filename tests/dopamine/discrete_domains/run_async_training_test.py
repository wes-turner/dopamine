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
"""Tests for run_experiment_test.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from dopamine.discrete_domains import run_experiment
from dopamine.utils import test_utils
from tensorflow import test


def _get_mock_environment_fn():
  mock_env = test.mock.Mock()
  mock_env.step.return_value = (0, 0, True, {})
  return test.mock.MagicMock(return_value=mock_env)


class AsyncRunnerTest(test.TestCase):
  """Tests for asynchronous trainer."""

  def testEnvironmentInitializationPerThread(self):
    """Tests that a new environment is created for a new thread.

    In synchronous model `create_environment_fn` is called only once at the
    runner initialization. In synchronous model, `create_environment_fn` is
    called for each new iteration.
    """
    environment_fn = _get_mock_environment_fn()

    runner = run_experiment.AsyncRunner(
        base_dir=self.get_temp_dir(), create_agent_fn=test.mock.MagicMock(),
        create_environment_fn=environment_fn, num_iterations=1,
        training_steps=1, evaluation_steps=0, max_simultaneous_iterations=1)
    runner._checkpoint_experiment = test.mock.Mock()
    runner._log_experiment = test.mock.Mock()
    # Environment called once in init.
    environment_fn.assert_called_once()
    with test_utils.mock_thread('other-thread'):
      runner.run_experiment()
    runner.run_experiment()
    self.assertEqual(environment_fn.call_count, 3)

  def testNumIterations(self):
    mock_agent = test.mock.Mock()
    agent_fn = test.mock.MagicMock(return_value=mock_agent)
    runner = run_experiment.AsyncRunner(
        base_dir=self.get_temp_dir(), create_agent_fn=agent_fn,
        create_environment_fn=_get_mock_environment_fn(), num_iterations=18,
        training_steps=1, evaluation_steps=0, max_simultaneous_iterations=1)
    runner._checkpoint_experiment = test.mock.Mock()
    runner._log_experiment = test.mock.Mock()
    runner._save_tensorboard_summaries = test.mock.Mock()
    runner.run_experiment()
    self.assertEqual(mock_agent.begin_episode.call_count, 18)

  @test.mock.patch.object(threading, 'Semaphore')
  def testMultipleIterationManagement(self, semaphore):
    mock_semaphore = test.mock.Mock()
    semaphore.return_value = mock_semaphore
    runner = run_experiment.AsyncRunner(
        base_dir=self.get_temp_dir(), create_agent_fn=test.mock.MagicMock(),
        create_environment_fn=_get_mock_environment_fn(), num_iterations=1,
        training_steps=1, evaluation_steps=0, max_simultaneous_iterations=1)
    runner._checkpoint_experiment = test.mock.Mock()
    runner._log_experiment = test.mock.Mock()
    runner._save_tensorboard_summaries = test.mock.Mock()
    runner.run_experiment()
    self.assertEqual(mock_semaphore.acquire.call_count, 2)
    self.assertEqual(mock_semaphore.release.call_count, 2)


class InternalIterationCounterTest(test.TestCase):
  """Tests for the iteration internal counter."""

  def setUp(self):
    runner = run_experiment.AsyncRunner(
        base_dir=self.get_temp_dir(), create_agent_fn=test.mock.MagicMock(),
        create_environment_fn=_get_mock_environment_fn(), num_iterations=1,
        training_steps=1, evaluation_steps=0, max_simultaneous_iterations=1)
    runner._checkpoint_experiment = test.mock.Mock()
    runner._log_experiment = test.mock.Mock()
    runner._save_tensorboard_summaries = test.mock.Mock()
    self.runner = runner
    super(InternalIterationCounterTest, self).setUp()

  def testCounterUsed(self):
    self.runner._completed_iteration = 20
    self.runner._run_one_iteration(36, test.mock.Mock(), False).join()
    self.runner._checkpoint_experiment.assert_called_once_with(20)

  def testCounterInitialized(self):
    self.runner.run_experiment()
    self.runner._checkpoint_experiment.assert_called_once_with(0)

  def testCounterIncremented(self):
    self.runner._completed_iteration = 20
    self.runner._run_one_iteration(36, test.mock.Mock(), False).join()
    self.assertEqual(self.runner._completed_iteration, 21)


if __name__ == '__main__':
  test.main()
