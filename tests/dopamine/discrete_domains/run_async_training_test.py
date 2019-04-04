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
"""Tests for async trainer in `run_experiment_test.py`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from dopamine.discrete_domains import run_experiment
from dopamine.utils import test_utils
import queue
import tensorflow as tf
from tensorflow import test


def _get_mock_environment_fn():
  mock_env = test.mock.Mock()
  mock_env.step.return_value = (0, 0, True, {})
  return test.mock.MagicMock(return_value=mock_env)


def _wrap_method_with_mock_call(method, mock_method):
  def _wrapped_method(self, *args, **kwargs):
    mock_method(*args, **kwargs)
    return method(self, *args, **kwargs)
  return _wrapped_method


class AsyncRunnerTest(test.TestCase, parameterized.TestCase):
  """Tests for asynchronous trainer."""

  def _get_runner(self, **kwargs):
    runner = run_experiment.AsyncRunner(
        base_dir=self.get_temp_dir(), **kwargs)
    runner._checkpoint_experiment = test.mock.Mock()
    runner._log_experiment = test.mock.Mock()
    runner._save_tensorboard_summaries = test.mock.Mock()
    return runner

  def testEnvironmentInitializationPerThread(self):
    """Tests that a new environment is created for a new thread.

    In synchronous model `create_environment_fn` is called only once at the
    runner initialization. In synchronous model, `create_environment_fn` is
    called for each new iteration.
    """
    environment_fn = _get_mock_environment_fn()
    runner = self._get_runner(
        create_agent_fn=test.mock.MagicMock(),
        create_environment_fn=environment_fn, num_iterations=1,
        training_steps=1, evaluation_steps=0, max_simultaneous_iterations=1)

    # Environment called once in init.
    environment_fn.assert_called_once()
    with test_utils.mock_thread('other-thread'):
      runner.run_experiment()
    runner.run_experiment()
    self.assertEqual(environment_fn.call_count, 3)

  def testNumIterations(self):
    mock_agent = test.mock.Mock()
    agent_fn = test.mock.MagicMock(return_value=mock_agent)
    runner = self._get_runner(
        create_agent_fn=agent_fn,
        create_environment_fn=_get_mock_environment_fn(), num_iterations=18,
        training_steps=1, evaluation_steps=0, max_simultaneous_iterations=1)
    runner.run_experiment()
    self.assertEqual(mock_agent.begin_episode.call_count, 18)

  def testNumberTrainingSteps(self,):
    """Tests that the right number of training steps are ran."""
    mock_put = test.mock.Mock()
    put = _wrap_method_with_mock_call(queue.Queue.put, mock_put)
    with test.mock.patch.object(queue.Queue, 'put', put):
      runner = self._get_runner(
          create_agent_fn=test.mock.MagicMock(),
          create_environment_fn=_get_mock_environment_fn(), num_iterations=3,
          training_steps=2, evaluation_steps=6, max_simultaneous_iterations=1)
      runner.run_experiment()

    def _put_call_cnt(v):
      return sum([list(call)[0] == v for call in mock_put.call_args_list])

    self.assertEqual(_put_call_cnt(('train',)), 3)
    self.assertEqual(_put_call_cnt(('eval',)), 3)
    self.assertEqual(_put_call_cnt((0,)), 6)
    self.assertEqual(_put_call_cnt((None,)), 1)

  def testNumberSteps(self):
    """Tests that the right number of agent steps are ran."""
    agent = test.mock.Mock()
    agent_fn = test.mock.MagicMock(return_value=agent)
    runner = self._get_runner(
        create_agent_fn=agent_fn,
        create_environment_fn=_get_mock_environment_fn(), num_iterations=3,
        training_steps=2, evaluation_steps=6, max_simultaneous_iterations=1)
    runner.run_experiment()
    self.assertEqual(agent.begin_episode.call_count, 24)

  @test.mock.patch.object(tf, 'Summary')
  def testSummariesExportedWithProperTags(self, summary):
    runner = run_experiment.AsyncRunner(
        base_dir=self.get_temp_dir(), create_agent_fn=test.mock.MagicMock(),
        create_environment_fn=_get_mock_environment_fn(),
        num_iterations=2, training_steps=1, evaluation_steps=0,
        max_simultaneous_iterations=2)
    runner._checkpoint_experiment = test.mock.Mock()
    runner._log_experiment = test.mock.Mock()
    runner._summary_writer = test.mock.Mock()
    runner.run_experiment()
    self.assertCountEqual(
        summary.Value.call_args_list,
        [test.mock.call(simple_value=0, tag='Eval/NumEpisodes'),
         test.mock.call(simple_value=0, tag='Eval/AverageReturns'),
         test.mock.call(simple_value=1, tag='Train/NumEpisodes'),
         test.mock.call(simple_value=0, tag='Train/AverageReturns'),
         test.mock.call(simple_value=1, tag='Train/NumEpisodes'),
         test.mock.call(simple_value=0, tag='Train/AverageReturns'),])


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

  def testCompletedIterationCounterIsUsed(self,):
    self.runner._completed_iteration = 20
    self.runner._experience_queue.put(1)
    self.runner._run_one_iteration(iteration=36, eval_mode=False)
    self.runner._checkpoint_experiment.assert_called_once_with(20)

  def testCompletedIterationCounterIsInitialized(self):
    self.runner.run_experiment()
    self.runner._checkpoint_experiment.assert_called_once_with(0)

  def testCompletedIterationCounterIsIncremented(self):
    self.runner._completed_iteration = 20
    self.runner._experience_queue.put(1)
    self.runner._run_one_iteration(iteration=36, eval_mode=False)
    self.assertEqual(self.runner._completed_iteration, 21)


if __name__ == '__main__':
  test.main()
