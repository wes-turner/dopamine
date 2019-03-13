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

from dopamine.discrete_domains import run_experiment
from dopamine.utils import test_utils
from tensorflow import test


class AsyncRunnerTest(test.TestCase):
  """Tests for asynchronous trainer."""

  def testEnvironmentInitializationPerThread(self):
    """Tests that a new environment is created for a new thread.

    In synchronous model `create_environment_fn` is called only once at the
    runner initialization. In synchronous model, `create_environment_fn` is
    called for each new iteration.
    """
    mock_env = test.mock.Mock()
    mock_env.step.return_value = (0, 0, True, {})
    environment_fn = test.mock.MagicMock(return_value=mock_env)

    runner = run_experiment.AsyncRunner(
        base_dir=self.get_temp_dir(), create_agent_fn=test.mock.MagicMock(),
        create_environment_fn=environment_fn, num_iterations=1,
        training_steps=1, evaluation_steps=1)
    runner._checkpoint_experiment = test.mock.Mock()
    runner.run_experiment()
    runner.run_experiment()
    # Environment initialized only once in this thread.
    environment_fn.assert_called_once()
    runner.run_experiment()
    with test_utils.mock_thread('other-thread'):
      runner.run_experiment()
    # Environment initialized a second time in 'other-thread'.
    self.assertEqual(environment_fn.call_count, 2)


if __name__ == '__main__':
  test.main()
