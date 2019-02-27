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
"""Tests for `threading_utils` integration with `DQNAgent`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from dopamine.agents.dqn import dqn_agent
from dopamine.utils import test_utils
import numpy as np
import tensorflow as tf
from tensorflow import test


class DQNIntegrationTest(test.TestCase, parameterized.TestCase):
  """Integration test for DQNAgent and threading utils."""

  def testBundling(self):
    """Tests that local values are poperly updated when reading a checkpoint."""
    with tf.Session() as sess:
      agent = dqn_agent.DQNAgent(sess, 3, observation_shape=(2, 2))
      sess.run(tf.global_variables_initializer())
      agent.state = 'state_val'
      bundle = agent.bundle_and_checkpoint(
          self.get_temp_dir(), iteration_number=10)
      self.assertIn('state', bundle)
      self.assertEqual(bundle['state'], 'state_val')
      bundle['state'] = 'new_state_val'

      with test_utils.mock_thread('other-thread'):
        agent.unbundle(
            self.get_temp_dir(), iteration_number=10, bundle_dictionary=bundle)
        self.assertEqual(agent.state, 'new_state_val')
      self.assertEqual(agent.state, 'state_val')

  def testLocalValues(self):
    """Tests that episode related variables are thread specific."""
    with tf.Session() as sess:
      observation_shape = (2, 2)
      agent = dqn_agent.DQNAgent(
          sess, 3, observation_shape=observation_shape)
      sess.run(tf.global_variables_initializer())

      with test_utils.mock_thread('baseline-thread'):
        agent.begin_episode(
            observation=np.zeros(observation_shape), training=False)
        local_values_1 = (agent._observation, agent._last_observation,
                          agent.state)

      with test_utils.mock_thread('different-thread'):
        agent.begin_episode(
            observation=np.zeros(observation_shape), training=False)
        agent.step(
            reward=10, observation=np.ones(observation_shape), training=False)
        local_values_3 = (agent._observation, agent._last_observation,
                          agent.state)

      with test_utils.mock_thread('identical-thread'):
        agent.begin_episode(
            observation=np.zeros(observation_shape), training=False)
        local_values_2 = (agent._observation, agent._last_observation,
                          agent.state)

      # Asserts that values in 'identical-thread' are same as baseline.
      for val_1, val_2 in zip(local_values_1, local_values_2):
        self.assertTrue(np.all(val_1 == val_2))

      # Asserts that values in 'different-thread' are differnt from baseline.
      for val_1, val_3 in zip(local_values_1, local_values_3):
        self.assertTrue(np.any(val_1 != val_3))

  @parameterized.parameters([('_last_observation', None),
                             ('_observation', None),
                             ('state', np.zeros((1, 2, 2, 4))),
                             ('eval_mode', False)])
  def testLocalVariablesSet(self, variable_name, expected_value):
    agent = dqn_agent.DQNAgent(
        tf.Session(), 3, observation_shape=(2, 2), stack_size=4)
    setattr(agent, variable_name, 'dummy-value')
    with test_utils.mock_thread('thread'):
      self.assertAllEqual(getattr(agent, variable_name), expected_value)

  def testActionIsNotDefined(self):
    agent = dqn_agent.DQNAgent(tf.Session(), 3, observation_shape=(2, 2))
    agent.action = 'dummy-value'
    with test_utils.mock_thread('thread'):
      with self.assertRaisesRegexp(
          AttributeError,
          'Local value for attribute `action` has not been set.*'):
          _ = agent.action


if __name__ == '__main__':
  test.main()
