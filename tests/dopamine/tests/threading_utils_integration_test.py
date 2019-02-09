# coding=utf-8
# Copyright 2018 The Dopamine Authors.
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

import tempfile

from dopamine.agents.dqn import dqn_agent
from dopamine.utils import threading_utils
import tensorflow as tf
from tensorflow import test


class DQNIntegrationTest(test.TestCase):
  """Integration test for DQNAgent and threading utils."""

  def testBundling(self):
    """Tests that local values are poperly updated when reading a checkpoint."""
    with tf.Session() as sess:
      agent = agent = dqn_agent.DQNAgent(sess, 3, observation_shape=(2, 2))
      sess.run(tf.global_variables_initializer())
      agent.state = 'state_val'
      self.assertEqual(
          getattr(agent, threading_utils._get_internal_name('state')),
          'state_val')
      test_dir = tempfile.mkdtemp()
      bundle = agent.bundle_and_checkpoint(test_dir, iteration_number=10)
      self.assertIn('state', bundle)
      self.assertEqual(bundle['state'], 'state_val')
      bundle['state'] = 'new_state_val'

      agent.unbundle(test_dir, iteration_number=10, bundle_dictionary=bundle)
      self.assertEqual(agent.state, 'new_state_val')
      self.assertEqual(
          getattr(agent, threading_utils._get_internal_name('state')),
          'new_state_val')


  if __name__ == '__main__':
    test.main()
