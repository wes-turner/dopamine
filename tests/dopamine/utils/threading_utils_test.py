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
"""Tests for threading_utils.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import threading

from dopamine.agents.dqn import dqn_agent
from dopamine.utils import threading_utils
import tensorflow as tf
from tensorflow import test


def _get_internal_name(attr):
  thread_id = str(threading.current_thread().ident)
  return '__' + attr + '_' + thread_id


_DummyClass = type('DummyClass', (object,), {})


class ThreadsTest(test.TestCase):
  """Unit tests for threading utils."""

  def testDefaultValueAlreadyExists(self):
    """Tests that an error is raised when overriding existing default value."""
    obj = _DummyClass()
    obj._attr_default = 'existing-default-value'
    with self.assertRaisesRegexp(
        AttributeError, 'Object `.*` already has .* attribute.'):
      threading_utils.initialize_local_attributes(obj, attr='new-default-value')

  def testAttributeNotInitialized(self):
    """Tests that error is raised when local value has not been initialized."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    with self.assertRaisesRegexp(
        AttributeError, 'Local value for attribute `attr` has not been set.*'):
      # Calling the attribute is expected to initialize it with the default
      # value. Hence the pointless statement to run the getter.
      obj.attr  # pylint: disable=pointless-statement

  def testDefaultValueIsAdded(self):
    """Tests that the default value is properly set by the helper."""
    obj = _DummyClass()
    threading_utils.initialize_local_attributes(obj, attr=3)
    self.assertEqual(obj._attr_default, 3)

  def testMultipleDefaultValuesAreSet(self):
    """Tests that multiple default values are properly set by the helper."""
    obj = _DummyClass()
    threading_utils.initialize_local_attributes(obj, attr1=3, attr2=4)
    self.assertEqual(obj._attr1_default, 3)
    self.assertEqual(obj._attr2_default, 4)

  def testAttributeDefaultValueIsCalled(self):
    """Tests that getter properly uses the default value."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    obj._attr_default = 'default-value'
    self.assertEqual(obj.attr, 'default-value')

  def testDefaultValueIsRead(self):
    """Tests that getter properly initializes the local value."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    obj._attr_default = 'default-value'
    # Calling the attribute is expected to initialize it with the default value.
    # Hence the pointless statement to run the getter.
    obj.attr  # pylint: disable=pointless-statement
    self.assertEqual(getattr(obj, _get_internal_name('attr')), 'default-value')

  def testInternalAttributeIsRead(self):
    """Tests that getter properly uses the internal value."""
    MockClass = threading_utils.local_attributes(['attr'])(
        _DummyClass)
    obj = MockClass()
    setattr(obj, _get_internal_name('attr'), 'intenal-value')
    self.assertEqual(obj.attr, 'intenal-value')

  def testInternalAttributeIsSet(self):
    """Tests that setter properly sets the internal value."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    obj.attr = 'internal-value'
    self.assertEqual(getattr(obj, _get_internal_name('attr')), 'internal-value')

  def testInternalValueOverDefault(self):
    """Tests that getter uese internal value over default one."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    obj._attr_default = 'default-value'
    setattr(obj, _get_internal_name('attr'), 'internal-value')
    self.assertEqual(obj.attr, 'internal-value')

  def testMultipleAttributes(self):
    """Tests the class decorator with multiple local attributes."""
    MockClass = threading_utils.local_attributes(
        ['attr1', 'attr2'])(_DummyClass)
    obj = MockClass()
    obj.attr1 = 10
    obj.attr2 = 20
    setattr(obj, _get_internal_name('attr1'), 1)
    setattr(obj, _get_internal_name('attr2'), 2)
    self.assertEqual(obj.attr1, 1)
    self.assertEqual(obj.attr2, 2)

  def testCallableAttribute(self):
    """Tests that internal value is properly called with callable attribute."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    internal_attr = test.mock.Mock()
    setattr(obj, _get_internal_name('attr'), internal_attr)
    obj.attr.callable_method()
    internal_attr.callable_method.assert_called_once()

  def testMultiThreads(self):
    """Tests that different threads create different local attributes."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    internal_name_method = 'dopamine.utils.threading_utils._get_internal_name'
    # Initializes attribute in thread 1.
    with test.mock.patch(internal_name_method, return_value='thread_1'):
      obj.attr = 1
    # Initializes attribute in thread 2.
    with test.mock.patch(internal_name_method, return_value='thread_2'):
      obj.attr = 2
    # Reads attribute in thread 1.
    with test.mock.patch(internal_name_method, return_value='thread_1'):
      self.assertEqual(obj.attr, 1)
    # Reads attribute in thread 2.
    with test.mock.patch(internal_name_method, return_value='thread_2'):
      self.assertEqual(obj.attr, 2)
    # Checks internal variables.
    self.assertEqual(getattr(obj, 'thread_1'), 1)
    self.assertEqual(getattr(obj, 'thread_2'), 2)


class DQNIntegrationTest(test.TestCase):
  """Integration test for DQNAgent and threading utils."""

  def testBundling(self):
    """Tests that local values are poperly updated when reading a checkpoint."""
    with tf.Session() as sess:
      agent = agent = dqn_agent.DQNAgent(sess, 3, observation_shape=(2, 2))
      sess.run(tf.global_variables_initializer())
      agent.state = 'state_val'
      self.assertEqual(
          getattr(agent, _get_internal_name('state')), 'state_val')
      test_dir = tempfile.mkdtemp()
      bundle = agent.bundle_and_checkpoint(test_dir, iteration_number=10)
      self.assertIn('state', bundle)
      self.assertEqual(bundle['state'], 'state_val')
      bundle['state'] = 'new_state_val'

      agent.unbundle(test_dir, iteration_number=10, bundle_dictionary=bundle)
      self.assertEqual(agent.state, 'new_state_val')
      self.assertEqual(
          getattr(agent, _get_internal_name('state')), 'new_state_val')


if __name__ == '__main__':
  test.main()
