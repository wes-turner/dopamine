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

from dopamine.utils import test_utils
from dopamine.utils import threading_utils
from tensorflow import test


_DummyClass = type('DummyClass', (object,), {})


class ThreadingUtilsAPITest(test.TestCase):
  """Unit tests for threading utils."""

  def testDefaultValueAlreadyExists(self):
    """Tests that an error is raised when overriding existing default value."""
    obj = _DummyClass()
    threading_utils.initialize_local_attributes(
        obj, attr=lambda: 'existing-default-value')
    with self.assertRaisesRegexp(
        AttributeError, 'Object `.*` already has .* attribute.'):
      threading_utils.initialize_local_attributes(
          obj, attr=lambda: 'new-default-value')

  def testAttributeNotInitialized(self):
    """Tests that error is raised when local value has not been initialized."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    with self.assertRaisesRegexp(
        AttributeError, 'Local value for attribute `attr` has not been set.*'):
      # Calling the attribute is expected to initialize it with the default
      # value. Hence the pointless statement to run the getter.
      _ = obj.attr

  def testDefaultAttributeIsNotCallable(self):
    """Tests the error raised when provided default attribute not callable."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    threading_utils.initialize_local_attributes(obj, attr='default-value')
    with self.assertRaisesRegexp(
        AttributeError, 'Default value initializer must be callable.'):
      # Calling the attribute is expected to initialize it with the default
      # value. Hence the pointless statement to run the getter.
      _ = obj.attr

  def testDefaultValueIsUsed(self):
    """Tests that the default value is properly read in thread."""
    obj = _DummyClass()
    threading_utils.initialize_local_attributes(
        obj, attr=lambda: 'default-value')
    obj.attr = 'dummy_value'
    with test_utils.mock_thread('thread'):
      self.assertEqual(obj.attr, 'default-value')

  def testMultipleDefaultValuesAreUsed(self):
    """Tests that multiple default values are properly set by the helper."""
    obj = _DummyClass()
    threading_utils.initialize_local_attributes(
        obj, attr1=lambda: 3, attr2=lambda: 4)
    obj.attr1 = 'dummy_value'
    obj.attr2 = 'dummy_value'
    with test_utils.mock_thread('thread'):
      self.assertEqual(obj.attr1, 3)
      self.assertEqual(obj.attr2, 4)

  def testCallableAttribute(self):
    """Tests that internal value is properly called with callable attribute."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    with test_utils.mock_thread('thread'):
      obj.attr = test.mock.Mock()
    obj.attr = test.mock.Mock()
    with test_utils.mock_thread('thread'):
      obj.attr.method()
      obj.attr.method.assert_called_once()
    obj.attr.method.assert_not_called()

  def testInternalAttributeIsInitializedOnce(self):
    """Tests that getter properly initializes the local value on first call."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    mock_value = test.mock.Mock()
    mock_callable_init = test.mock.Mock(return_value=mock_value)
    threading_utils.initialize_local_attributes(obj, attr=mock_callable_init)
    # Calling the attribute is expected to initialize it with the default value.
    # Hence the pointless statement to run the getter.
    _ = obj.attr
    mock_callable_init.assert_called_once()
    mock_value.assert_not_called()
    obj.attr()
    mock_callable_init.assert_called_once()
    mock_value.assert_called_once()

  def testInternalAttributeIsUsed(self):
    """Tests that setter/getter properly uses the internal value."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    with test_utils.mock_thread('thread'):
      obj.attr = 'internal-value'
    obj.attr = 'dummy_value'
    with test_utils.mock_thread('thread'):
      self.assertEqual(obj.attr, 'internal-value')

  def testLocalValueOverDefault(self):
    """Tests that getter uses internal value over default one."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    mock_default_init = test.mock.Mock()
    threading_utils.initialize_local_attributes(obj, attr=mock_default_init)
    with test_utils.mock_thread('thread'):
      obj.attr = 'internal-value'
    obj.attr = 'dummy_value'
    with test_utils.mock_thread('thread'):
      self.assertEqual(obj.attr, 'internal-value')
    mock_default_init.assert_not_called()

  def testMultipleAttributes(self):
    """Tests the class decorator with multiple local attributes."""
    MockClass = threading_utils.local_attributes(
        ['attr1', 'attr2'])(_DummyClass)
    obj = MockClass()
    with test_utils.mock_thread('thread'):
      obj.attr1 = 10
      obj.attr2 = 20
    obj.attr1 = obj.attr = 'dummy_value'
    obj.attr2 = obj.attr = 'dummy_value'
    with test_utils.mock_thread('thread'):
      self.assertEqual(obj.attr1, 10)
      self.assertEqual(obj.attr2, 20)

  def testMultiThreads(self):
    """Tests that different threads create different local attributes."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    # Initializes attribute in thread 1.
    with test_utils.mock_thread('thread_1'):
      obj.attr = 1
    # Initializes attribute in thread 2.
    with test_utils.mock_thread('thread_2'):
      obj.attr = 2
    # Reads attribute in thread 1.
    with test_utils.mock_thread('thread_1'):
      self.assertEqual(obj.attr, 1)
    # Reads attribute in thread 2.
    with test_utils.mock_thread('thread_2'):
      self.assertEqual(obj.attr, 2)

  def testMultiThreadsMultipleAttributes(self):
    """Tests that different threads create different local attributes."""
    MockClass = threading_utils.local_attributes(
        ['attr1', 'attr2'])(_DummyClass)
    obj = MockClass()
    # Initializes attribute in thread 1.
    with test_utils.mock_thread('thread_1'):
      obj.attr1 = 1
      obj.attr2 = 2
    with test_utils.mock_thread('thread_2'):
      obj.attr1 = 3
      obj.attr2 = 4
    with test_utils.mock_thread('thread_1'):
      self.assertEqual(obj.attr1, 1)
      self.assertEqual(obj.attr2, 2)
    with test_utils.mock_thread('thread_2'):
      self.assertEqual(obj.attr1, 3)
      self.assertEqual(obj.attr2, 4)


class ThreadingUtilsImplementationTest(test.TestCase):
  """Tests specific to `threading_utils.py` implementation."""

  def testGetInternalName(self):
    """Tests that the name of the internal attribute has proper format."""
    with test_utils.mock_thread(123):
      self.assertEqual(threading_utils._get_internal_name('attr'), '__attr_123')

  def testGetDefaultValueName(self):
    """Tests that the name of the internal attribute has proper format."""
    self.assertEqual(
        threading_utils._get_default_value_name('attr'),
        '_attr_default')


if __name__ == '__main__':
  test.main()
