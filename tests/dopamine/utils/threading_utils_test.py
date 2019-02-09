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
    """Tests that the default value is properly set by the helper."""
    obj = _DummyClass()
    threading_utils.initialize_local_attributes(
        obj, attr=lambda: 'default-value')
    self.assertEqual(obj.attr, 'default-value')

  def testMultipleDefaultValuesAreUsed(self):
    """Tests that multiple default values are properly set by the helper."""
    obj = _DummyClass()
    threading_utils.initialize_local_attributes(
        obj, attr1=lambda: 3, attr2=lambda: 4)
    self.assertEqual(obj.attr1, 3)
    self.assertEqual(obj.attr2, 4)

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

  def testCallableAttribute(self):
    """Tests that different attributes are initialized in each thread.."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    threading_utils.initialize_local_attributes(
        obj, attr=lambda: test.mock.Mock())  # pylint: disable=unnecessary-lambda
    # Call attribute in thread 1.
    with test_utils.mock_thread('thread_1'):
      obj.attr()
    # Call attribute in thread 2.
    with test_utils.mock_thread('thread_2'):
      obj.attr()

    # Check that attribute was called once in thread 1.
    with test_utils.mock_thread('thread_1'):
      obj.attr.assert_called_once()

    # Check that attribute was called once in thread 2.
    with test_utils.mock_thread('thread_2'):
      obj.attr.assert_called_once()

    # Check that attribute was not called in thread 3.
    with test_utils.mock_thread('thread_3'):
      obj.attr.assert_not_called()


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

  def testDefaultValueIsSet(self):
    """Tests that the default value is properly set by the helper."""
    obj = _DummyClass()
    threading_utils.initialize_local_attributes(obj, attr=lambda: 3)
    self.assertEqual(obj._attr_default(), 3)

  def testAttributeDefaultValueIsRead(self):
    """Tests that getter properly uses the default value."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    obj._attr_default = lambda: 'default-value'
    self.assertEqual(obj.attr, 'default-value')

  def testInternalAttributeIsInitialized(self):
    """Tests that getter properly initializes the local value."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    obj._attr_default = lambda: 'default-value'
    # Calling the attribute is expected to initialize it with the default value.
    # Hence the pointless statement to run the getter.
    _ = obj.attr
    self.assertEqual(getattr(obj, threading_utils._get_internal_name('attr')),
                     'default-value')

  def testInternalAttributeIsRead(self):
    """Tests that getter properly uses the internal value."""
    MockClass = threading_utils.local_attributes(['attr'])(
        _DummyClass)
    obj = MockClass()
    setattr(obj, threading_utils._get_internal_name('attr'), 'internal-value')
    self.assertEqual(obj.attr, 'internal-value')

  def testInternalAttributeIsSet(self):
    """Tests that setter properly sets the internal value."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    obj.attr = 'internal-value'
    self.assertEqual(getattr(obj, threading_utils._get_internal_name('attr')),
                     'internal-value')

  def testLocalValueOverDefault(self):
    """Tests that getter uese internal value over default one."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    obj._attr_default = lambda: 'default-value'
    setattr(obj, threading_utils._get_internal_name('attr'), 'internal-value')
    self.assertEqual(obj.attr, 'internal-value')

  def testMultipleAttributes(self):
    """Tests the class decorator with multiple local attributes."""
    MockClass = threading_utils.local_attributes(
        ['attr1', 'attr2'])(_DummyClass)
    obj = MockClass()
    obj.attr1 = 10
    obj.attr2 = 20
    setattr(obj, threading_utils._get_internal_name('attr1'), 1)
    setattr(obj, threading_utils._get_internal_name('attr2'), 2)
    self.assertEqual(obj.attr1, 1)
    self.assertEqual(obj.attr2, 2)

  def testCallableAttribute(self):
    """Tests that internal value is properly called with callable attribute."""
    MockClass = threading_utils.local_attributes(['attr'])(_DummyClass)
    obj = MockClass()
    internal_attr = test.mock.Mock()
    setattr(obj, threading_utils._get_internal_name('attr'), internal_attr)
    obj.attr.method()
    internal_attr.method.assert_called_once()


if __name__ == '__main__':
  test.main()
