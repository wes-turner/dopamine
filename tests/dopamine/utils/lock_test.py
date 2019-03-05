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
"""Tests for lock_decorator.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dopamine.utils import lock
from tensorflow import test


class _MockLock(object):
  """Mock lock for testing purposes."""

  def __enter__(self, *args, **kwargs):
    """Locks the lock."""
    raise ValueError('Lock is locked.')

  def __exit__(self, *args, **kwargs):
    pass


class _DummyClass(object):
  """Dummy class to test the lock decorator against."""

  @lock.locked_method()
  def mock_method(self):
    """Dummy method to apply decorator to."""
    pass


class LockDecoratorTest(test.TestCase):
  """Runs tests for lock_decorator function."""

  def testLocksApplies(self):
    """Tests that the lock properly applies to a given function."""

    mock_object = _DummyClass()
    lock.initialize_lock(mock_object, _MockLock())
    with self.assertRaisesRegexp(ValueError, 'Lock is locked.'):
      mock_object.mock_method()

  def testDoesntEnforceLockWhenNone(self):
    """Tests that no lock is enforced when lock is None."""
    mock_object = _DummyClass()
    lock.initialize_lock(mock_object, lock=None)
    mock_object.mock_method()

  def testNoLockAttribute(self):
    """Tests the behavior when the lock is not initialized."""
    mock_object = _DummyClass()
    with self.assertRaisesRegexp(
        AttributeError, r'Object .* expected to have a `_lock` attribute.'):
      mock_object.mock_method()

  def testWrapperFunctionName(self,):
    self.assertEqual(_DummyClass.mock_method.__name__, 'mock_method')


if __name__ == '__main__':
  test.main()
