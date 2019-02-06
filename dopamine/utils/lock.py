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
"""Creates a function decorator to protect execution with a lock."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import threading

_DEFAULT_LOCK_ATTR_NAME = '_lock'


# TODO(aarg): Add a lock implementation that lets read operation occur at the
# same time (i.e. less restrictive).


def initialize_lock(object_to_lock, lock=threading.Lock(),
                    lock_attribute_name=_DEFAULT_LOCK_ATTR_NAME):
  """Initializes a lock attribute for a specified object.

  Args:
    object_to_lock: object to which the lock applies.
    lock: lock object to use for protection against concurrent access.
    lock_attribute_name: str, name of the lock attribute to assign to
      `object_to_lock`.

  Raises:
    AttributeError: If `object_to_lock` already has an attribute named
      `lock_attribute_name`.
  """
  if hasattr(object_to_lock, lock_attribute_name):
    raise AttributeError(
        'Object already has a `{}` attribute.'.format(lock_attribute_name))
  setattr(object_to_lock, lock_attribute_name, lock)


def locked_method(lock_attribute_name=_DEFAULT_LOCK_ATTR_NAME):
  """Creates decorator to apply lock to class methods.

  Args:
    lock_attribute_name: str, name of the instance attribute to use as a lock.

  Returns:
    A decorator function.
  """
  def _decorator(fn):
    """Wraps a class's method so it's locked.

    Args:
      fn: Object's method with the following signature:
        * Args:
          * self: Instance of the class.
          * *args: Additional positional arguments.
          * **kwargs: Additional keyword arguments.
        Note that the instance must have a `_lock` attribute.

    Returns:
      A function with same signature as the input function.

    Raises:
      AttributeError: If object doesn't have a lock attribute.
    """
    @functools.wraps(fn)
    def _decorated(self, *args, **kwargs):
      if not hasattr(self, lock_attribute_name):
        raise AttributeError(
            'Object {} expected to have a `{}` attribute.'.format(
                self, lock_attribute_name))
      lock = getattr(self, lock_attribute_name, None)
      if lock is None:
        return fn(self, *args, **kwargs)
      with lock:
        return fn(self, *args, **kwargs)
    return _decorated
  return _decorator
