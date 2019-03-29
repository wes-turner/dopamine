# coding=utf-8
# Copyright 2019 Google Inc. All Rights Reserved.
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
"""Module to decouple object attributes and make them local to threads.

This module's public interface is a `local_attributes` class decorator that
defines a custom getter, setter and deleter for a list of specified class
properties to allow an object to store variables using that property name whose
values are isolated to a specific thread. A user interacts with attributes
through these getter, setter and deleter methods as though they're regular
object attributes. Internally, they are methods that wrap object variables that
are specific to the id of the thread they're accessed from. We call those
internal attributes the "thread local" version of the class attribute.

Each attribute "attr" has a callable default value that initializes a thread
local variable whenever a caller tries to get the class attribute from an object
and the thread local version does not exist in an object's dictionary.
Typically, this is the first time in a thread that the variable is accessed from
that object.

Cases where this does not hold:

  - If the thread itself is new but uses a pre-existing thread id, the variable
    may have already existed earlier, and the default will not be called.

  - If the variable var has at some point been unbound from the class cls (say
    by calling `del cls.var`) the initializer will again be called if cls.var is
    later accessed.

To set these default values, use the `initialize_local_attributes` helper of
this module.

Example of usage:
  ```python
  @local_attributes(['attr'])
  class MyClass(object):

    def __init__(self, attr_default_value):
      initialize_local_attributes(self, attr=lambda: attr_default_value)

  obj = MyClass('default-value')
  assert obj.attr == 'default-value'
  ```

WARNING (reminder): `local_attributes` localizes to threads via thread ID. It
cannot tell whether two instances of a thread having the same id are the same
thread or are different threads.

WARNING (edge case): Because `@local_attributes(['attr'])` defines a method
`MyClass.attr` in the wrapper, any class method or variable as defined in the
definition of `MyClass` will be overwritten by accessor functions to a
thread-local `attr`. This overriding behavior is similar to assignment of a
variable within an object, but takes place at the class level rather than at the
object level.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading


def _get_internal_name(name):
  """Returns the internal thread-local name of an attribute based on its id.

  For each specified attribute, we create an attribute whose name depends on
  the current thread's id to store thread-local value for that attribute. This
  method provides the name of this thread-specific attribute.

  Args:
    name: str, name of the exposed attribute.
  Returns:
    str, name of the internal attribute storing thread local value.
  """
  return '__' + name + '_' + str(threading.current_thread().ident)


def _get_default_value_name(name):
  """Returns the class default value of an attribute.

  For each specified attribute, we create a default value that is class-specific
  and not thread-specific. This class-wide default value is stored in an
  internal attribute that is named `_attr_default` where `attr` is the name of
  the specified attribute.

  Args:
    name: str, name of the exposed attribute.
  Returns:
    str, name of the internal attribute storing thread local value.
  """
  return '_' + name + '_default'


def _add_property(cls, attr_name):
  """Adds a property to a given class.

  The setter, getter and deleter are added to the given class and correspond to
  the provided attribute name.

  These methods actually apply to an internal variable that is thread-local.
  Hence the result of these methods depends on the local thread's id.

  Note that when the getter is called and the local attribute is not found the
  getter will initialize the local value to the class default value.

  See `initialize_local_attributes` for more details.

  Args:
    cls: A class to add the property to.
    attr_name: str, name of the property to create.
  """
  def _set(self, val):
    """Defines a custom setter for the thread-local attribute.

    This setter assigns the value to the internal local variable.

    Args:
      self: The object the thread-local attr_name will be written to.
      val: The value to assign to the attribute.
    """
    setattr(self, _get_internal_name(attr_name), val)

  def _get(self):
    """Defines a custom getter for the attribute.

    This getter reads and returns the internal, thread-local variable. If this
    local variable has does not exist in the object's dictionary, the getter
    will look for the class initializer for the attribute and initialize the
    local variable.

    Args:
      self: The object with the dict attr_name will be resolved from.

    Returns:
      The value of the thread local attribute.

    Raises:
      AttributeError: If the local variable has not been set and no class
        initializer was specified.
      AttributeError: If the specified class initializer is not callable.
    """
    if not hasattr(self, _get_internal_name(attr_name)):
      if not hasattr(self, _get_default_value_name(attr_name)):
        raise AttributeError(
            'Local value for attribute `{}` has not been set. You can '
            'initialize it locally (`self.{} = <initial-value>`) or set a '
            'global value using the `initialize_local_attributes` '
            'helper.'.format(attr_name, attr_name))
      default_attr_fn = getattr(self, _get_default_value_name(attr_name))
      if not callable(default_attr_fn):
        raise AttributeError('Default value initializer must be callable.')
      _set(self, default_attr_fn())
    return getattr(self, _get_internal_name(attr_name))

  def _del(self):
    """Defines a custom deleter that deletes the thread-local variable."""
    delattr(self, _get_internal_name(attr_name))

  setattr(cls, attr_name, property(_get, _set, _del))


def local_attributes(attributes):
  """Creates a decorator that adds properties to the decorated class.

  See the module docstring for a detailed discussion of this function's usage
  and semantics.

  Args:
    attributes: List[str], names of the wrapped attributes to add to the class.
  Returns:
    A class decorator.
  """
  def _decorator(cls):
    for attr_name in attributes:
      _add_property(cls, attr_name)
    return cls
  return _decorator


def initialize_local_attributes(obj, **kwargs):
  """Sets global default values for local attributes.

  Each class attribute has a global default value initializer and local values
  that are specific to each thread.

  If an attribute's thread-local name doesn't exist in the object's dictionary
  (perhaps because this is the first time the attribute's getter is called for a
  given thread id), it is initialized by calling the class default value
  initializer.

  This helper function lets a user define those default value initializers.

  Args:
    obj: The object that has the local attributes.
    **kwargs: The default callable initializer for each local attribute.
  Raises:
    AttributeError: If a value has already been assigned to the thread-local
      attribute when its initializer is called.
  """
  for key, val in kwargs.items():
    default_attr = _get_default_value_name(key)
    if hasattr(obj, default_attr):
      raise AttributeError(
          'Object `{}` already has a `{}` attribute.'.format(obj, default_attr))
    setattr(obj, default_attr, val)
