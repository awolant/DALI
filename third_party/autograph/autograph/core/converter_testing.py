# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base class for tests in this module."""

import contextlib
import imp
import inspect
import sys

import unittest

import six

from autograph.core import config
from autograph.core import converter
from autograph.impl import api
from autograph.utils import hooks


def allowlist(f):
  """Helper that marks a callable as allowlisted."""
  if 'allowlisted_module_for_testing' not in sys.modules:
    allowlisted_mod = imp.new_module('allowlisted_module_for_testing')
    sys.modules['allowlisted_module_for_testing'] = allowlisted_mod
    config.CONVERSION_RULES = (
        (config.DoNotConvert('allowlisted_module_for_testing'),) +
        config.CONVERSION_RULES)

  f.__module__ = 'allowlisted_module_for_testing'


def is_inside_generated_code():
  """Tests whether the caller is generated code. Implementation-specific."""
  frame = inspect.currentframe()
  try:
    frame = frame.f_back

    internal_stack_functions = ('converted_call', '_call_unconverted')
    # Walk up the stack until we're out of the internal functions.
    while (frame is not None and
           frame.f_code.co_name in internal_stack_functions):
      frame = frame.f_back
    if frame is None:
      return False

    return 'ag__' in frame.f_locals
  finally:
    del frame


class TestingTranspiler(api.PyToLib):
  """Testing version that only applies given transformations."""

  def __init__(self, converters, ag_overrides, operator_overload=hooks.OperatorBase()):
    super(TestingTranspiler, self).__init__(name="autograph", operator_overload=operator_overload)
    if isinstance(converters, (list, tuple)):
      self._converters = converters
    else:
      self._converters = (converters,)
    self.transformed_ast = None
    self._ag_overrides = ag_overrides

  def get_extra_locals(self):
    retval = super(TestingTranspiler, self).get_extra_locals()
    if self._ag_overrides:
      modified_ag = imp.new_module('fake_autograph')
      modified_ag.__dict__.update(retval['ag__'].__dict__)
      modified_ag.__dict__.update(self._ag_overrides)
      retval['ag__'] = modified_ag
    return retval

  def transform_ast(self, node, ctx):
    node = self.initial_analysis(node, ctx)

    for c in self._converters:
      node = c.transform(node, ctx)

    self.transformed_ast = node
    self.transform_ctx = ctx
    return node


class TestCase(unittest.TestCase):
  """Base class for unit tests in this module. Contains relevant utilities."""

  @contextlib.contextmanager
  def assertPrints(self, expected_result):
    try:
      out_capturer = six.StringIO()
      sys.stdout = out_capturer
      yield
      self.assertEqual(out_capturer.getvalue(), expected_result)
    finally:
      sys.stdout = sys.__stdout__

  def transform(self, f, converter_module, include_ast=False, ag_overrides=None,
                operator_overload=hooks.OperatorBase()):
    program_ctx = converter.ProgramContext(
        options=converter.ConversionOptions(recursive=True),
        autograph_module=api)

    tr = TestingTranspiler(converter_module, ag_overrides, operator_overload=operator_overload)
    transformed, _, _ = tr.transform_function(f, program_ctx)

    if include_ast:
      return transformed, tr.transformed_ast, tr.transform_ctx

    return transformed
