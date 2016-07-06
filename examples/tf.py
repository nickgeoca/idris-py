#!/usr/bin/env python

import sys
import importlib
import math

Unit = object()
World = object()

class IdrisError(Exception):
  pass

def _idris_error(msg):
  raise IdrisError(msg)

def _idris_pymodule(name):
  return importlib.import_module(name)

def _idris_call(f, args):
  return f(*list(args))

def _idris_foreach(it, st, f):
  for x in it:
    # Apply st, x, world
    st = APPLY0(APPLY0(APPLY0(f, st), x), World)
  return st

def _idris_try(f, fail, succ):
  try:
    result = APPLY0(f, World)  # apply to world
    return APPLY0(succ, result)
  except Exception as e:
    return APPLY0(fail, e)

def _idris_raise(e):
  raise e

def _idris_marshal_PIO(action):
  return lambda: APPLY0(action, World)  # delayed apply-to-world

def _idris_get_global(name):
  return globals()[name]

class _ConsIter(object):
  def __init__(self, node):
    self.node = node

  def next(self):
    if self.node.isNil:
      raise StopIteration
    else:
      result = self.node.head
      self.node = self.node.tail
      return result

class ConsList(object):
  def __init__(self, isNil=True, head=None, tail=None):
    self.isNil = isNil
    self.head  = head
    self.tail  = tail

  def __nonzero__(self):
    return not self.isNil

  def __len__(self):
    cnt = 0
    while not self.isNil:
      self = self.tail
      cnt += 1
    return cnt

  def cons(self, x):
    return ConsList(isNil=False, head=x, tail=self)

  def __iter__(self):
    return _ConsIter(self)

# Python.Functions.$.
def _idris_Python_46_Functions_46__36__46_(e0, e1, e2, e3, e4, e5):
  while True:
    return _idris_Prelude_46_Functor_46_Prelude_46_Monad_46__64_Prelude_46_Functor_46_Functor_36_IO_39__32_ffi_58__33_map_58_0(
      None,
      None,
      None,
      (65702, None),  # {U_Python.IO.unRaw1}
      (65701, e3, e2, e5)  # {U_Python.Functions.{$.0}1}
    )

# Prelude.Bool.&&
def _idris_Prelude_46_Bool_46__38__38_(e0, e1):
  while True:
    if not e0:  # Prelude.Bool.False
      return False
    else:  # Prelude.Bool.True
      return EVAL0(e1)
    return _idris_error("unreachable due to case in tail position")

# Prelude.List.++
def _idris_Prelude_46_List_46__43__43_(e0, e1, e2):
  while True:
    if e1:  # Prelude.List.::
      in0, in1 = e1.head, e1.tail
      return _idris_Prelude_46_List_46__43__43_(None, in1, e2).cons(in0)
    else:  # Prelude.List.Nil
      return e2
    return _idris_error("unreachable due to case in tail position")

# Prelude.Basics..
def _idris_Prelude_46_Basics_46__46_(e0, e1, e2, e3, e4, _idris_x):
  while True:
    return APPLY0(e3, APPLY0(e4, _idris_x))

# Python.Fields./.
def _idris_Python_46_Fields_46__47__46_(e0, e1, e2, e3, e4):
  while True:
    return _idris_unsafePerformIO(None, None, (65700, e2, e3))  # {U_Python.Fields.{/.0}1}

# Force
def _idris_Force(e0, e1, e2):
  while True:
    in0 = EVAL0(e2)
    return in0

# PE_show_49dea51
def _idris_PE_95_show_95_49dea51(e0):
  while True:
    return _idris_Prelude_46_Show_46_Prelude_46_Show_46__64_Prelude_46_Show_46_Show_36_String_58__33_show_58_0(
      e0
    )

# believe_me
def _idris_believe_95_me(e0, e1, e2):
  while True:
    return e2

# call__IO
def _idris_call_95__95_IO(e0, e1, e2):
  while True:
    return APPLY0(e2, None)

# Prelude.List.drop
def _idris_Prelude_46_List_46_drop(e0, e1, e2):
  while True:
    if e1 == 0:
      return e2
    else:
      if e2:  # Prelude.List.::
        in0, in1 = e2.head, e2.tail
        e0, e1, e2, = None, (e1 - 1), in1,
        continue
        return _idris_error("unreachable due to tail call")
      else:  # Prelude.List.Nil
        return ConsList()
      return _idris_error("unreachable due to case in tail position")
    return _idris_error("unreachable due to case in tail position")

# Prelude.Maybe.fromMaybe
def _idris_Prelude_46_Maybe_46_fromMaybe(e0, e1, e2):
  while True:
    if e2 is not None:  # Prelude.Maybe.Just
      in0 = e2
      return in0
    else:  # Prelude.Maybe.Nothing
      return EVAL0(e1)
    return _idris_error("unreachable due to case in tail position")

# Python.getGlobal
def _idris_Python_46_getGlobal(e0, e1):
  while True:
    return _idris_unsafePerformIO(None, None, (65728, e1))  # {U_Python.{getGlobal0}1}

# Prelude.Basics.id
def _idris_Prelude_46_Basics_46_id(e0, e1):
  while True:
    return e1

# Prelude.Bool.ifThenElse
def _idris_Prelude_46_Bool_46_ifThenElse(e0, e1, e2, e3):
  while True:
    if not e1:  # Prelude.Bool.False
      return EVAL0(e3)
    else:  # Prelude.Bool.True
      return EVAL0(e2)
    return _idris_error("unreachable due to case in tail position")

# Python.importModule
def _idris_Python_46_importModule(e0, e1, _idris_w):
  while True:
    return _idris_pymodule(e1)

# Prelude.List.init'
def _idris_Prelude_46_List_46_init_39_(e0, e1):
  while True:
    if e1:  # Prelude.List.::
      in0, in1 = e1.head, e1.tail
      if in1:  # Prelude.List.::
        in2, in3 = in1.head, in1.tail
        aux1 = _idris_Prelude_46_List_46_init_39_(None, in3.cons(in2))
        if aux1 is not None:  # Prelude.Maybe.Just
          in4 = aux1
          return in4.cons(in0)
        else:  # Prelude.Maybe.Nothing
          return None
        return _idris_error("unreachable due to case in tail position")
      else:  # Prelude.List.Nil
        return ConsList()
      return _idris_error("unreachable due to case in tail position")
    else:  # Prelude.List.Nil
      return None
    return _idris_error("unreachable due to case in tail position")

# Prelude.Interfaces.intToBool
def _idris_Prelude_46_Interfaces_46_intToBool(e0):
  while True:
    if e0 == 0:
      return False
    else:
      return True
    return _idris_error("unreachable due to case in tail position")

# io_bind
def _idris_io_95_bind(e0, e1, e2, e3, e4, _idris_w):
  while True:
    return APPLY0(io_bind2(e0, e1, e2, e3, e4, _idris_w), APPLY0(e3, _idris_w))

# io_return
def _idris_io_95_return(e0, e1, e2, _idris_w):
  while True:
    return e2

# Prelude.Chars.isDigit
def _idris_Prelude_46_Chars_46_isDigit(e0):
  while True:
    aux1 = _idris_Prelude_46_Interfaces_46_Prelude_46_Interfaces_46__64_Prelude_46_Interfaces_46_Ord_36_Char_58__33__62__61__58_0(
      e0, u'0'
    )
    if not aux1:  # Prelude.Bool.False
      return False
    else:  # Prelude.Bool.True
      return _idris_Prelude_46_Chars_46__123_isDigit0_125_(e0)
    return _idris_error("unreachable due to case in tail position")

# Prelude.List.length
def _idris_Prelude_46_List_46_length(e0, e1):
  while True:
    if e1:  # Prelude.List.::
      in0, in1 = e1.head, e1.tail
      return (1 + _idris_Prelude_46_List_46_length(None, in1))
    else:  # Prelude.List.Nil
      return 0
    return _idris_error("unreachable due to case in tail position")

# Main.main
def _idris_Main_46_main():
  while True:
    return (
      65729,  # {U_io_bind1}
      None,
      None,
      None,
      (65729, None, None, None, (65668,), (65671,)),  # {U_io_bind1}, {U_Main.{main0}1}, {U_Main.{main1}1}
      (65670,)  # {U_Main.{main11}1}
    )

# Prelude.List.mergeBy
def _idris_Prelude_46_List_46_mergeBy(e0, e1, e2, e3):
  while True:
    if not e2:  # Prelude.List.Nil
      return e3
    else:
      if e3:  # Prelude.List.::
        in0, in1 = e3.head, e3.tail
        assert e2  # Prelude.List.::
        in2, in3 = e2.head, e2.tail
        aux1 = APPLY0(APPLY0(e1, in2), in0)
        if aux1[0] == 0:  # Prelude.Interfaces.LT
          return _idris_Prelude_46_List_46_mergeBy(None, e1, in3, in1.cons(in0)).cons(in2)
        else:
          return _idris_Prelude_46_List_46_mergeBy(None, e1, in3.cons(in2), in1).cons(in0)
        return _idris_error("unreachable due to case in tail position")
        return _idris_error("unreachable due to case in tail position")
      else:  # Prelude.List.Nil
        return e2
      return _idris_error("unreachable due to case in tail position")
    return _idris_error("unreachable due to case in tail position")

# Prelude.Nat.minus
def _idris_Prelude_46_Nat_46_minus(e0, e1):
  while True:
    if e0 == 0:
      return 0
    else:
      if e1 == 0:
        return e0
      else:
        in0 = (e1 - 1)
        e0, e1, = (e0 - 1), in0,
        continue
        return _idris_error("unreachable due to tail call")
      return _idris_error("unreachable due to case in tail position")
    return _idris_error("unreachable due to case in tail position")

# mkForeignPrim
def _idris_mkForeignPrim():
  while True:
    return None

# Prelude.natEnumFromTo
def _idris_Prelude_46_natEnumFromTo(e0, e1):
  while True:
    return _idris_Prelude_46_Functor_46_Prelude_46_List_46__64_Prelude_46_Functor_46_Functor_36_List_58__33_map_58_0(
      None,
      None,
      (65731, e0),  # {U_prim__addBigInt1}
      _idris_Prelude_46_List_46_reverse_58_reverse_39__58_0(
        None,
        ConsList(),
        _idris_Prelude_46_natRange_58_go_58_0(
          None,
          _idris_Prelude_46_Nat_46_minus((e1 + 1), e0)
        )
      )
    )

# Prelude.Bool.not
def _idris_Prelude_46_Bool_46_not(e0):
  while True:
    if not e0:  # Prelude.Bool.False
      return True
    else:  # Prelude.Bool.True
      return False
    return _idris_error("unreachable due to case in tail position")

# Python.Lib.TensorFlow.Matrix.ones
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_ones(e0, e1):
  while True:
    return (
      0,  # Python.Lib.TensorFlow.Matrix.MkT
      e1,
      _idris_unsafePerformIO(
        None,
        None,
        _idris_Python_46_Functions_46__36__46_(
          None,
          None,
          (1, (0,), (65704,)),  # Python.Telescope.Bind, Python.Telescope.Pi, {U_Python.Lib.TensorFlow.Matrix.{ones0}1}
          _idris_Python_46_Fields_46__47__46_(
            None,
            None,
            _idris_unsafePerformIO(None, None, (65727, None, u'tensorflow')),  # {U_Python.importModule1}
            u'ones',
            None
          ),
          None,
          (0, _idris_Python_46_Prim_46_pyList(None, e1), Unit)  # Builtins.MkDPair
        )
      )
    )

# Prelude.Show.precCon
def _idris_Prelude_46_Show_46_precCon(e0):
  while True:
    if e0[0] == 6:  # Prelude.Show.App
      return 6
    elif e0[0] == 3:  # Prelude.Show.Backtick
      return 3
    elif e0[0] == 2:  # Prelude.Show.Dollar
      return 2
    elif e0[0] == 1:  # Prelude.Show.Eq
      return 1
    elif e0[0] == 0:  # Prelude.Show.Open
      return 0
    elif e0[0] == 5:  # Prelude.Show.PrefixMinus
      return 5
    else:  # Prelude.Show.User
      in0 = e0[1]
      return 4
    return _idris_error("unreachable due to case in tail position")

# Prelude.Show.primNumShow
def _idris_Prelude_46_Show_46_primNumShow(e0, e1, e2, e3):
  while True:
    in0 = APPLY0(e1, e3)
    aux2 = _idris_Prelude_46_Interfaces_46_Prelude_46_Show_46__64_Prelude_46_Interfaces_46_Ord_36_Prec_58__33__62__61__58_0(
      e2, (5,)  # Prelude.Show.PrefixMinus
    )
    if not aux2:  # Prelude.Bool.False
      aux3 = False
    else:  # Prelude.Bool.True
      aux3 = _idris_Prelude_46_Show_46__123_primNumShow2_125_(in0, e0, e1, e2, e3)
    aux1 = aux3
    if not aux1:  # Prelude.Bool.False
      return in0
    else:  # Prelude.Bool.True
      return (u'(' + (in0 + u')'))
    return _idris_error("unreachable due to case in tail position")

# prim__addBigInt
def _idris_prim_95__95_addBigInt(op0, op1):
  while True:
    return (op0 + op1)

# prim__asPtr
def _idris_prim_95__95_asPtr(op0):
  while True:
    return _idris_error("unimplemented external: prim__asPtr")

# prim__charToInt
def _idris_prim_95__95_charToInt(op0):
  while True:
    return ord(op0)

# prim__concat
def _idris_prim_95__95_concat(op0, op1):
  while True:
    return (op0 + op1)

# prim__eqBigInt
def _idris_prim_95__95_eqBigInt(op0, op1):
  while True:
    return (op0 == op1)

# prim__eqChar
def _idris_prim_95__95_eqChar(op0, op1):
  while True:
    return (op0 == op1)

# prim__eqManagedPtr
def _idris_prim_95__95_eqManagedPtr(op0, op1):
  while True:
    return _idris_error("unimplemented external: prim__eqManagedPtr")

# prim__eqPtr
def _idris_prim_95__95_eqPtr(op0, op1):
  while True:
    return _idris_error("unimplemented external: prim__eqPtr")

# prim__eqString
def _idris_prim_95__95_eqString(op0, op1):
  while True:
    return (op0 == op1)

# prim__null
def _idris_prim_95__95_null():
  while True:
    return None

# prim__peek16
def _idris_prim_95__95_peek16(op0, op1, op2):
  while True:
    return _idris_error("unimplemented external: prim__peek16")

# prim__peek32
def _idris_prim_95__95_peek32(op0, op1, op2):
  while True:
    return _idris_error("unimplemented external: prim__peek32")

# prim__peek64
def _idris_prim_95__95_peek64(op0, op1, op2):
  while True:
    return _idris_error("unimplemented external: prim__peek64")

# prim__peek8
def _idris_prim_95__95_peek8(op0, op1, op2):
  while True:
    return _idris_error("unimplemented external: prim__peek8")

# prim__peekDouble
def _idris_prim_95__95_peekDouble(op0, op1, op2):
  while True:
    return _idris_error("unimplemented external: prim__peekDouble")

# prim__peekPtr
def _idris_prim_95__95_peekPtr(op0, op1, op2):
  while True:
    return _idris_error("unimplemented external: prim__peekPtr")

# prim__peekSingle
def _idris_prim_95__95_peekSingle(op0, op1, op2):
  while True:
    return _idris_error("unimplemented external: prim__peekSingle")

# prim__poke16
def _idris_prim_95__95_poke16(op0, op1, op2, op3):
  while True:
    return _idris_error("unimplemented external: prim__poke16")

# prim__poke32
def _idris_prim_95__95_poke32(op0, op1, op2, op3):
  while True:
    return _idris_error("unimplemented external: prim__poke32")

# prim__poke64
def _idris_prim_95__95_poke64(op0, op1, op2, op3):
  while True:
    return _idris_error("unimplemented external: prim__poke64")

# prim__poke8
def _idris_prim_95__95_poke8(op0, op1, op2, op3):
  while True:
    return _idris_error("unimplemented external: prim__poke8")

# prim__pokeDouble
def _idris_prim_95__95_pokeDouble(op0, op1, op2, op3):
  while True:
    return _idris_error("unimplemented external: prim__pokeDouble")

# prim__pokePtr
def _idris_prim_95__95_pokePtr(op0, op1, op2, op3):
  while True:
    return _idris_error("unimplemented external: prim__pokePtr")

# prim__pokeSingle
def _idris_prim_95__95_pokeSingle(op0, op1, op2, op3):
  while True:
    return _idris_error("unimplemented external: prim__pokeSingle")

# prim__ptrOffset
def _idris_prim_95__95_ptrOffset(op0, op1):
  while True:
    return _idris_error("unimplemented external: prim__ptrOffset")

# prim__readFile
def _idris_prim_95__95_readFile(op0, op1):
  while True:
    return _idris_error("unimplemented external: prim__readFile")

# prim__registerPtr
def _idris_prim_95__95_registerPtr(op0, op1):
  while True:
    return _idris_error("unimplemented external: prim__registerPtr")

# prim__sextInt_BigInt
def _idris_prim_95__95_sextInt_95_BigInt(op0):
  while True:
    return op0

# prim__sizeofPtr
def _idris_prim_95__95_sizeofPtr():
  while True:
    return _idris_error("unimplemented external: prim__sizeofPtr")

# prim__sltBigInt
def _idris_prim_95__95_sltBigInt(op0, op1):
  while True:
    return (op0 < op1)

# prim__sltChar
def _idris_prim_95__95_sltChar(op0, op1):
  while True:
    return (op0 < op1)

# prim__stderr
def _idris_prim_95__95_stderr():
  while True:
    return _idris_error("unimplemented external: prim__stderr")

# prim__stdin
def _idris_prim_95__95_stdin():
  while True:
    return _idris_error("unimplemented external: prim__stdin")

# prim__stdout
def _idris_prim_95__95_stdout():
  while True:
    return _idris_error("unimplemented external: prim__stdout")

# prim__strCons
def _idris_prim_95__95_strCons(op0, op1):
  while True:
    return (op0 + op1)

# prim__strHead
def _idris_prim_95__95_strHead(op0):
  while True:
    return op0[0]

# prim__strTail
def _idris_prim_95__95_strTail(op0):
  while True:
    return op0[1:]

# prim__toStrInt
def _idris_prim_95__95_toStrInt(op0):
  while True:
    return str(op0)

# prim__vm
def _idris_prim_95__95_vm():
  while True:
    return _idris_error("unimplemented external: prim__vm")

# prim__writeFile
def _idris_prim_95__95_writeFile(op0, op1, op2):
  while True:
    return _idris_error("unimplemented external: prim__writeFile")

# prim__writeString
def _idris_prim_95__95_writeString(op0, op1):
  while True:
    return sys.stdout.write(op1)

# prim_io_bind
def _idris_prim_95_io_95_bind(e0, e1, e2, e3):
  while True:
    return APPLY0(e3, e2)

# prim_write
def _idris_prim_95_write(e0, e1, _idris_w):
  while True:
    return sys.stdout.write(e1)

# Prelude.Show.protectEsc
def _idris_Prelude_46_Show_46_protectEsc(e0, e1, e2):
  while True:
    aux2 = _idris_Prelude_46_Strings_46_strM(e2)
    if aux2[0] == 1:  # Prelude.Strings.StrCons
      in0, in1 = aux2[1:]
      aux3 = APPLY0(e0, in0)
    else:  # Prelude.Strings.StrNil
      aux3 = False
    aux1 = aux3
    if not aux1:  # Prelude.Bool.False
      aux4 = u''
    else:  # Prelude.Bool.True
      aux4 = u'\\&'
    return (e1 + (aux4 + e2))

# Prelude.Interactive.putStr'
def _idris_Prelude_46_Interactive_46_putStr_39_(e0, e1):
  while True:
    return (65729, None, None, None, (65734, None, e1), (65684,))  # {U_io_bind1}, {U_prim_write1}, {U_Prelude.Interactive.{putStr'0}1}

# Python.Prim.pyList
def _idris_Python_46_Prim_46_pyList(e0, e1):
  while True:
    return _idris_unsafePerformIO(
      None,
      None,
      _idris_Python_46_Functions_46__36__46_(
        None,
        None,
        (1, (1,), (65726,)),  # Python.Telescope.Bind, Python.Telescope.Forall, {U_Python.Prim.{pyList1}1}
        _idris_Python_46_Fields_46__47__46_(
          None,
          None,
          _idris_Python_46_getGlobal(None, u'__builtins__'),
          u'list',
          None
        ),
        None,
        (0, (0,), (0, e1, Unit))  # Builtins.MkDPair, Data.Erased.Erase, Builtins.MkDPair
      )
    )

# really_believe_me
def _idris_really_95_believe_95_me(e0, e1, e2):
  while True:
    return e2

# Python.Lib.TensorFlow.Matrix.reduce_mean
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_mean(
  e0, e1, e2, e3, e4
):
  while True:
    assert e2[0] == 0  # Python.Lib.TensorFlow.Matrix.MkT
    in0, in1 = e2[1:]
    if not e4:  # Prelude.Bool.False
      aux1 = _idris_Prelude_46_Foldable_46_Prelude_46_List_46__64_Prelude_46_Foldable_46_Foldable_36_List_58__33_foldl_58_0(
        None,
        None,
        (65739, None, None, None, None, None, None),  # {U_Python.Lib.TensorFlow.Matrix.reduce_reshape, removeFn, removeElem2}
        in0,
        _idris_Prelude_46_List_46_reverse_58_reverse_39__58_0(
          None,
          ConsList(),
          _idris_Prelude_46_List_46_sortBy(None, (65709,), e3)  # {U_Python.Lib.TensorFlow.Matrix.{reduce_mean1}1}
        )
      )
    else:  # Prelude.Bool.True
      aux1 = in0
    return (
      0,  # Python.Lib.TensorFlow.Matrix.MkT
      aux1,
      _idris_unsafePerformIO(
        None,
        None,
        _idris_Python_46_Functions_46__36__46_(
          None,
          None,
          (1, (0,), (65712,)),  # Python.Telescope.Bind, Python.Telescope.Pi, {U_Python.Lib.TensorFlow.Matrix.{reduce_mean4}1}
          _idris_Python_46_Fields_46__47__46_(
            None,
            None,
            _idris_unsafePerformIO(None, None, (65727, None, u'tensorflow')),  # {U_Python.importModule1}
            u'reduce_mean',
            None
          ),
          None,
          (0, in1, (0, _idris_Python_46_Prim_46_pyList(None, e3), (0, e4, Unit)))  # Builtins.MkDPair, Builtins.MkDPair, Builtins.MkDPair
        )
      )
    )
    return _idris_error("unreachable due to case in tail position")

# Python.Lib.TensorFlow.Matrix.reduce_mean'
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_mean_39_(
  e0, e1, e2, e3
):
  while True:
    assert e2[0] == 0  # Python.Lib.TensorFlow.Matrix.MkT
    in0, in1 = e2[1:]
    if not e3:  # Prelude.Bool.False
      aux1 = ConsList()
    else:  # Prelude.Bool.True
      aux1 = in0
    return (
      0,  # Python.Lib.TensorFlow.Matrix.MkT
      aux1,
      _idris_unsafePerformIO(
        None,
        None,
        _idris_Python_46_Functions_46__36__46_(
          None,
          None,
          (1, (0,), (65707,)),  # Python.Telescope.Bind, Python.Telescope.Pi, {U_Python.Lib.TensorFlow.Matrix.{reduce_mean'2}1}
          _idris_Python_46_Fields_46__47__46_(
            None,
            None,
            _idris_unsafePerformIO(None, None, (65727, None, u'tensorflow')),  # {U_Python.importModule1}
            u'reduce_mean',
            None
          ),
          None,
          (
            0,  # Builtins.MkDPair
            in1,
            (
              0,  # Builtins.MkDPair
              _idris_Python_46_Prim_46_pyList(
                None,
                _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_remove_95_all_95_dims(in0)
              ),
              (0, e3, Unit)  # Builtins.MkDPair
            )
          )
        )
      )
    )
    return _idris_error("unreachable due to case in tail position")

# Python.Lib.TensorFlow.Matrix.reduce_sum
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_sum(
  e0, e1, e2, e3, e4
):
  while True:
    assert e2[0] == 0  # Python.Lib.TensorFlow.Matrix.MkT
    in0, in1 = e2[1:]
    if not e4:  # Prelude.Bool.False
      aux1 = _idris_Prelude_46_Foldable_46_Prelude_46_List_46__64_Prelude_46_Foldable_46_Foldable_36_List_58__33_foldl_58_0(
        None,
        None,
        (65739, None, None, None, None, None, None),  # {U_Python.Lib.TensorFlow.Matrix.reduce_reshape, removeFn, removeElem2}
        in0,
        _idris_Prelude_46_List_46_reverse_58_reverse_39__58_0(
          None,
          ConsList(),
          _idris_Prelude_46_List_46_sortBy(None, (65717,), e3)  # {U_Python.Lib.TensorFlow.Matrix.{reduce_sum1}1}
        )
      )
    else:  # Prelude.Bool.True
      aux1 = in0
    return (
      0,  # Python.Lib.TensorFlow.Matrix.MkT
      aux1,
      _idris_unsafePerformIO(
        None,
        None,
        _idris_Python_46_Functions_46__36__46_(
          None,
          None,
          (1, (0,), (65720,)),  # Python.Telescope.Bind, Python.Telescope.Pi, {U_Python.Lib.TensorFlow.Matrix.{reduce_sum4}1}
          _idris_Python_46_Fields_46__47__46_(
            None,
            None,
            _idris_unsafePerformIO(None, None, (65727, None, u'tensorflow')),  # {U_Python.importModule1}
            u'reduce_sum',
            None
          ),
          None,
          (0, in1, (0, _idris_Python_46_Prim_46_pyList(None, e3), (0, e4, Unit)))  # Builtins.MkDPair, Builtins.MkDPair, Builtins.MkDPair
        )
      )
    )
    return _idris_error("unreachable due to case in tail position")

# Python.Lib.TensorFlow.Matrix.reduce_sum'
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_sum_39_(
  e0, e1, e2, e3
):
  while True:
    assert e2[0] == 0  # Python.Lib.TensorFlow.Matrix.MkT
    in0, in1 = e2[1:]
    if not e3:  # Prelude.Bool.False
      aux1 = ConsList()
    else:  # Prelude.Bool.True
      aux1 = in0
    return (
      0,  # Python.Lib.TensorFlow.Matrix.MkT
      aux1,
      _idris_unsafePerformIO(
        None,
        None,
        _idris_Python_46_Functions_46__36__46_(
          None,
          None,
          (1, (0,), (65715,)),  # Python.Telescope.Bind, Python.Telescope.Pi, {U_Python.Lib.TensorFlow.Matrix.{reduce_sum'2}1}
          _idris_Python_46_Fields_46__47__46_(
            None,
            None,
            _idris_unsafePerformIO(None, None, (65727, None, u'tensorflow')),  # {U_Python.importModule1}
            u'reduce_sum',
            None
          ),
          None,
          (
            0,  # Builtins.MkDPair
            in1,
            (
              0,  # Builtins.MkDPair
              _idris_Python_46_Prim_46_pyList(
                None,
                _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_remove_95_all_95_dims(in0)
              ),
              (0, e3, Unit)  # Builtins.MkDPair
            )
          )
        )
      )
    )
    return _idris_error("unreachable due to case in tail position")

# Python.Lib.TensorFlow.Matrix.remove_all_dims
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_remove_95_all_95_dims(e0):
  while True:
    aux1 = _idris_Prelude_46_List_46_init_39_(
      None,
      _idris_Prelude_46_natEnumFromTo(0, _idris_Prelude_46_List_46_length(None, e0))
    )
    if aux1 is not None:  # Prelude.Maybe.Just
      in0 = aux1
      return in0
    else:  # Prelude.Maybe.Nothing
      return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_remove_95_all_95_dims0_125_()
    return _idris_error("unreachable due to case in tail position")

# Python.Lib.TensorFlow.Matrix.run
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_run(e0, e1, e2, e3):
  while True:
    assert e3[0] == 0  # Python.Lib.TensorFlow.Matrix.MkT
    in0, in1 = e3[1:]
    return (65729, None, None, None, e2, (65723, in1))  # {U_io_bind1}, {U_Python.Lib.TensorFlow.Matrix.{run2}1}
    return _idris_error("unreachable due to case in tail position")

# run__IO
def _idris_run_95__95_IO(e0, e1):
  while True:
    return APPLY0(e1, None)

# Python.Lib.TensorFlow.Matrix.session
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_session():
  while True:
    return (
      65729,  # {U_io_bind1}
      None,
      None,
      None,
      _idris_Python_46_Functions_46__36__46_(
        None,
        None,
        (0,),  # Python.Telescope.Return
        _idris_Python_46_Fields_46__47__46_(
          None,
          None,
          _idris_unsafePerformIO(None, None, (65727, None, u'tensorflow')),  # {U_Python.importModule1}
          u'Session',
          None
        ),
        None,
        Unit
      ),
      (65724,)  # {U_Python.Lib.TensorFlow.Matrix.{session0}1}
    )

# Python.Lib.TensorFlow.Matrix.Session'.session'
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_Session_39__46_session_39_(e0):
  while True:
    return e0

# Prelude.Show.showLitChar
def _idris_Prelude_46_Show_46_showLitChar(e0):
  while True:
    aux1 = _idris_Prelude_46_Show_46_showLitChar_58_getAt_58_10(
      None,
      ord(e0),
      _idris_Prelude_46_Show_46_showLitChar_58_asciiTab_58_10(None)
    )
    if aux1 is not None:  # Prelude.Maybe.Just
      in10 = aux1
      aux2 = (65680, None, None, None, (65732, u'\\'), (65688, in10))  # {U_Prelude.Basics..1}, {U_prim__strCons1}, {U_Prelude.Show.{showLitChar10}1}
    else:  # Prelude.Maybe.Nothing
      aux4 = _idris_Prelude_46_Interfaces_46_Prelude_46_Interfaces_46__64_Prelude_46_Interfaces_46_Ord_36_Char_58__33_compare_58_0(
        e0,
        u'\u007f'
      )
      if aux4[0] == 2:  # Prelude.Interfaces.GT
        aux5 = True
      else:
        aux5 = False
      aux3 = aux5
      if not aux3:  # Prelude.Bool.False
        aux6 = (65732, e0)  # {U_prim__strCons1}
      else:  # Prelude.Bool.True
        aux6 = (
          65680,  # {U_Prelude.Basics..1}
          None,
          None,
          None,
          (65732, u'\\'),  # {U_prim__strCons1}
          (
            65685,  # {U_Prelude.Show.protectEsc1}
            (65682,),  # {U_Prelude.Chars.isDigit1}
            _idris_Prelude_46_Show_46_primNumShow(None, (65733,), (0,), ord(e0))  # {U_prim__toStrInt1}, Prelude.Show.Open
          )
        )
      aux2 = aux6
    return {
      u'\u0007': (65687,),  # {U_Prelude.Show.{showLitChar0}1}
      u'\u0008': (65689,),  # {U_Prelude.Show.{showLitChar1}1}
      u'\u0009': (65690,),  # {U_Prelude.Show.{showLitChar2}1}
      u'\u000a': (65691,),  # {U_Prelude.Show.{showLitChar3}1}
      u'\u000b': (65692,),  # {U_Prelude.Show.{showLitChar4}1}
      u'\u000c': (65693,),  # {U_Prelude.Show.{showLitChar5}1}
      u'\u000d': (65694,),  # {U_Prelude.Show.{showLitChar6}1}
      u'\u000e': (65685, (65695,), u'\\SO'),  # {U_Prelude.Show.protectEsc1}, {U_Prelude.Show.{showLitChar7}1}
      u'\\': (65696,),  # {U_Prelude.Show.{showLitChar8}1}
      u'\u007f': (65697,)  # {U_Prelude.Show.{showLitChar9}1}
    }.get(e0, aux2)

# Prelude.Show.showLitString
def _idris_Prelude_46_Show_46_showLitString(e0):
  while True:
    if e0:  # Prelude.List.::
      in0, in1 = e0.head, e0.tail
      if in0 == u'"':
        return (
          65680,  # {U_Prelude.Basics..1}
          None,
          None,
          None,
          (65699,),  # {U_Prelude.Show.{showLitString0}1}
          _idris_Prelude_46_Show_46_showLitString(in1)
        )
      else:
        return (
          65680,  # {U_Prelude.Basics..1}
          None,
          None,
          None,
          _idris_Prelude_46_Show_46_showLitChar(in0),
          _idris_Prelude_46_Show_46_showLitString(in1)
        )
      return _idris_error("unreachable due to case in tail position")
    else:  # Prelude.List.Nil
      return (65681, None)  # {U_Prelude.Basics.id1}
    return _idris_error("unreachable due to case in tail position")

# Prelude.Show.showParens
def _idris_Prelude_46_Show_46_showParens(e0, e1):
  while True:
    if not e0:  # Prelude.Bool.False
      return e1
    else:  # Prelude.Bool.True
      return (u'(' + (e1 + u')'))
    return _idris_error("unreachable due to case in tail position")

# Prelude.List.sortBy
def _idris_Prelude_46_List_46_sortBy(e0, e1, e2):
  while True:
    if e2:  # Prelude.List.::
      in0, in1 = e2.head, e2.tail
      if not in1:  # Prelude.List.Nil
        return ConsList().cons(in0)
      else:
        aux1 = _idris_Prelude_46_List_46_sortBy_58_splitRec_58_2(
          None,
          None,
          None,
          e2,
          e2,
          (65681, None)  # {U_Prelude.Basics.id1}
        )
        assert True  # Builtins.MkPair
        in2, in3 = aux1
        return _idris_Prelude_46_List_46_mergeBy(
          None,
          e1,
          _idris_Prelude_46_List_46_sortBy(None, e1, in2),
          _idris_Prelude_46_List_46_sortBy(None, e1, in3)
        )
        return _idris_error("unreachable due to case in tail position")
      return _idris_error("unreachable due to case in tail position")
    elif not e2:  # Prelude.List.Nil
      return ConsList()
    else:
      aux2 = _idris_Prelude_46_List_46_sortBy_58_splitRec_58_2(
        None,
        None,
        None,
        e2,
        e2,
        (65681, None)  # {U_Prelude.Basics.id1}
      )
      assert True  # Builtins.MkPair
      in4, in5 = aux2
      return _idris_Prelude_46_List_46_mergeBy(
        None,
        e1,
        _idris_Prelude_46_List_46_sortBy(None, e1, in4),
        _idris_Prelude_46_List_46_sortBy(None, e1, in5)
      )
      return _idris_error("unreachable due to case in tail position")
    return _idris_error("unreachable due to case in tail position")

# Prelude.List.splitAt
def _idris_Prelude_46_List_46_splitAt(e0, e1, e2):
  while True:
    return (_idris_Prelude_46_List_46_take(None, e1, e2), _idris_Prelude_46_List_46_drop(None, e1, e2))

# Prelude.Strings.strM
def _idris_Prelude_46_Strings_46_strM(e0):
  while True:
    aux3 = (e0 == u'')
    if aux3 == 0:
      aux4 = False
    else:
      aux4 = True
    aux2 = aux4
    if not aux2:  # Prelude.Bool.False
      aux5 = True
    else:  # Prelude.Bool.True
      aux5 = False
    aux1 = _idris_Decidable_46_Equality_46_Decidable_46_Equality_46__64_Decidable_46_Equality_46_DecEq_36_Bool_58__33_decEq_58_0(
      aux5, True
    )
    if aux1[0] == 1:  # Prelude.Basics.No
      return _idris_really_95_believe_95_me(None, None, (0,))  # Prelude.Strings.StrNil
    else:  # Prelude.Basics.Yes
      return _idris_really_95_believe_95_me(None, None, (1, e0[0], e0[1:]))  # Prelude.Strings.StrCons
    return _idris_error("unreachable due to case in tail position")

# Python.Functions.strip
def _idris_Python_46_Functions_46_strip(e0, e1, e2):
  while True:
    if e1[0] == 1:  # Python.Telescope.Bind
      in0, in1 = e1[1:]
      if in0[0] == 2:  # Python.Telescope.Default
        in2 = in0[1]
        assert e2[0] == 0  # Builtins.MkDPair
        in3, in4 = e2[1:]
        if in3 is not None:  # Prelude.Maybe.Just
          in5 = in3
          aux1 = in5
        else:  # Prelude.Maybe.Nothing
          aux1 = _idris_Python_46_Functions_46__123_strip0_125_(in2)
        return _idris_Python_46_Functions_46_strip(None, APPLY0(in1, aux1), in4).cons(_idris_believe_95_me(None, None, in3))
        return _idris_error("unreachable due to case in tail position")
      elif in0[0] == 1:  # Python.Telescope.Forall
        assert e2[0] == 0  # Builtins.MkDPair
        in6, in7 = e2[1:]
        e0, e1, e2, = None, APPLY0(in1, in6), in7,
        continue
        return _idris_error("unreachable due to tail call")
        return _idris_error("unreachable due to case in tail position")
      else:  # Python.Telescope.Pi
        assert e2[0] == 0  # Builtins.MkDPair
        in8, in9 = e2[1:]
        return _idris_Python_46_Functions_46_strip(None, APPLY0(in1, in8), in9).cons(_idris_believe_95_me(None, None, in8))
        return _idris_error("unreachable due to case in tail position")
      return _idris_error("unreachable due to case in tail position")
    else:  # Python.Telescope.Return
      return ConsList()
    return _idris_error("unreachable due to case in tail position")

# Prelude.List.tail'
def _idris_Prelude_46_List_46_tail_39_(e0, e1):
  while True:
    if e1:  # Prelude.List.::
      in0, in1 = e1.head, e1.tail
      return in1
    else:  # Prelude.List.Nil
      return None
    return _idris_error("unreachable due to case in tail position")

# Prelude.List.take
def _idris_Prelude_46_List_46_take(e0, e1, e2):
  while True:
    if e1 == 0:
      return ConsList()
    else:
      if e2:  # Prelude.List.::
        in0, in1 = e2.head, e2.tail
        return _idris_Prelude_46_List_46_take(None, (e1 - 1), in1).cons(in0)
      else:  # Prelude.List.Nil
        return ConsList()
      return _idris_error("unreachable due to case in tail position")
    return _idris_error("unreachable due to case in tail position")

# Python.IO.unRaw
def _idris_Python_46_IO_46_unRaw(e0, e1):
  while True:
    return e1

# Prelude.Strings.unpack
def _idris_Prelude_46_Strings_46_unpack(e0):
  while True:
    aux1 = _idris_Prelude_46_Strings_46_strM(e0)
    if aux1[0] == 1:  # Prelude.Strings.StrCons
      in0, in1 = aux1[1:]
      return _idris_Prelude_46_Strings_46_unpack(in1).cons(in0)
    else:  # Prelude.Strings.StrNil
      return ConsList()
    return _idris_error("unreachable due to case in tail position")

# unsafePerformIO
def _idris_unsafePerformIO(e0, e1, e2):
  while True:
    return APPLY0(unsafePerformIO1(e0, e1, e2), APPLY0(e2, None))

# unsafePerformPrimIO
def _idris_unsafePerformPrimIO():
  while True:
    return None

# world
def _idris_world(e0):
  while True:
    return e0

# Main.x
def _idris_Main_46_x():
  while True:
    return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_ones(
      None,
      ConsList().cons(4).cons(3)
    )

# Prelude.Bool.||
def _idris_Prelude_46_Bool_46__124__124_(e0, e1):
  while True:
    if not e0:  # Prelude.Bool.False
      return EVAL0(e1)
    else:  # Prelude.Bool.True
      return True
    return _idris_error("unreachable due to case in tail position")

# Python.Functions.{$.0}
def _idris_Python_46_Functions_46__123__36__46_0_125_(e3, e2, e5, in0):
  while True:
    return _idris_call(e3, _idris_Python_46_Functions_46_strip(None, e2, e5))

# Python.Fields.{/.0}
def _idris_Python_46_Fields_46__123__47__46_0_125_(e2, e3, in0):
  while True:
    return getattr(e2, e3)

# {APPLY0}
def APPLY0(fn0, arg0):
  while True:
    if fn0[0] < 65704:
      if fn0[0] < 65686:
        if fn0[0] < 65677:
          if fn0[0] < 65672:
            if fn0[0] < 65670:
              if fn0[0] == 65668:  # {U_Main.{main0}1}
                return _idris_Main_46__123_main0_125_(arg0)
              else:  # {U_Main.{main10}1}
                return _idris_Main_46__123_main10_125_(arg0)
            else:
              if fn0[0] == 65670:  # {U_Main.{main11}1}
                return _idris_Main_46__123_main11_125_(arg0)
              else:  # {U_Main.{main1}1}
                return _idris_Main_46__123_main1_125_(arg0)
          else:
            if fn0[0] < 65674:
              if fn0[0] == 65672:  # {U_Main.{main2}1}
                return _idris_Main_46__123_main2_125_(arg0)
              else:  # {U_Main.{main3}1}
                return _idris_Main_46__123_main3_125_(arg0)
            else:
              if fn0[0] == 65674:  # {U_Main.{main4}1}
                return _idris_Main_46__123_main4_125_(arg0)
              elif fn0[0] == 65675:  # {U_Main.{main5}1}
                return _idris_Main_46__123_main5_125_(arg0)
              else:  # {U_Main.{main6}1}
                return _idris_Main_46__123_main6_125_(arg0)
        else:
          if fn0[0] < 65681:
            if fn0[0] < 65679:
              if fn0[0] == 65677:  # {U_Main.{main7}1}
                return _idris_Main_46__123_main7_125_(arg0)
              else:  # {U_Main.{main8}1}
                return _idris_Main_46__123_main8_125_(arg0)
            else:
              if fn0[0] == 65679:  # {U_Main.{main9}1}
                return _idris_Main_46__123_main9_125_(arg0)
              else:  # {U_Prelude.Basics..1}
                P_c0, P_c1, P_c2, P_c3, P_c4 = fn0[1:]
                return _idris_Prelude_46_Basics_46__46_(P_c0, P_c1, P_c2, P_c3, P_c4, arg0)
          else:
            if fn0[0] < 65683:
              if fn0[0] == 65681:  # {U_Prelude.Basics.id1}
                P_c0 = fn0[1]
                return _idris_Prelude_46_Basics_46_id(P_c0, arg0)
              else:  # {U_Prelude.Chars.isDigit1}
                return _idris_Prelude_46_Chars_46_isDigit(arg0)
            else:
              if fn0[0] == 65683:  # {U_Prelude.Functor.{Prelude.Monad.@Prelude.Functor.Functor$IO' ffi:!map:0_lam0}1}
                P_c0 = fn0[1]
                return _idris_Prelude_46_Functor_46__123_Prelude_46_Monad_46__64_Prelude_46_Functor_46_Functor_36_IO_39__32_ffi_58__33_map_58_0_95_lam0_125_(
                  P_c0, arg0
                )
              elif fn0[0] == 65684:  # {U_Prelude.Interactive.{putStr'0}1}
                return _idris_Prelude_46_Interactive_46__123_putStr_39_0_125_(arg0)
              else:  # {U_Prelude.Show.protectEsc1}
                P_c0, P_c1 = fn0[1:]
                return _idris_Prelude_46_Show_46_protectEsc(P_c0, P_c1, arg0)
      else:
        if fn0[0] < 65695:
          if fn0[0] < 65690:
            if fn0[0] < 65688:
              if fn0[0] == 65686:  # {U_Prelude.Show.{primNumShow0}1}
                return _idris_Prelude_46_Show_46__123_primNumShow0_125_(arg0)
              else:  # {U_Prelude.Show.{showLitChar0}1}
                return _idris_Prelude_46_Show_46__123_showLitChar0_125_(arg0)
            else:
              if fn0[0] == 65688:  # {U_Prelude.Show.{showLitChar10}1}
                P_c0 = fn0[1]
                return _idris_Prelude_46_Show_46__123_showLitChar10_125_(P_c0, arg0)
              else:  # {U_Prelude.Show.{showLitChar1}1}
                return _idris_Prelude_46_Show_46__123_showLitChar1_125_(arg0)
          else:
            if fn0[0] < 65692:
              if fn0[0] == 65690:  # {U_Prelude.Show.{showLitChar2}1}
                return _idris_Prelude_46_Show_46__123_showLitChar2_125_(arg0)
              else:  # {U_Prelude.Show.{showLitChar3}1}
                return _idris_Prelude_46_Show_46__123_showLitChar3_125_(arg0)
            else:
              if fn0[0] == 65692:  # {U_Prelude.Show.{showLitChar4}1}
                return _idris_Prelude_46_Show_46__123_showLitChar4_125_(arg0)
              elif fn0[0] == 65693:  # {U_Prelude.Show.{showLitChar5}1}
                return _idris_Prelude_46_Show_46__123_showLitChar5_125_(arg0)
              else:  # {U_Prelude.Show.{showLitChar6}1}
                return _idris_Prelude_46_Show_46__123_showLitChar6_125_(arg0)
        else:
          if fn0[0] < 65699:
            if fn0[0] < 65697:
              if fn0[0] == 65695:  # {U_Prelude.Show.{showLitChar7}1}
                return _idris_Prelude_46_Show_46__123_showLitChar7_125_(arg0)
              else:  # {U_Prelude.Show.{showLitChar8}1}
                return _idris_Prelude_46_Show_46__123_showLitChar8_125_(arg0)
            else:
              if fn0[0] == 65697:  # {U_Prelude.Show.{showLitChar9}1}
                return _idris_Prelude_46_Show_46__123_showLitChar9_125_(arg0)
              else:  # {U_Prelude.Show.{showLitChar_____Prelude__Show__idr_128_27_case_lam0}1}
                P_c0 = fn0[1]
                return _idris_Prelude_46_Show_46__123_showLitChar_95__95__95__95__95_Prelude_95__95_Show_95__95_idr_95_128_95_27_95_case_95_lam0_125_(
                  P_c0, arg0
                )
          else:
            if fn0[0] < 65701:
              if fn0[0] == 65699:  # {U_Prelude.Show.{showLitString0}1}
                return _idris_Prelude_46_Show_46__123_showLitString0_125_(arg0)
              else:  # {U_Python.Fields.{/.0}1}
                P_c0, P_c1 = fn0[1:]
                return _idris_Python_46_Fields_46__123__47__46_0_125_(P_c0, P_c1, arg0)
            else:
              if fn0[0] == 65701:  # {U_Python.Functions.{$.0}1}
                P_c0, P_c1, P_c2 = fn0[1:]
                return _idris_Python_46_Functions_46__123__36__46_0_125_(P_c0, P_c1, P_c2, arg0)
              elif fn0[0] == 65702:  # {U_Python.IO.unRaw1}
                P_c0 = fn0[1]
                return _idris_Python_46_IO_46_unRaw(P_c0, arg0)
              else:  # {U_Python.Lib.TensorFlow.Matrix.reduce_reshape, removeFn, removeElem1}
                P_c0, P_c1, P_c2, P_c3, P_c4, P_c5, P_c6 = fn0[1:]
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_reshape_58_removeFn_58_0_58_removeElem_58_0(
                  P_c0, P_c1, P_c2, P_c3, P_c4, P_c5, P_c6, arg0
                )
    else:
      if fn0[0] < 65722:
        if fn0[0] < 65713:
          if fn0[0] < 65708:
            if fn0[0] < 65706:
              if fn0[0] == 65704:  # {U_Python.Lib.TensorFlow.Matrix.{ones0}1}
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_ones0_125_(arg0)
              else:  # {U_Python.Lib.TensorFlow.Matrix.{reduce_mean'0}1}
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_mean_39_0_125_(
                  arg0
                )
            else:
              if fn0[0] == 65706:  # {U_Python.Lib.TensorFlow.Matrix.{reduce_mean'1}1}
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_mean_39_1_125_(
                  arg0
                )
              else:  # {U_Python.Lib.TensorFlow.Matrix.{reduce_mean'2}1}
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_mean_39_2_125_(
                  arg0
                )
          else:
            if fn0[0] < 65710:
              if fn0[0] == 65708:  # {U_Python.Lib.TensorFlow.Matrix.{reduce_mean0}1}
                P_c0 = fn0[1]
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_mean0_125_(
                  P_c0, arg0
                )
              else:  # {U_Python.Lib.TensorFlow.Matrix.{reduce_mean1}1}
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_mean1_125_(arg0)
            else:
              if fn0[0] == 65710:  # {U_Python.Lib.TensorFlow.Matrix.{reduce_mean2}1}
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_mean2_125_(arg0)
              elif fn0[0] == 65711:  # {U_Python.Lib.TensorFlow.Matrix.{reduce_mean3}1}
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_mean3_125_(arg0)
              else:  # {U_Python.Lib.TensorFlow.Matrix.{reduce_mean4}1}
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_mean4_125_(arg0)
        else:
          if fn0[0] < 65717:
            if fn0[0] < 65715:
              if fn0[0] == 65713:  # {U_Python.Lib.TensorFlow.Matrix.{reduce_sum'0}1}
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_sum_39_0_125_(
                  arg0
                )
              else:  # {U_Python.Lib.TensorFlow.Matrix.{reduce_sum'1}1}
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_sum_39_1_125_(
                  arg0
                )
            else:
              if fn0[0] == 65715:  # {U_Python.Lib.TensorFlow.Matrix.{reduce_sum'2}1}
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_sum_39_2_125_(
                  arg0
                )
              else:  # {U_Python.Lib.TensorFlow.Matrix.{reduce_sum0}1}
                P_c0 = fn0[1]
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_sum0_125_(
                  P_c0, arg0
                )
          else:
            if fn0[0] < 65719:
              if fn0[0] == 65717:  # {U_Python.Lib.TensorFlow.Matrix.{reduce_sum1}1}
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_sum1_125_(arg0)
              else:  # {U_Python.Lib.TensorFlow.Matrix.{reduce_sum2}1}
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_sum2_125_(arg0)
            else:
              if fn0[0] == 65719:  # {U_Python.Lib.TensorFlow.Matrix.{reduce_sum3}1}
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_sum3_125_(arg0)
              elif fn0[0] == 65720:  # {U_Python.Lib.TensorFlow.Matrix.{reduce_sum4}1}
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_sum4_125_(arg0)
              else:  # {U_Python.Lib.TensorFlow.Matrix.{run0}1}
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_run0_125_(arg0)
      else:
        if fn0[0] < 65731:
          if fn0[0] < 65726:
            if fn0[0] < 65724:
              if fn0[0] == 65722:  # {U_Python.Lib.TensorFlow.Matrix.{run1}1}
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_run1_125_(arg0)
              else:  # {U_Python.Lib.TensorFlow.Matrix.{run2}1}
                P_c0 = fn0[1]
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_run2_125_(P_c0, arg0)
            else:
              if fn0[0] == 65724:  # {U_Python.Lib.TensorFlow.Matrix.{session0}1}
                return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_session0_125_(arg0)
              else:  # {U_Python.Prim.{pyList0}1}
                return _idris_Python_46_Prim_46__123_pyList0_125_(arg0)
          else:
            if fn0[0] < 65728:
              if fn0[0] == 65726:  # {U_Python.Prim.{pyList1}1}
                return _idris_Python_46_Prim_46__123_pyList1_125_(arg0)
              else:  # {U_Python.importModule1}
                P_c0, P_c1 = fn0[1:]
                return _idris_Python_46_importModule(P_c0, P_c1, arg0)
            else:
              if fn0[0] == 65728:  # {U_Python.{getGlobal0}1}
                P_c0 = fn0[1]
                return _idris_Python_46__123_getGlobal0_125_(P_c0, arg0)
              elif fn0[0] == 65729:  # {U_io_bind1}
                P_c0, P_c1, P_c2, P_c3, P_c4 = fn0[1:]
                return _idris_io_95_bind(P_c0, P_c1, P_c2, P_c3, P_c4, arg0)
              else:  # {U_io_return1}
                P_c0, P_c1, P_c2 = fn0[1:]
                return _idris_io_95_return(P_c0, P_c1, P_c2, arg0)
        else:
          if fn0[0] < 65736:
            if fn0[0] < 65733:
              if fn0[0] == 65731:  # {U_prim__addBigInt1}
                P_c0 = fn0[1]
                return _idris_prim_95__95_addBigInt(P_c0, arg0)
              else:  # {U_prim__strCons1}
                P_c0 = fn0[1]
                return _idris_prim_95__95_strCons(P_c0, arg0)
            else:
              if fn0[0] == 65733:  # {U_prim__toStrInt1}
                return _idris_prim_95__95_toStrInt(arg0)
              elif fn0[0] == 65734:  # {U_prim_write1}
                P_c0, P_c1 = fn0[1:]
                return _idris_prim_95_write(P_c0, P_c1, arg0)
              else:  # {U_{Main.main:printOp:0_lam0}1}
                return _idris__123_Main_46_main_58_printOp_58_0_95_lam0_125_(arg0)
          else:
            if fn0[0] < 65738:
              if fn0[0] == 65736:  # {U_{Prelude.List.sortBy:splitRec:2_lam0}1}
                P_c0 = fn0[1]
                return _idris__123_Prelude_46_List_46_sortBy_58_splitRec_58_2_95_lam0_125_(P_c0, arg0)
              else:  # {U_{io_bind1}1}
                P_c0, P_c1, P_c2, P_c3, P_c4, P_c5 = fn0[1:]
                return io_bind1(P_c0, P_c1, P_c2, P_c3, P_c4, P_c5, arg0)
            else:
              if fn0[0] == 65738:  # {U_{unsafePerformIO0}1}
                return unsafePerformIO0(arg0)
              else:  # {U_Python.Lib.TensorFlow.Matrix.reduce_reshape, removeFn, removeElem2}
                P_c0, P_c1, P_c2, P_c3, P_c4, P_c5 = fn0[1:]
                return (65703, P_c0, P_c1, P_c2, P_c3, P_c4, P_c5, arg0)  # {U_Python.Lib.TensorFlow.Matrix.reduce_reshape, removeFn, removeElem1}
    return _idris_error("unreachable due to case in tail position")

# {APPLY20}
def _idris__123_APPLY20_125_(fn0, _idris__123_arg00_125_, _idris__123_arg10_125_):
  while True:
    if fn0[0] == 65739:  # {U_Python.Lib.TensorFlow.Matrix.reduce_reshape, removeFn, removeElem2}
      P_c0, P_c1, P_c2, P_c3, P_c4, P_c5 = fn0[1:]
      return _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_reshape_58_removeFn_58_0_58_removeElem_58_0(
        P_c0,
        P_c1,
        P_c2,
        P_c3,
        P_c4,
        P_c5,
        _idris__123_arg00_125_,
        _idris__123_arg10_125_
      )
    else:
      return APPLY0(APPLY0(fn0, _idris__123_arg00_125_), _idris__123_arg10_125_)
    return _idris_error("unreachable due to case in tail position")

# {EVAL0}
def EVAL0(arg0):
  while True:
    return arg0

# {Main.main:printOp:0_lam0}
def _idris__123_Main_46_main_58_printOp_58_0_95_lam0_125_(in0):
  while True:
    return _idris_Prelude_46_Interactive_46_putStr_39_(
      None,
      (_idris_Prelude_46_Show_46_Python_46_Lib_46_Numpy_46_Matrix_46__64_Prelude_46_Show_46_Show_36_MatrixN_32_xs_32_dt_58__33_show_58_0(
        None, None, None, in0
      ) + u'\u000a')
    )

# Prelude.Interfaces.{Prelude.Interfaces.@Prelude.Interfaces.Ord$Char:!<=:0_lam0}
def _idris_Prelude_46_Interfaces_46__123_Prelude_46_Interfaces_46__64_Prelude_46_Interfaces_46_Ord_36_Char_58__33__60__61__58_0_95_lam0_125_(
  e0, e1
):
  while True:
    aux1 = (e0 == e1)
    if aux1 == 0:
      return False
    else:
      return True
    return _idris_error("unreachable due to case in tail position")

# Prelude.Interfaces.{Prelude.Interfaces.@Prelude.Interfaces.Ord$Char:!>=:0_lam0}
def _idris_Prelude_46_Interfaces_46__123_Prelude_46_Interfaces_46__64_Prelude_46_Interfaces_46_Ord_36_Char_58__33__62__61__58_0_95_lam0_125_(
  e0, e1
):
  while True:
    aux1 = (e0 == e1)
    if aux1 == 0:
      return False
    else:
      return True
    return _idris_error("unreachable due to case in tail position")

# {Prelude.List.sortBy:splitRec:2_lam0}
def _idris__123_Prelude_46_List_46_sortBy_58_splitRec_58_2_95_lam0_125_(in0, in6):
  while True:
    return in6.cons(in0)

# Prelude.Functor.{Prelude.Monad.@Prelude.Functor.Functor$IO' ffi:!map:0_lam0}
def _idris_Prelude_46_Functor_46__123_Prelude_46_Monad_46__64_Prelude_46_Functor_46_Functor_36_IO_39__32_ffi_58__33_map_58_0_95_lam0_125_(
  e3, in0
):
  while True:
    return (65730, None, None, APPLY0(e3, in0))  # {U_io_return1}

# Prelude.Interfaces.{Prelude.Show.@Prelude.Interfaces.Ord$Prec:!>=:0_lam0}
def _idris_Prelude_46_Interfaces_46__123_Prelude_46_Show_46__64_Prelude_46_Interfaces_46_Ord_36_Prec_58__33__62__61__58_0_95_lam0_125_(
  e0, e1
):
  while True:
    return _idris_Prelude_46_Interfaces_46_Prelude_46_Show_46__64_Prelude_46_Interfaces_46_Eq_36_Prec_58__33__61__61__58_0(
      e0, e1
    )

# Python.Lib.TensorFlow.Matrix.{Python.Lib.TensorFlow.Matrix.reduce_reshape:removeFn:0:removeElem:0_____Python__Lib__TensorFlow__Matrix__idr_183_32_case_lam0}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_reshape_58_removeFn_58_0_58_removeElem_58_0_95__95__95__95__95_Python_95__95_Lib_95__95_TensorFlow_95__95_Matrix_95__95_idr_95_183_95_32_95_case_95_lam0_125_():
  while True:
    return ConsList()

# {Python.Lib.TensorFlow.Matrix.reduce_reshape:removeFn:0:removeElem:0_lam0}
def _idris__123_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_reshape_58_removeFn_58_0_58_removeElem_58_0_95_lam0_125_():
  while True:
    return ConsList()

# Python.{getGlobal0}
def _idris_Python_46__123_getGlobal0_125_(e1, in0):
  while True:
    return _idris_get_global(e1)

# {io_bind0}
def io_bind0(e0, e1, e2, e3, e4, _idris_w, in0):
  while True:
    return APPLY0(e4, in0)

# Prelude.Chars.{isDigit0}
def _idris_Prelude_46_Chars_46__123_isDigit0_125_(e0):
  while True:
    return _idris_Prelude_46_Interfaces_46_Prelude_46_Interfaces_46__64_Prelude_46_Interfaces_46_Ord_36_Char_58__33__60__61__58_0(
      e0, u'9'
    )

# Main.{main0}
def _idris_Main_46__123_main0_125_(in0):
  while True:
    return sys.stdout.write((_idris_PE_95_show_95_49dea51(u'hey') + u'\u000a'))

# Python.Lib.TensorFlow.Matrix.{ones0}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_ones0_125_(in0):
  while True:
    return (0,)  # Python.Telescope.Return

# Prelude.Show.{primNumShow0}
def _idris_Prelude_46_Show_46__123_primNumShow0_125_(in1):
  while True:
    aux1 = (in1 == u'-')
    if aux1 == 0:
      return False
    else:
      return True
    return _idris_error("unreachable due to case in tail position")

# Prelude.Interactive.{putStr'0}
def _idris_Prelude_46_Interactive_46__123_putStr_39_0_125_(in0):
  while True:
    return (65730, None, None, Unit)  # {U_io_return1}

# Python.Prim.{pyList0}
def _idris_Python_46_Prim_46__123_pyList0_125_(in1):
  while True:
    return (0,)  # Python.Telescope.Return

# Python.Lib.TensorFlow.Matrix.{reduce_mean0}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_mean0_125_(
  in2, in3
):
  while True:
    return _idris_Prelude_46_Interfaces_46_Prelude_46_Nat_46__64_Prelude_46_Interfaces_46_Ord_36_Nat_58__33_compare_58_0(
      in2, in3
    )

# Python.Lib.TensorFlow.Matrix.{reduce_mean'0}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_mean_39_0_125_(
  in4
):
  while True:
    return (0,)  # Python.Telescope.Return

# Python.Lib.TensorFlow.Matrix.{reduce_sum0}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_sum0_125_(
  in2, in3
):
  while True:
    return _idris_Prelude_46_Interfaces_46_Prelude_46_Nat_46__64_Prelude_46_Interfaces_46_Ord_36_Nat_58__33_compare_58_0(
      in2, in3
    )

# Python.Lib.TensorFlow.Matrix.{reduce_sum'0}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_sum_39_0_125_(
  in4
):
  while True:
    return (0,)  # Python.Telescope.Return

# Python.Lib.TensorFlow.Matrix.{remove_all_dims0}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_remove_95_all_95_dims0_125_():
  while True:
    return ConsList()

# Python.Lib.TensorFlow.Matrix.{run0}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_run0_125_(in3):
  while True:
    return (0,)  # Python.Telescope.Return

# {runMain0}
def runMain0():
  while True:
    return EVAL0(APPLY0(_idris_Main_46_main(), None))

# Python.Lib.TensorFlow.Matrix.{session0}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_session0_125_(in0):
  while True:
    return (65730, None, None, in0)  # {U_io_return1}

# Prelude.Show.{showLitChar0}
def _idris_Prelude_46_Show_46__123_showLitChar0_125_(in0):
  while True:
    return (u'\\a' + in0)

# Prelude.Show.{showLitChar_____Prelude__Show__idr_128_27_case_lam0}
def _idris_Prelude_46_Show_46__123_showLitChar_95__95__95__95__95_Prelude_95__95_Show_95__95_idr_95_128_95_27_95_case_95_lam0_125_(
  in0, in1
):
  while True:
    return (in0 + in1)

# Prelude.Show.{showLitString0}
def _idris_Prelude_46_Show_46__123_showLitString0_125_(in2):
  while True:
    return (u'\\"' + in2)

# Python.Functions.{strip0}
def _idris_Python_46_Functions_46__123_strip0_125_(in2):
  while True:
    return in2

# {unsafePerformIO0}
def unsafePerformIO0(in0):
  while True:
    return in0

# {io_bind1}
def io_bind1(e0, e1, e2, e3, e4, _idris_w, in0):
  while True:
    return APPLY0(io_bind0(e0, e1, e2, e3, e4, _idris_w, in0), _idris_w)

# Main.{main1}
def _idris_Main_46__123_main1_125_(in1):
  while True:
    return (65730, None, None, Unit)  # {U_io_return1}

# Prelude.Show.{primNumShow1}
def _idris_Prelude_46_Show_46__123_primNumShow1_125_(e0, e1, e2, e3, in0, in2, in3):
  while True:
    return (65686,)  # {U_Prelude.Show.{primNumShow0}1}

# Python.Prim.{pyList1}
def _idris_Python_46_Prim_46__123_pyList1_125_(in0):
  while True:
    return (1, (0,), (65725,))  # Python.Telescope.Bind, Python.Telescope.Pi, {U_Python.Prim.{pyList0}1}

# Python.Lib.TensorFlow.Matrix.{reduce_mean1}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_mean1_125_(in2):
  while True:
    return (65708, in2)  # {U_Python.Lib.TensorFlow.Matrix.{reduce_mean0}1}

# Python.Lib.TensorFlow.Matrix.{reduce_mean'1}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_mean_39_1_125_(
  in3
):
  while True:
    return (1, (0,), (65705,))  # Python.Telescope.Bind, Python.Telescope.Pi, {U_Python.Lib.TensorFlow.Matrix.{reduce_mean'0}1}

# Python.Lib.TensorFlow.Matrix.{reduce_sum1}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_sum1_125_(in2):
  while True:
    return (65716, in2)  # {U_Python.Lib.TensorFlow.Matrix.{reduce_sum0}1}

# Python.Lib.TensorFlow.Matrix.{reduce_sum'1}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_sum_39_1_125_(
  in3
):
  while True:
    return (1, (0,), (65713,))  # Python.Telescope.Bind, Python.Telescope.Pi, {U_Python.Lib.TensorFlow.Matrix.{reduce_sum'0}1}

# Python.Lib.TensorFlow.Matrix.{run1}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_run1_125_(in4):
  while True:
    return (65730, None, None, in4)  # {U_io_return1}

# Prelude.Show.{showLitChar1}
def _idris_Prelude_46_Show_46__123_showLitChar1_125_(in1):
  while True:
    return (u'\\b' + in1)

# {unsafePerformIO1}
def unsafePerformIO1(e0, e1, e2):
  while True:
    return (65738,)  # {U_{unsafePerformIO0}1}

# {io_bind2}
def io_bind2(e0, e1, e2, e3, e4, _idris_w):
  while True:
    return (65737, e0, e1, e2, e3, e4, _idris_w)  # {U_{io_bind1}1}

# Main.{main2}
def _idris_Main_46__123_main2_125_(in11):
  while True:
    return _idris_Main_46_main_58_printOp_58_0(
      None,
      None,
      _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_sum_39_(
        None,
        None,
        _idris_Main_46_x(),
        True
      )
    )

# Prelude.Show.{primNumShow2}
def _idris_Prelude_46_Show_46__123_primNumShow2_125_(in0, e0, e1, e2, e3):
  while True:
    aux1 = _idris_Prelude_46_Strings_46_strM(in0)
    if aux1[0] == 1:  # Prelude.Strings.StrCons
      in2, in3 = aux1[1:]
      return APPLY0(
        _idris_Prelude_46_Show_46__123_primNumShow1_125_(e0, e1, e2, e3, in0, in2, in3),
        in2
      )
    else:  # Prelude.Strings.StrNil
      return False
    return _idris_error("unreachable due to case in tail position")

# Python.Lib.TensorFlow.Matrix.{reduce_mean2}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_mean2_125_(in6):
  while True:
    return (0,)  # Python.Telescope.Return

# Python.Lib.TensorFlow.Matrix.{reduce_mean'2}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_mean_39_2_125_(
  in2
):
  while True:
    return (1, (0,), (65706,))  # Python.Telescope.Bind, Python.Telescope.Pi, {U_Python.Lib.TensorFlow.Matrix.{reduce_mean'1}1}

# Python.Lib.TensorFlow.Matrix.{reduce_sum2}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_sum2_125_(in6):
  while True:
    return (0,)  # Python.Telescope.Return

# Python.Lib.TensorFlow.Matrix.{reduce_sum'2}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_sum_39_2_125_(
  in2
):
  while True:
    return (1, (0,), (65714,))  # Python.Telescope.Bind, Python.Telescope.Pi, {U_Python.Lib.TensorFlow.Matrix.{reduce_sum'1}1}

# Python.Lib.TensorFlow.Matrix.{run2}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_run2_125_(in1, in2):
  while True:
    return (
      65729,  # {U_io_bind1}
      None,
      None,
      None,
      _idris_Python_46_Functions_46__36__46_(
        None,
        None,
        (1, (0,), (65721,)),  # Python.Telescope.Bind, Python.Telescope.Pi, {U_Python.Lib.TensorFlow.Matrix.{run0}1}
        _idris_Python_46_Fields_46__47__46_(None, None, in2, u'run', None),
        None,
        (0, in1, Unit)  # Builtins.MkDPair
      ),
      (65722,)  # {U_Python.Lib.TensorFlow.Matrix.{run1}1}
    )

# Prelude.Show.{showLitChar2}
def _idris_Prelude_46_Show_46__123_showLitChar2_125_(in2):
  while True:
    return (u'\\t' + in2)

# Main.{main3}
def _idris_Main_46__123_main3_125_(in10):
  while True:
    return (
      65729,  # {U_io_bind1}
      None,
      None,
      None,
      _idris_Main_46_main_58_printOp_58_0(
        None,
        None,
        _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_sum_39_(
          None,
          None,
          _idris_Main_46_x(),
          False
        )
      ),
      (65672,)  # {U_Main.{main2}1}
    )

# Python.Lib.TensorFlow.Matrix.{reduce_mean3}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_mean3_125_(in5):
  while True:
    return (1, (0,), (65710,))  # Python.Telescope.Bind, Python.Telescope.Pi, {U_Python.Lib.TensorFlow.Matrix.{reduce_mean2}1}

# Python.Lib.TensorFlow.Matrix.{reduce_sum3}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_sum3_125_(in5):
  while True:
    return (1, (0,), (65718,))  # Python.Telescope.Bind, Python.Telescope.Pi, {U_Python.Lib.TensorFlow.Matrix.{reduce_sum2}1}

# Prelude.Show.{showLitChar3}
def _idris_Prelude_46_Show_46__123_showLitChar3_125_(in3):
  while True:
    return (u'\\n' + in3)

# Main.{main4}
def _idris_Main_46__123_main4_125_(in9):
  while True:
    return (
      65729,  # {U_io_bind1}
      None,
      None,
      None,
      _idris_Main_46_main_58_printOp_58_0(
        None,
        None,
        _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_sum(
          None,
          None,
          _idris_Main_46_x(),
          ConsList().cons(0),
          False
        )
      ),
      (65673,)  # {U_Main.{main3}1}
    )

# Python.Lib.TensorFlow.Matrix.{reduce_mean4}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_mean4_125_(in4):
  while True:
    return (1, (0,), (65711,))  # Python.Telescope.Bind, Python.Telescope.Pi, {U_Python.Lib.TensorFlow.Matrix.{reduce_mean3}1}

# Python.Lib.TensorFlow.Matrix.{reduce_sum4}
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_reduce_95_sum4_125_(in4):
  while True:
    return (1, (0,), (65719,))  # Python.Telescope.Bind, Python.Telescope.Pi, {U_Python.Lib.TensorFlow.Matrix.{reduce_sum3}1}

# Prelude.Show.{showLitChar4}
def _idris_Prelude_46_Show_46__123_showLitChar4_125_(in4):
  while True:
    return (u'\\v' + in4)

# Main.{main5}
def _idris_Main_46__123_main5_125_(in8):
  while True:
    return (
      65729,  # {U_io_bind1}
      None,
      None,
      None,
      _idris_Main_46_main_58_printOp_58_0(
        None,
        None,
        _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_sum(
          None,
          None,
          _idris_Main_46_x(),
          ConsList().cons(1).cons(0),
          True
        )
      ),
      (65674,)  # {U_Main.{main4}1}
    )

# Prelude.Show.{showLitChar5}
def _idris_Prelude_46_Show_46__123_showLitChar5_125_(in5):
  while True:
    return (u'\\f' + in5)

# Main.{main6}
def _idris_Main_46__123_main6_125_(in7):
  while True:
    return (
      65729,  # {U_io_bind1}
      None,
      None,
      None,
      _idris_Main_46_main_58_printOp_58_0(
        None,
        None,
        _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_mean_39_(
          None,
          None,
          _idris_Main_46_x(),
          False
        )
      ),
      (65675,)  # {U_Main.{main5}1}
    )

# Prelude.Show.{showLitChar6}
def _idris_Prelude_46_Show_46__123_showLitChar6_125_(in6):
  while True:
    return (u'\\r' + in6)

# Main.{main7}
def _idris_Main_46__123_main7_125_(in6):
  while True:
    return (
      65729,  # {U_io_bind1}
      None,
      None,
      None,
      _idris_Main_46_main_58_printOp_58_0(
        None,
        None,
        _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_mean_39_(
          None,
          None,
          _idris_Main_46_x(),
          True
        )
      ),
      (65676,)  # {U_Main.{main6}1}
    )

# Prelude.Show.{showLitChar7}
def _idris_Prelude_46_Show_46__123_showLitChar7_125_(in7):
  while True:
    aux1 = (in7 == u'H')
    if aux1 == 0:
      return False
    else:
      return True
    return _idris_error("unreachable due to case in tail position")

# Main.{main8}
def _idris_Main_46__123_main8_125_(in5):
  while True:
    return (
      65729,  # {U_io_bind1}
      None,
      None,
      None,
      _idris_Main_46_main_58_printOp_58_0(
        None,
        None,
        _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_mean(
          None,
          None,
          _idris_Main_46_x(),
          ConsList().cons(0),
          False
        )
      ),
      (65677,)  # {U_Main.{main7}1}
    )

# Prelude.Show.{showLitChar8}
def _idris_Prelude_46_Show_46__123_showLitChar8_125_(in8):
  while True:
    return (u'\\\\' + in8)

# Main.{main9}
def _idris_Main_46__123_main9_125_(in4):
  while True:
    return (
      65729,  # {U_io_bind1}
      None,
      None,
      None,
      _idris_Main_46_main_58_printOp_58_0(
        None,
        None,
        _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_mean(
          None,
          None,
          _idris_Main_46_x(),
          ConsList().cons(1).cons(0),
          True
        )
      ),
      (65678,)  # {U_Main.{main8}1}
    )

# Prelude.Show.{showLitChar9}
def _idris_Prelude_46_Show_46__123_showLitChar9_125_(in9):
  while True:
    return (u'\\DEL' + in9)

# Main.{main10}
def _idris_Main_46__123_main10_125_(in3):
  while True:
    return (
      65729,  # {U_io_bind1}
      None,
      None,
      None,
      _idris_Main_46_main_58_printOp_58_0(None, None, _idris_Main_46_x()),
      (65679,)  # {U_Main.{main9}1}
    )

# Prelude.Show.{showLitChar10}
def _idris_Prelude_46_Show_46__123_showLitChar10_125_(in10, in11):
  while True:
    return (in10 + in11)

# Main.{main11}
def _idris_Main_46__123_main11_125_(in2):
  while True:
    return (
      65729,  # {U_io_bind1}
      None,
      None,
      None,
      _idris_Prelude_46_Interactive_46_putStr_39_(
        None,
        (_idris_Prelude_46_Show_46_Python_46_Lib_46_TensorFlow_46_Matrix_46__64_Prelude_46_Show_46_Show_36_Tensor_32_xs_32_dt_58__33_show_58_0(
          None,
          None,
          _idris_Main_46_x()
        ) + u'\u000a')
      ),
      (65669,)  # {U_Main.{main10}1}
    )

# Main.exports, greet2
def _idris_Main_46_exports_58_greet2_58_0(e0):
  while True:
    return (65730, None, None, (u'Hello ' + (e0 + u'!')))  # {U_io_return1}

# Main.main, printOp
def _idris_Main_46_main_58_printOp_58_0(e0, e1, e2):
  while True:
    return (
      65729,  # {U_io_bind1}
      None,
      None,
      None,
      _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_run(
        None,
        None,
        _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_session(),
        e2
      ),
      (65735,)  # {U_{Main.main:printOp:0_lam0}1}
    )

# Prelude.natRange, go
def _idris_Prelude_46_natRange_58_go_58_0(e0, e1):
  while True:
    if e1 == 0:
      return ConsList()
    else:
      in0 = (e1 - 1)
      return _idris_Prelude_46_natRange_58_go_58_0(None, in0).cons(in0)
    return _idris_error("unreachable due to case in tail position")

# Prelude.List.reverse, reverse'
def _idris_Prelude_46_List_46_reverse_58_reverse_39__58_0(e0, e1, e2):
  while True:
    if e2:  # Prelude.List.::
      in0, in1 = e2.head, e2.tail
      e0, e1, e2, = None, e1.cons(in0), in1,
      continue
      return _idris_error("unreachable due to tail call")
    else:  # Prelude.List.Nil
      return e1
    return _idris_error("unreachable due to case in tail position")

# Decidable.Equality.Decidable.Equality.Char implementation of Decidable.Equality.DecEq, method decEq, primitiveNotEq
def _idris_Decidable_46_Equality_46_Decidable_46_Equality_46__64_Decidable_46_Equality_46_DecEq_36_Char_58__33_decEq_58_0_58_primitiveNotEq_58_0():
  while True:
    return None

# Decidable.Equality.Decidable.Equality.Int implementation of Decidable.Equality.DecEq, method decEq, primitiveNotEq
def _idris_Decidable_46_Equality_46_Decidable_46_Equality_46__64_Decidable_46_Equality_46_DecEq_36_Int_58__33_decEq_58_0_58_primitiveNotEq_58_0():
  while True:
    return None

# Decidable.Equality.Decidable.Equality.Integer implementation of Decidable.Equality.DecEq, method decEq, primitiveNotEq
def _idris_Decidable_46_Equality_46_Decidable_46_Equality_46__64_Decidable_46_Equality_46_DecEq_36_Integer_58__33_decEq_58_0_58_primitiveNotEq_58_0():
  while True:
    return None

# Decidable.Equality.Decidable.Equality.ManagedPtr implementation of Decidable.Equality.DecEq, method decEq, primitiveNotEq
def _idris_Decidable_46_Equality_46_Decidable_46_Equality_46__64_Decidable_46_Equality_46_DecEq_36_ManagedPtr_58__33_decEq_58_0_58_primitiveNotEq_58_0():
  while True:
    return None

# Decidable.Equality.Decidable.Equality.Ptr implementation of Decidable.Equality.DecEq, method decEq, primitiveNotEq
def _idris_Decidable_46_Equality_46_Decidable_46_Equality_46__64_Decidable_46_Equality_46_DecEq_36_Ptr_58__33_decEq_58_0_58_primitiveNotEq_58_0():
  while True:
    return None

# Decidable.Equality.Decidable.Equality.String implementation of Decidable.Equality.DecEq, method decEq, primitiveNotEq
def _idris_Decidable_46_Equality_46_Decidable_46_Equality_46__64_Decidable_46_Equality_46_DecEq_36_String_58__33_decEq_58_0_58_primitiveNotEq_58_0():
  while True:
    return None

# Decidable.Equality.Decidable.Equality.Bool implementation of Decidable.Equality.DecEq, method decEq
def _idris_Decidable_46_Equality_46_Decidable_46_Equality_46__64_Decidable_46_Equality_46_DecEq_36_Bool_58__33_decEq_58_0(
  e0, e1
):
  while True:
    if not e1:  # Prelude.Bool.False
      if not e0:  # Prelude.Bool.False
        return (0,)  # Prelude.Basics.Yes
      else:  # Prelude.Bool.True
        return (1,)  # Prelude.Basics.No
      return _idris_error("unreachable due to case in tail position")
    else:  # Prelude.Bool.True
      if not e0:  # Prelude.Bool.False
        return (1,)  # Prelude.Basics.No
      else:  # Prelude.Bool.True
        return (0,)  # Prelude.Basics.Yes
      return _idris_error("unreachable due to case in tail position")
    return _idris_error("unreachable due to case in tail position")

# Prelude.Interfaces.Prelude.Nat.Nat implementation of Prelude.Interfaces.Eq, method ==
def _idris_Prelude_46_Interfaces_46_Prelude_46_Nat_46__64_Prelude_46_Interfaces_46_Eq_36_Nat_58__33__61__61__58_0(
  e0, e1
):
  while True:
    if e1 == 0:
      if e0 == 0:
        return True
      else:
        return False
      return _idris_error("unreachable due to case in tail position")
    elif True:
      in0 = (e1 - 1)
      if e0 == 0:
        return False
      else:
        in1 = (e0 - 1)
        e0, e1, = in1, in0,
        continue
        return _idris_error("unreachable due to tail call")
      return _idris_error("unreachable due to case in tail position")
    else:
      return False
    return _idris_error("unreachable due to case in tail position")

# Prelude.Interfaces.Prelude.Show.Prec implementation of Prelude.Interfaces.Eq, method ==
def _idris_Prelude_46_Interfaces_46_Prelude_46_Show_46__64_Prelude_46_Interfaces_46_Eq_36_Prec_58__33__61__61__58_0(
  e0, e1
):
  while True:
    if e1[0] == 4:  # Prelude.Show.User
      in0 = e1[1]
      if e0[0] == 4:  # Prelude.Show.User
        in1 = e0[1]
        return _idris_Prelude_46_Interfaces_46_Prelude_46_Nat_46__64_Prelude_46_Interfaces_46_Eq_36_Nat_58__33__61__61__58_0(
          in1, in0
        )
      else:
        aux1 = (_idris_Prelude_46_Show_46_precCon(e0) == _idris_Prelude_46_Show_46_precCon(e1))
        if aux1 == 0:
          return False
        else:
          return True
        return _idris_error("unreachable due to case in tail position")
      return _idris_error("unreachable due to case in tail position")
    else:
      aux2 = (_idris_Prelude_46_Show_46_precCon(e0) == _idris_Prelude_46_Show_46_precCon(e1))
      if aux2 == 0:
        return False
      else:
        return True
      return _idris_error("unreachable due to case in tail position")
    return _idris_error("unreachable due to case in tail position")

# Prelude.Foldable.Prelude.List.List implementation of Prelude.Foldable.Foldable, method foldl
def _idris_Prelude_46_Foldable_46_Prelude_46_List_46__64_Prelude_46_Foldable_46_Foldable_36_List_58__33_foldl_58_0(
  e0, e1, e2, e3, e4
):
  while True:
    if e4:  # Prelude.List.::
      in0, in1 = e4.head, e4.tail
      e0, e1, e2, e3, e4, = None, None, e2, APPLY0(APPLY0(e2, e3), in0), in1,
      continue
      return _idris_error("unreachable due to tail call")
    else:  # Prelude.List.Nil
      return e3
    return _idris_error("unreachable due to case in tail position")

# Prelude.Functor.Prelude.Monad.IO' ffi implementation of Prelude.Functor.Functor, method map
def _idris_Prelude_46_Functor_46_Prelude_46_Monad_46__64_Prelude_46_Functor_46_Functor_36_IO_39__32_ffi_58__33_map_58_0(
  e0, e1, e2, e3, e4
):
  while True:
    return (65729, None, None, None, e4, (65683, e3))  # {U_io_bind1}, {U_Prelude.Functor.{Prelude.Monad.@Prelude.Functor.Functor$IO' ffi:!map:0_lam0}1}

# Prelude.Functor.Prelude.List.List implementation of Prelude.Functor.Functor, method map
def _idris_Prelude_46_Functor_46_Prelude_46_List_46__64_Prelude_46_Functor_46_Functor_36_List_58__33_map_58_0(
  e0, e1, e2, e3
):
  while True:
    if e3:  # Prelude.List.::
      in0, in1 = e3.head, e3.tail
      return _idris_Prelude_46_Functor_46_Prelude_46_List_46__64_Prelude_46_Functor_46_Functor_36_List_58__33_map_58_0(
        None, None, e2, in1
      ).cons(APPLY0(e2, in0))
    else:  # Prelude.List.Nil
      return ConsList()
    return _idris_error("unreachable due to case in tail position")

# Prelude.Interfaces.Prelude.Interfaces.Char implementation of Prelude.Interfaces.Ord, method <=
def _idris_Prelude_46_Interfaces_46_Prelude_46_Interfaces_46__64_Prelude_46_Interfaces_46_Ord_36_Char_58__33__60__61__58_0(
  e0, e1
):
  while True:
    aux2 = _idris_Prelude_46_Interfaces_46_Prelude_46_Interfaces_46__64_Prelude_46_Interfaces_46_Ord_36_Char_58__33_compare_58_0(
      e0, e1
    )
    if aux2[0] == 0:  # Prelude.Interfaces.LT
      aux3 = True
    else:
      aux3 = False
    aux1 = aux3
    if not aux1:  # Prelude.Bool.False
      return _idris_Prelude_46_Interfaces_46__123_Prelude_46_Interfaces_46__64_Prelude_46_Interfaces_46_Ord_36_Char_58__33__60__61__58_0_95_lam0_125_(
        e0, e1
      )
    else:  # Prelude.Bool.True
      return True
    return _idris_error("unreachable due to case in tail position")

# Prelude.Interfaces.Prelude.Interfaces.Char implementation of Prelude.Interfaces.Ord, method >=
def _idris_Prelude_46_Interfaces_46_Prelude_46_Interfaces_46__64_Prelude_46_Interfaces_46_Ord_36_Char_58__33__62__61__58_0(
  e0, e1
):
  while True:
    aux2 = _idris_Prelude_46_Interfaces_46_Prelude_46_Interfaces_46__64_Prelude_46_Interfaces_46_Ord_36_Char_58__33_compare_58_0(
      e0, e1
    )
    if aux2[0] == 2:  # Prelude.Interfaces.GT
      aux3 = True
    else:
      aux3 = False
    aux1 = aux3
    if not aux1:  # Prelude.Bool.False
      return _idris_Prelude_46_Interfaces_46__123_Prelude_46_Interfaces_46__64_Prelude_46_Interfaces_46_Ord_36_Char_58__33__62__61__58_0_95_lam0_125_(
        e0, e1
      )
    else:  # Prelude.Bool.True
      return True
    return _idris_error("unreachable due to case in tail position")

# Prelude.Interfaces.Prelude.Interfaces.Char implementation of Prelude.Interfaces.Ord, method compare
def _idris_Prelude_46_Interfaces_46_Prelude_46_Interfaces_46__64_Prelude_46_Interfaces_46_Ord_36_Char_58__33_compare_58_0(
  e0, e1
):
  while True:
    aux2 = (e0 == e1)
    if aux2 == 0:
      aux3 = False
    else:
      aux3 = True
    aux1 = aux3
    if not aux1:  # Prelude.Bool.False
      aux5 = (e0 < e1)
      if aux5 == 0:
        aux6 = False
      else:
        aux6 = True
      aux4 = aux6
      if not aux4:  # Prelude.Bool.False
        return (2,)  # Prelude.Interfaces.GT
      else:  # Prelude.Bool.True
        return (0,)  # Prelude.Interfaces.LT
      return _idris_error("unreachable due to case in tail position")
    else:  # Prelude.Bool.True
      return (1,)  # Prelude.Interfaces.EQ
    return _idris_error("unreachable due to case in tail position")

# Prelude.Interfaces.Prelude.Interfaces.Integer implementation of Prelude.Interfaces.Ord, method compare
def _idris_Prelude_46_Interfaces_46_Prelude_46_Interfaces_46__64_Prelude_46_Interfaces_46_Ord_36_Integer_58__33_compare_58_0(
  e0, e1
):
  while True:
    aux2 = (e0 == e1)
    if aux2 == 0:
      aux3 = False
    else:
      aux3 = True
    aux1 = aux3
    if not aux1:  # Prelude.Bool.False
      aux5 = (e0 < e1)
      if aux5 == 0:
        aux6 = False
      else:
        aux6 = True
      aux4 = aux6
      if not aux4:  # Prelude.Bool.False
        return (2,)  # Prelude.Interfaces.GT
      else:  # Prelude.Bool.True
        return (0,)  # Prelude.Interfaces.LT
      return _idris_error("unreachable due to case in tail position")
    else:  # Prelude.Bool.True
      return (1,)  # Prelude.Interfaces.EQ
    return _idris_error("unreachable due to case in tail position")

# Prelude.Interfaces.Prelude.Nat.Nat implementation of Prelude.Interfaces.Ord, method compare
def _idris_Prelude_46_Interfaces_46_Prelude_46_Nat_46__64_Prelude_46_Interfaces_46_Ord_36_Nat_58__33_compare_58_0(
  e0, e1
):
  while True:
    if e1 == 0:
      if e0 == 0:
        return (1,)  # Prelude.Interfaces.EQ
      else:
        in0 = (e0 - 1)
        return (2,)  # Prelude.Interfaces.GT
      return _idris_error("unreachable due to case in tail position")
    else:
      in1 = (e1 - 1)
      if e0 == 0:
        return (0,)  # Prelude.Interfaces.LT
      else:
        in2 = (e0 - 1)
        e0, e1, = in2, in1,
        continue
        return _idris_error("unreachable due to tail call")
      return _idris_error("unreachable due to case in tail position")
    return _idris_error("unreachable due to case in tail position")

# Prelude.Interfaces.Prelude.Show.Prec implementation of Prelude.Interfaces.Ord, method >=
def _idris_Prelude_46_Interfaces_46_Prelude_46_Show_46__64_Prelude_46_Interfaces_46_Ord_36_Prec_58__33__62__61__58_0(
  e0, e1
):
  while True:
    aux2 = _idris_Prelude_46_Interfaces_46_Prelude_46_Show_46__64_Prelude_46_Interfaces_46_Ord_36_Prec_58__33_compare_58_0(
      e0, e1
    )
    if aux2[0] == 2:  # Prelude.Interfaces.GT
      aux3 = True
    else:
      aux3 = False
    aux1 = aux3
    if not aux1:  # Prelude.Bool.False
      return _idris_Prelude_46_Interfaces_46__123_Prelude_46_Show_46__64_Prelude_46_Interfaces_46_Ord_36_Prec_58__33__62__61__58_0_95_lam0_125_(
        e0, e1
      )
    else:  # Prelude.Bool.True
      return True
    return _idris_error("unreachable due to case in tail position")

# Prelude.Interfaces.Prelude.Show.Prec implementation of Prelude.Interfaces.Ord, method compare
def _idris_Prelude_46_Interfaces_46_Prelude_46_Show_46__64_Prelude_46_Interfaces_46_Ord_36_Prec_58__33_compare_58_0(
  e0, e1
):
  while True:
    if e1[0] == 4:  # Prelude.Show.User
      in0 = e1[1]
      if e0[0] == 4:  # Prelude.Show.User
        in1 = e0[1]
        return _idris_Prelude_46_Interfaces_46_Prelude_46_Nat_46__64_Prelude_46_Interfaces_46_Ord_36_Nat_58__33_compare_58_0(
          in1, in0
        )
      else:
        return _idris_Prelude_46_Interfaces_46_Prelude_46_Interfaces_46__64_Prelude_46_Interfaces_46_Ord_36_Integer_58__33_compare_58_0(
          _idris_Prelude_46_Show_46_precCon(e0),
          _idris_Prelude_46_Show_46_precCon(e1)
        )
      return _idris_error("unreachable due to case in tail position")
    else:
      return _idris_Prelude_46_Interfaces_46_Prelude_46_Interfaces_46__64_Prelude_46_Interfaces_46_Ord_36_Integer_58__33_compare_58_0(
        _idris_Prelude_46_Show_46_precCon(e0),
        _idris_Prelude_46_Show_46_precCon(e1)
      )
    return _idris_error("unreachable due to case in tail position")

# Prelude.Show.Python.Lib.Numpy.Matrix.MatrixN xs dt implementation of Prelude.Show.Show, method show
def _idris_Prelude_46_Show_46_Python_46_Lib_46_Numpy_46_Matrix_46__64_Prelude_46_Show_46_Show_36_MatrixN_32_xs_32_dt_58__33_show_58_0(
  e0, e1, e2, e3
):
  while True:
    return _idris_unsafePerformIO(
      None,
      None,
      _idris_Python_46_Functions_46__36__46_(
        None,
        None,
        (0,),  # Python.Telescope.Return
        _idris_Python_46_Fields_46__47__46_(None, None, e3, u'__str__', None),
        None,
        Unit
      )
    )

# Prelude.Show.Prelude.Show.String implementation of Prelude.Show.Show, method show
def _idris_Prelude_46_Show_46_Prelude_46_Show_46__64_Prelude_46_Show_46_Show_36_String_58__33_show_58_0(
  e0
):
  while True:
    aux1 = _idris_Prelude_46_Strings_46_strM(e0)
    if aux1[0] == 1:  # Prelude.Strings.StrCons
      in0, in1 = aux1[1:]
      aux2 = _idris__95_Prelude_46_Strings_46_unpack_95_with_95_25(
        None,
        _idris_Prelude_46_Strings_46_strM(in1)
      ).cons(in0)
    else:  # Prelude.Strings.StrNil
      aux2 = ConsList()
    return (u'"' + APPLY0(_idris_Prelude_46_Show_46_showLitString(aux2), u'"'))

# Prelude.Show.Python.Lib.TensorFlow.Matrix.Tensor xs dt implementation of Prelude.Show.Show, method show
def _idris_Prelude_46_Show_46_Python_46_Lib_46_TensorFlow_46_Matrix_46__64_Prelude_46_Show_46_Show_36_Tensor_32_xs_32_dt_58__33_show_58_0(
  e0, e1, e2
):
  while True:
    assert e2[0] == 0  # Python.Lib.TensorFlow.Matrix.MkT
    in0, in1 = e2[1:]
    return _idris_unsafePerformIO(
      None,
      None,
      _idris_Python_46_Functions_46__36__46_(
        None,
        None,
        (0,),  # Python.Telescope.Return
        _idris_Python_46_Fields_46__47__46_(None, None, in1, u'__str__', None),
        None,
        Unit
      )
    )
    return _idris_error("unreachable due to case in tail position")

# Python.Lib.TensorFlow.Matrix.reduce_reshape, removeFn, removeElem
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_reshape_58_removeFn_58_0_58_removeElem_58_0(
  e0, e1, e2, e3, e4, e5, e6, e7
):
  while True:
    aux1 = _idris_Prelude_46_List_46_splitAt(None, e7, e6)
    assert True  # Builtins.MkPair
    in0, in1 = aux1
    if in1:  # Prelude.List.::
      in2, in3 = in1.head, in1.tail
      aux3 = in3
    else:  # Prelude.List.Nil
      aux3 = None
    aux2 = aux3
    if aux2 is not None:  # Prelude.Maybe.Just
      in4 = aux2
      aux4 = in4
    else:  # Prelude.Maybe.Nothing
      aux4 = _idris__123_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_reshape_58_removeFn_58_0_58_removeElem_58_0_95_lam0_125_()
    return _idris_Prelude_46_List_46__43__43_(None, in0, aux4)
    return _idris_error("unreachable due to case in tail position")

# Prelude.List.sortBy, splitRec
def _idris_Prelude_46_List_46_sortBy_58_splitRec_58_2(e0, e1, e2, e3, e4, e5):
  while True:
    if e4:  # Prelude.List.::
      in0, in1 = e4.head, e4.tail
      if e3:  # Prelude.List.::
        in2, in3 = e3.head, e3.tail
        if in3:  # Prelude.List.::
          in4, in5 = in3.head, in3.tail
          e0, e1, e2, e3, e4, e5, = None, None, None, in5, in1, (65680, None, None, None, e5, (65736, in0)),  # {U_Prelude.Basics..1}, {U_{Prelude.List.sortBy:splitRec:2_lam0}1}
          continue
          return _idris_error("unreachable due to tail call")
        else:
          return (APPLY0(e5, ConsList()), e4)
        return _idris_error("unreachable due to case in tail position")
      else:
        return (APPLY0(e5, ConsList()), e4)
      return _idris_error("unreachable due to case in tail position")
    else:
      return (APPLY0(e5, ConsList()), e4)
    return _idris_error("unreachable due to case in tail position")

# Prelude.Show.showLitChar, asciiTab
def _idris_Prelude_46_Show_46_showLitChar_58_asciiTab_58_10(e0):
  while True:
    return ConsList().cons(u'US').cons(u'RS').cons(u'GS').cons(u'FS').cons(u'ESC').cons(u'SUB').cons(u'EM').cons(u'CAN').cons(u'ETB').cons(u'SYN').cons(u'NAK').cons(u'DC4').cons(u'DC3').cons(u'DC2').cons(u'DC1').cons(u'DLE').cons(u'SI').cons(u'SO').cons(u'CR').cons(u'FF').cons(u'VT').cons(u'LF').cons(u'HT').cons(u'BS').cons(u'BEL').cons(u'ACK').cons(u'ENQ').cons(u'EOT').cons(u'ETX').cons(u'STX').cons(u'SOH').cons(u'NUL')

# Prelude.Show.showLitChar, getAt
def _idris_Prelude_46_Show_46_showLitChar_58_getAt_58_10(e0, e1, e2):
  while True:
    if e2:  # Prelude.List.::
      in0, in1 = e2.head, e2.tail
      if e1 == 0:
        return in0
      else:
        in2 = (e1 - 1)
        e0, e1, e2, = None, in2, in1,
        continue
        return _idris_error("unreachable due to tail call")
      return _idris_error("unreachable due to case in tail position")
    else:  # Prelude.List.Nil
      return None
    return _idris_error("unreachable due to case in tail position")

# with block in Prelude.Strings.strM
def _idris__95_Prelude_46_Strings_46_strM_95_with_95_22(e0, e1):
  while True:
    if e1[0] == 1:  # Prelude.Basics.No
      return _idris_really_95_believe_95_me(None, None, (0,))  # Prelude.Strings.StrNil
    else:  # Prelude.Basics.Yes
      return _idris_really_95_believe_95_me(None, None, (1, e0[0], e0[1:]))  # Prelude.Strings.StrCons
    return _idris_error("unreachable due to case in tail position")

# with block in Prelude.Strings.unpack
def _idris__95_Prelude_46_Strings_46_unpack_95_with_95_25(e0, e1):
  while True:
    if e1[0] == 1:  # Prelude.Strings.StrCons
      in0, in1 = e1[1:]
      return _idris__95_Prelude_46_Strings_46_unpack_95_with_95_25(
        None,
        _idris_Prelude_46_Strings_46_strM(in1)
      ).cons(in0)
    else:  # Prelude.Strings.StrNil
      return ConsList()
    return _idris_error("unreachable due to case in tail position")

# with block in Prelude.Interfaces.Prelude.Show.Prec implementation of Prelude.Interfaces.Ord, method >
def _idris__95_Prelude_46_Interfaces_46_Prelude_46_Show_46__64_Prelude_46_Interfaces_46_Ord_36_Prec_58__33__62__58_0_95_with_95_27(
  e0, e1, e2
):
  while True:
    if e0[0] == 2:  # Prelude.Interfaces.GT
      return True
    else:
      return False
    return _idris_error("unreachable due to case in tail position")

# with block in Prelude.Show.firstCharIs
def _idris__95_Prelude_46_Show_46_firstCharIs_95_with_95_45(e0, e1, e2):
  while True:
    if e2[0] == 1:  # Prelude.Strings.StrCons
      in0, in1 = e2[1:]
      return APPLY0(e0, in0)
    else:  # Prelude.Strings.StrNil
      return False
    return _idris_error("unreachable due to case in tail position")

# with block in Prelude.Interfaces.Prelude.Interfaces.Char implementation of Prelude.Interfaces.Ord, method <
def _idris__95_Prelude_46_Interfaces_46_Prelude_46_Interfaces_46__64_Prelude_46_Interfaces_46_Ord_36_Char_58__33__60__58_0_95_with_95_146(
  e0, e1, e2
):
  while True:
    if e0[0] == 0:  # Prelude.Interfaces.LT
      return True
    else:
      return False
    return _idris_error("unreachable due to case in tail position")

# with block in Prelude.Interfaces.Prelude.Interfaces.Char implementation of Prelude.Interfaces.Ord, method >
def _idris__95_Prelude_46_Interfaces_46_Prelude_46_Interfaces_46__64_Prelude_46_Interfaces_46_Ord_36_Char_58__33__62__58_0_95_with_95_149(
  e0, e1, e2
):
  while True:
    if e0[0] == 2:  # Prelude.Interfaces.GT
      return True
    else:
      return False
    return _idris_error("unreachable due to case in tail position")

# Prelude.List.case block in init' at ./Prelude/List.idr:175:8
def _idris_Prelude_46_List_46_init_39__95__95__95__95__95_Prelude_95__95_List_95__95_idr_95_175_95_8_95_case(
  e0, e1, e2, e3
):
  while True:
    if e2:  # Prelude.List.::
      in0, in1 = e2.head, e2.tail
      aux1 = _idris_Prelude_46_List_46_init_39_(None, in1.cons(in0))
      if aux1 is not None:  # Prelude.Maybe.Just
        in2 = aux1
        return in2.cons(e1)
      else:  # Prelude.Maybe.Nothing
        return None
      return _idris_error("unreachable due to case in tail position")
    else:  # Prelude.List.Nil
      return ConsList()
    return _idris_error("unreachable due to case in tail position")

# Prelude.List.case block in case block in init' at ./Prelude/List.idr:175:8 at ./Prelude/List.idr:178:12
def _idris_Prelude_46_List_46_init_39__95__95__95__95__95_Prelude_95__95_List_95__95_idr_95_175_95_8_95_case_95__95__95__95__95_Prelude_95__95_List_95__95_idr_95_178_95_12_95_case(
  e0, e1, e2, e3, e4, e5
):
  while True:
    if e5 is not None:  # Prelude.Maybe.Just
      in0 = e5
      return in0.cons(e1)
    else:  # Prelude.Maybe.Nothing
      return None
    return _idris_error("unreachable due to case in tail position")

# Prelude.List.case block in mergeBy at ./Prelude/List.idr:772:8
def _idris_Prelude_46_List_46_mergeBy_95__95__95__95__95_Prelude_95__95_List_95__95_idr_95_772_95_8_95_case(
  e0, e1, e2, e3, e4, e5, e6
):
  while True:
    if e6[0] == 0:  # Prelude.Interfaces.LT
      return _idris_Prelude_46_List_46_mergeBy(None, e1, e3, e5.cons(e4)).cons(e2)
    else:
      return _idris_Prelude_46_List_46_mergeBy(None, e1, e3.cons(e2), e5).cons(e4)
    return _idris_error("unreachable due to case in tail position")

# Prelude.List.case block in sortBy at ./Prelude/List.idr:787:22
def _idris_Prelude_46_List_46_sortBy_95__95__95__95__95_Prelude_95__95_List_95__95_idr_95_787_95_22_95_case(
  e0, e1, e2, e3
):
  while True:
    assert True  # Builtins.MkPair
    in0, in1 = e3
    return _idris_Prelude_46_List_46_mergeBy(
      None,
      e1,
      _idris_Prelude_46_List_46_sortBy(None, e1, in0),
      _idris_Prelude_46_List_46_sortBy(None, e1, in1)
    )
    return _idris_error("unreachable due to case in tail position")

# Prelude.Show.case block in showLitChar at ./Prelude/Show.idr:128:27
def _idris_Prelude_46_Show_46_showLitChar_95__95__95__95__95_Prelude_95__95_Show_95__95_idr_95_128_95_27_95_case(
  e0, e1
):
  while True:
    if e1 is not None:  # Prelude.Maybe.Just
      in0 = e1
      return (65680, None, None, None, (65732, u'\\'), (65698, in0))  # {U_Prelude.Basics..1}, {U_prim__strCons1}, {U_Prelude.Show.{showLitChar_____Prelude__Show__idr_128_27_case_lam0}1}
    else:  # Prelude.Maybe.Nothing
      aux2 = _idris_Prelude_46_Interfaces_46_Prelude_46_Interfaces_46__64_Prelude_46_Interfaces_46_Ord_36_Char_58__33_compare_58_0(
        e0,
        u'\u007f'
      )
      if aux2[0] == 2:  # Prelude.Interfaces.GT
        aux3 = True
      else:
        aux3 = False
      aux1 = aux3
      if not aux1:  # Prelude.Bool.False
        return (65732, e0)  # {U_prim__strCons1}
      else:  # Prelude.Bool.True
        return (
          65680,  # {U_Prelude.Basics..1}
          None,
          None,
          None,
          (65732, u'\\'),  # {U_prim__strCons1}
          (
            65685,  # {U_Prelude.Show.protectEsc1}
            (65682,),  # {U_Prelude.Chars.isDigit1}
            _idris_Prelude_46_Show_46_primNumShow(None, (65733,), (0,), ord(e0))  # {U_prim__toStrInt1}, Prelude.Show.Open
          )
        )
      return _idris_error("unreachable due to case in tail position")
    return _idris_error("unreachable due to case in tail position")

# Python.Lib.TensorFlow.Matrix.case block in Python.Lib.TensorFlow.Matrix.reduce_reshape, removeFn, removeElem at ./Python/Lib/TensorFlow/Matrix.idr:183:32
def _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_reshape_58_removeFn_58_0_58_removeElem_58_0_95__95__95__95__95_Python_95__95_Lib_95__95_TensorFlow_95__95_Matrix_95__95_idr_95_183_95_32_95_case(
  e0, e1, e2, e3, e4, e5, e6, e7, e8
):
  while True:
    assert True  # Builtins.MkPair
    in0, in1 = e8
    if in1:  # Prelude.List.::
      in2, in3 = in1.head, in1.tail
      aux2 = in3
    else:  # Prelude.List.Nil
      aux2 = None
    aux1 = aux2
    if aux1 is not None:  # Prelude.Maybe.Just
      in4 = aux1
      aux3 = in4
    else:  # Prelude.Maybe.Nothing
      aux3 = _idris_Python_46_Lib_46_TensorFlow_46_Matrix_46__123_Python_46_Lib_46_TensorFlow_46_Matrix_46_reduce_95_reshape_58_removeFn_58_0_58_removeElem_58_0_95__95__95__95__95_Python_95__95_Lib_95__95_TensorFlow_95__95_Matrix_95__95_idr_95_183_95_32_95_case_95_lam0_125_()
    return _idris_Prelude_46_List_46__43__43_(None, in0, aux3)
    return _idris_error("unreachable due to case in tail position")

# case block in io_bind at IO.idr:107:34
def _idris_io_95_bind_95_IO_95__95_idr_95_107_95_34_95_case(
  e0, e1, e2, e3, e4, e5, e6, e7
):
  while True:
    return APPLY0(e7, e5)

# case block in Void
def _idris_Void_95_case():
  while True:
    return None

# <<Void eliminator>>
def _idris_Void_95_elim():
  while True:
    return None

# export: Main.exports, greet2
def greet2(arg1):
  APPLY0(_idris_Main_46_exports_58_greet2_58_0(arg1), World)

if __name__ == '__main__':
  runMain0()
