module Python.Lib.Numpy

import Python
import Python.Prim
import Data.Erased

%default total
%access public export

NDArray : Signature
NDArray f = case f of
  "__str__" => [] ~~> String
  _ => Object f

Arr : Type
Arr = Obj NDArray

{-
export
implementation Show Arr where
  show x = unsafePerformIO $ x /. "__str__" $. []
-}

--------------------------------------------------
NpElemType_PS : Signature
NpElemType_PS f = case f of
  _ => Object f

NpElemType_P : Type
NpElemType_P = Obj NpElemType_PS

--------------------------------------------------
ArithT : Type -> Signature
ArithT a f = case f of
  "__add__" => [a, a] ~~> a
  "__mul__" => [a, a] ~~> a
  "__sub__" => [a, a] ~~> a
  "__div__" => [a, a] ~~> a
  "__str__" => [a] ~~> String
  _ => PyType f

Mat : Nat -> Nat -> Maybe a -> Signature
Mat _ _ _ = Object

NDArrayT : Signature
NDArrayT f = case f of
  "transpose" => [Arr] ~~> Arr
  _ => ArithT Arr f

  "transpose_fulldep" => fun $
    forall r : Nat .
      forall c : Nat .
        forall a : Type .
          forall dtype : (Maybe a) .
            pi m : (Obj $ Mat r c dtype) .
              Return (Obj $ Mat c r dtype)

Numpy : Signature
Numpy f = case f of

  "array" => PAttr _ $ \a : Type =>
      [Obj (PyList (Obj (PyList a))), String] ~> Arr

  "reshape" => [Arr, (Nat, Nat)] ~~> Arr
  "abs" => [Arr] ~~> Arr
  "dot" => [Arr, Arr] ~~> Arr
  "transpose" => [Arr] ~~> Arr
  "full" => [Obj $ PyList Nat, Double, NpElemType_P] ~~> Arr
  "ndarray" => Attr $ Obj NDArrayT

  -- Datatypes
  "bool" => [] ~~> NpElemType_P
  "int" => [] ~~> NpElemType_P
  "intc" => [] ~~> NpElemType_P
  "intp" => [] ~~> NpElemType_P
  "int8" => [] ~~> NpElemType_P
  "int16" => [] ~~> NpElemType_P
  "int32" => [] ~~> NpElemType_P
  "int64" => [] ~~> NpElemType_P
  "uint8" => [] ~~> NpElemType_P
  "uint16" => [] ~~> NpElemType_P
  "uint32" => [] ~~> NpElemType_P
  "uint64" => [] ~~> NpElemType_P
  "float" => [] ~~> NpElemType_P
  "float16" => [] ~~> NpElemType_P
  "float32" => [] ~~> NpElemType_P
  "float64" => [] ~~> NpElemType_P
  "complex" => [] ~~> NpElemType_P
  "complex64" => [] ~~> NpElemType_P
  "complex128" => [] ~~> NpElemType_P



  _ => Module f

import_ : PIO $ Obj Numpy
import_ = importModule "numpy"
