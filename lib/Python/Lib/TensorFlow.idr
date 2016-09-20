module Python.Lib.TensorFlow

import Python.Lib.Numpy as Np
import Python
import Python.Prim
import Data.Erased

%default total
%access public export


-------------------------------------------------- 
-- Tensor 
Tensor_PS : Signature
Tensor_PS f = case f of
  "__str__" => [] ~~> String
  _ => Object f

Tensor_P : Type
Tensor_P = Obj Tensor_PS

-------------------------------------------------- 
-- Tensor Element
TensorElemType_PS : Signature
TensorElemType_PS f = case f of
  _ => Object f

TensorElemType_P : Type
TensorElemType_P = Obj TensorElemType_PS

-------------------------------------------------- 
-- Op type
Op_PS : Signature
Op_PS f = case f of
  "__str__" => [] ~~> String
  _ => Object f

Op_P : Type
Op_P = Obj Op_PS

-------------------------------------------------- 
-- Tensor Variable
Variable_PS : Signature
Variable_PS f = case f of
  "__str__" => [] ~~> String
  "assign"  => [Tensor_P] ~~> Op_P
  _ => Object f

Variable_P : Type
Variable_P = Obj Variable_PS

-------------------------------------------------- 
-- Fetch type
Fetch_PS : Signature
Fetch_PS f = case f of
  _ => Object f

Fetch_P : Type
Fetch_P = Obj Fetch_PS

--------------------------------------------------
-- Session type

GraphElem_PS : Signature
GraphElem_PS f = case f of
  _ => Object f
  
GraphElem_P : Type
GraphElem_P = Obj GraphElem_PS


Session_PS : Signature
Session_PS f = case f of -- TODO: Is the signature on run List Fetch_P or List Op_P
 "__str__" => [] ~~> String
 "run" => [Obj $ PyList Fetch_P, Dictionary_P (Tensor_P, Arr)] ~~> Obj (PyList Fetch_P) -- NOTE: The return on this signature is incorrect and must be cast.
 "close" => [] ~~> ()
 _ => Object f

Session_P : Type
Session_P = Obj Session_PS

-------------------------------------------------- 
-- 
TensorFlow : Signature
TensorFlow f = case f of
  -- Element types
  "float16" => Attr TensorElemType_P
  "float32" => Attr TensorElemType_P
  "float64" => Attr TensorElemType_P
  "int8" => Attr TensorElemType_P
  "int16" => Attr TensorElemType_P
  "int32" => Attr TensorElemType_P
  "int64" => Attr TensorElemType_P
  "uint8" => Attr TensorElemType_P
  "bool" => Attr TensorElemType_P
  "complex64" => Attr TensorElemType_P
  "complex128" => Attr TensorElemType_P
  "qint8" => Attr TensorElemType_P
  "qint32" => Attr TensorElemType_P
  "quint8" => Attr TensorElemType_P

  -- Session
  "Session" => [] ~~> Session_P

  -- Variable
  "Variable" => [Tensor_P, TensorElemType_P] ~~> Tensor_P
  "random_uniform_initializer" => [Double, Double, Int, TensorElemType_P] ~~> ([Obj $ PyList Nat] ~> Tensor_P)

  -- Ops
  "initialize_all_variables" => [] ~~> Op_P

  -- Tensor transformations
  "cast" => [Tensor_P, TensorElemType_P] ~~> Tensor_P

  -- Math
  "abs"  => [Tensor_P] ~~> Tensor_P
  "add"  => [Tensor_P, Tensor_P] ~~> Tensor_P
  "div"  => [Tensor_P, Tensor_P] ~~> Tensor_P
  "log"  => [Tensor_P] ~~> Tensor_P
  "matmul" => [Tensor_P, Tensor_P] ~~> Tensor_P
  "mul" => [Tensor_P, Tensor_P] ~~> Tensor_P
  "neg" => [Tensor_P] ~~> Tensor_P
  "sub" => [Tensor_P, Tensor_P] ~~> Tensor_P

  -- Reduction
  "reduce_mean" => [Tensor_P, Obj $ PyList Nat, Bool] ~~> Tensor_P
  "reduce_sum" => [Tensor_P, Obj $ PyList Nat, Bool] ~~> Tensor_P

  -- ...
  "ones" => [Obj $ PyList Nat, TensorElemType_P] ~~> Tensor_P
  "zeros" => [Obj $ PyList Nat, TensorElemType_P] ~~> Tensor_P
  "constant" => [Arr, TensorElemType_P] ~~> Tensor_P

  -- Control flow
  "argmax" => [Tensor_P, Obj $ PyList Nat] ~~> Tensor_P

  -- Control flow
  --..

  -- Comparison operators
  "equal" => [Tensor_P, Tensor_P] ~~> Tensor_P
  "not_equal" => [Tensor_P, Tensor_P] ~~> Tensor_P
  "greater" => [Tensor_P, Tensor_P] ~~> Tensor_P
  "greater_equal" => [Tensor_P, Tensor_P] ~~> Tensor_P
  "less" => [Tensor_P, Tensor_P] ~~> Tensor_P
  "less_equal" => [Tensor_P, Tensor_P] ~~> Tensor_P

  -- Comparison operators
  "greater" => [Tensor_P, Tensor_P] ~~> Bool
  "greater_equal" => [Tensor_P, Tensor_P] ~~> Bool
  "less" => [Tensor_P, Tensor_P] ~~> Bool
  "less_equal" => [Tensor_P, Tensor_P] ~~> Bool

  -- Placeholders
  "placeholder" => [TensorElemType_P, Obj $ PyList Nat] ~~> Tensor_P

  -- Training
  "gradients" => [Obj $ PyList Tensor_P, Obj $ PyList Tensor_P, Maybe $ Obj $ PyList Tensor_P, String, Bool, Bool] ~~> Obj (PyList Tensor_P)

  -- Module
  _ => Module f

TensorFlowKludge : Signature
TensorFlowKludge f = case f of
  -- "test" => [] ~~> List (Tensor_P) -- Change to Obj $ PyList Tensor_P
  -- "while_loop" => [List (Tensor_P) -> Bool, List (Tensor_P) -> List (Tensor_P), List (Tensor_P), Int, Bool, Bool] ~~> List (Tensor_P)
  _ => Module f


TensorFlowNN : Signature
TensorFlowNN f = case f of
  "softmax" => [Tensor_P] ~~> Tensor_P
  "sigmoid" => [Tensor_P] ~~> Tensor_P
  _ => Module f


import_ : PIO $ Obj TensorFlow
import_ = importModule "tensorflow"

-- ziman wisdom:
-- "float32" => Attr TensorElemType_P
-- you'll lose the ability to call it as a function this way, though
-- to do that, you'd probably have to extend TensorElemType_P with __call__
