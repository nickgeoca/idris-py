module Python.Lib.TensorFlow

import Python.Lib.Numpy as Np
import Python
import Python.Prim
import Data.Erased

%default total
%access public export


-------------------------------------------------- 
-- Tensor types
Tensor_PS : Signature
Tensor_PS f = case f of
  "__str__" => [] ~~> String
  _ => Object f

Tensor_P : Type
Tensor_P = Obj Tensor_PS

TensorElemType_PS : Signature
TensorElemType_PS f = case f of
  _ => Object f
  
TensorElemType_P : Type
TensorElemType_P = Obj TensorElemType_PS

--------------------------------------------------
-- Session type
Session_PS : Signature
Session_PS f = case f of
 "run" => [Tensor_P] ~~> Obj Np.NDArray
 "run'" => [List $ Tensor_P] ~~> (List $ Obj Np.NDArray)
 "close" => [] ~~> ()
 _ => Object f

Session_P : Type
Session_P = Obj Session_PS

Variable : Signature
Variable f = case f of
 "run" => [Tensor_P] ~~> Obj Np.NDArray
 "run'" => [List $ Tensor_P] ~~> (List $ Obj Np.NDArray)
 "close" => [] ~~> ()
 _ => Object f


-------------------------------------------------- 
-- 
TensorFlow : Signature
TensorFlow f = case f of
  -- Session
  "Session" => [] ~~> Obj Session

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
  "ones" => [Obj $ PyList Nat] ~~> Tensor_P
  "zeros" => [Obj $ PyList Nat] ~~> Tensor_P

  -- Control flow
  --..

  -- Comparison operators
  "greater" => [Tensor_P, Tensor_P] ~~> Bool
  "greater_equal" => [Tensor_P, Tensor_P] ~~> Bool
  "less" => [Tensor_P, Tensor_P] ~~> Bool
  "less_equal" => [Tensor_P, Tensor_P] ~~> Bool

  -- Datatypes
  "float64" => [] ~~> TensorElemType_P
  "float32" => [] ~~> TensorElemType_P
  "float16" => [] ~~> TensorElemType_P
  "int32" => [] ~~> TensorElemType_P

  -- Comparison operators
  "greater" => [Tensor_P, Tensor_P] ~~> Bool
  "greater_equal" => [Tensor_P, Tensor_P] ~~> Bool
  "less" => [Tensor_P, Tensor_P] ~~> Bool
  "less_equal" => [Tensor_P, Tensor_P] ~~> Bool

  -- Placeholders
  "placeholder" => [TensorElemType_P] ~~> Tensor_P

  -- Module
  _ => Module f

TensorFlowKludge : Signature
TensorFlowKludge f = case f of
  "test" => [] ~~> List (Tensor_P)
  "while_loop" => [List (Tensor_P) -> Bool, List (Tensor_P) -> List (Tensor_P), List (Tensor_P), Int, Bool, Bool] ~~> List (Tensor_P)
  _ => Module f


import_ : PIO $ Obj TensorFlow
import_ = importModule "tensorflow"

