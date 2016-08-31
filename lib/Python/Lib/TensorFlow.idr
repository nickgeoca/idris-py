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
TensorElemType : Type
TensorElemType = String

-------------------------------------------------- 
-- Tensor Variable
Variable_PS : Signature
Variable_PS f = case f of
  "__str__" => [] ~~> String
  "assign"  => [Tensor_P] ~~> ()
  _ => Object f

Variable_P : Type
Variable_P = Obj Variable_PS



-------------------------------------------------- 
-- Op type
Op_PS : Signature
Op_PS f = case f of
  "__str__" => [] ~~> String
  _ => Object f

Op_P : Type
Op_P = Obj Op_PS

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
Session_PS f = case f of
 "run" => [Fetch_P, Dictionary_P (Tensor_P, Arr)] ~~> Fetch_P -- NOTE: The return on this signature is incorrect and must be cast.
 "close" => [] ~~> ()
 _ => Object f

Session_P : Type
Session_P = Obj Session_PS

-------------------------------------------------- 
-- 
TensorFlow : Signature
TensorFlow f = case f of
  -- Session
  "Session" => [] ~~> Session_P

  -- Variable
  "Variable" => [Tensor_P, TensorElemType] ~~> Tensor_P
  "random_uniform_initializer" => [Double, Double, Int, TensorElemType] ~~> ([Obj $ PyList Nat] ~> Tensor_P)

  -- Ops
  "initialize_all_variables" => [] ~~> Op_P

  -- Tensor transformations
  "cast" => [Tensor_P, TensorElemType] ~~> Tensor_P

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
  "ones" => [Obj $ PyList Nat, TensorElemType] ~~> Tensor_P
  "zeros" => [Obj $ PyList Nat, TensorElemType] ~~> Tensor_P

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
  "placeholder" => [TensorElemType, Obj $ PyList Nat] ~~> Tensor_P

  -- Module
  _ => Module f

TensorFlowKludge : Signature
TensorFlowKludge f = case f of
  "test" => [] ~~> List (Tensor_P)
  "while_loop" => [List (Tensor_P) -> Bool, List (Tensor_P) -> List (Tensor_P), List (Tensor_P), Int, Bool, Bool] ~~> List (Tensor_P)
  _ => Module f


TensorFlowNN : Signature
TensorFlowNN f = case f of
  "softmax" => [Tensor_P] ~~> Tensor_P
  _ => Module f


import_ : PIO $ Obj TensorFlow
import_ = importModule "tensorflow"
