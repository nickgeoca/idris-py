module Python.Lib.TensorFlow.Matrix

import Python
import Python.Prim
import Python.Lib.TensorFlow
import Python.Lib.Numpy as Np
import Python.Lib.Numpy.Matrix

import Data.List.Quantifiers
import Data.Vect

%access public export
%default total


data ElemType 
  = Float64
  | Float32
  | Float16
  | Int32

-- Tensor types
Shape : Type
Shape = List Nat

export 
record Tensor (shape : Shape) (dtype : ElemType) where
  constructor MkT
  tensor_p : Tensor_P

TensorList : List (Shape, ElemType) -> Type 
TensorList ts = All fn ts
  where fn : (Shape, ElemType) -> Type
        fn (xs,dt) = Tensor xs dt


TFCondition : List (Shape, ElemType) -> Type
TFCondition a = TensorList a -> Bool

TFBody : List (Shape, ElemType) -> Type
TFBody a = TensorList a -> TensorList a 


-- Type Session
export 
record Session where
  constructor MkS
  session_p : Session_P

-- Type Session
export 
record Variable where
  constructor MkV
  variable_p : Variable_P

-- Fn
private
tf : Obj TensorFlow
tf = unsafePerformIO TensorFlow.import_

private 
unsafeTfSession : PIO (Session_P) -> Session
unsafeTfSession = MkS . unsafePerformIO

-------------------------
implementation Show (Tensor xs dt) where
  show (MkT x) = unsafePerformIO $ x /. "__str__" $. []


----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
-- Tensorflow Api

-------------------------
-- Session
export
session : PIO Session
session = do sess <- tf /. "Session" $. []
             return $ MkS sess

export 
run : PIO Session -> Tensor xs dt -> PIO $ MatrixN xs DDouble
run sM (MkT tObj) = do s <- sM
                       m <- (session_p s) /. "run" $. [tObj]
                       return $ MkM' m

tensorToMatrix : List (Shape, ElemType) -> List (Shape, DType Double)
tensorToMatrix = map convert
  where convert : (Shape, ElemType) -> (Shape, DType Double)
        convert (s,_) = (s, DDouble)


export 
run' : PIO Session -> TensorList xs -> PIO $ MatrixList $ tensorToMatrix xs 
run' sM ts = do s <- sM
                m <- (session_p s) /. "run'" $. [tensorObjects]
                believe_me 'a' -- Bug: Fix this function
                -- return $ MkM' m
  where     
  tensorObjects = believe_me 'a' -- MkM param

export 
close : PIO Session -> PIO ()
close sM = do s <- sM
              m <- (session_p s) /. "close" $. []
              return ()

-------------------------
-- Variable
export -- tf.Variable.__init__(initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None)
session : Tensor xs dt -> PIO Variable
session (MkT initial_value) = do var <- tf /. "Variable" $. [initial_value]
                                 return $ MkV var
{-
-- __init__(
initial_value=None
trainable=True
collections=None
validate_shape=True
caching_device=None
name=None
variable_def=None
dtype=None)
-}

-------------------------
-- Math

export -- tf.abs(x, name=None)
abs : Tensor xs dt -> Tensor xs dt
abs (MkT x) = MkT . unsafePerformIO $ tf /. "abs" $. [x]

export -- tf.add(x, y, name=None)
add : Tensor xs dt -> Tensor xs dt -> Tensor xs dt
add (MkT x) (MkT y) = MkT . unsafePerformIO $ tf /. "add" $. [x, y]

export
(+) : Tensor xs dt -> Tensor xs dt -> Tensor xs dt
(+) = add

export -- tf.div(x, y, name=None)
div : Tensor xs dt -> Tensor xs dt -> Tensor xs dt
div (MkT x) (MkT y) = MkT . unsafePerformIO $ tf /. "div" $. [x, y]

export
(/) : Tensor xs dt -> Tensor xs dt -> Tensor xs dt
(/) = div

export -- tf.log(x, name=None)
log : Tensor xs dt -> Tensor xs dt
log (MkT x) = MkT . unsafePerformIO $ tf /. "log" $. [x]

export -- tf.matmul(a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None) 
matmul : Tensor [b, k] dt -> Tensor [k, a] dt -> Tensor [b, a] dt
matmul (MkT x) (MkT y) = MkT . unsafePerformIO $ tf /. "matmul" $. [x, y]

export
(*>)   : Tensor [b, k] dt -> Tensor [k, a] dt -> Tensor [b, a] dt
(*>)   = matmul

export -- tf.mul(x, y, name=None)
mul : Tensor xs dt -> Tensor xs dt -> Tensor xs dt
mul (MkT x) (MkT y) = MkT . unsafePerformIO $ tf /. "mul" $. [x, y]

export
(*) : Tensor xs dt -> Tensor xs dt -> Tensor xs dt
(*) = mul

export -- tf.neg(x, name=None)
neg : Tensor xs dt -> Tensor xs dt
neg (MkT x) = MkT . unsafePerformIO $ tf /. "neg" $. [x]

export -- tf.sub(x, y, name=None)
sub : Tensor xs dt -> Tensor xs dt -> Tensor xs dt
sub (MkT x) (MkT y) = MkT . unsafePerformIO $ tf /. "sub" $. [x, y]

export
(-) : Tensor xs dt -> Tensor xs dt -> Tensor xs dt
(-) = sub

-------------------------
-- Reduction

-- TODO: Is it better to replace remove_dims parameter with Maybe type? 
-- TODO:  * It removes use of having two different named functions. E.g. reduce_mean and reduce_mean'
-- TODO: Is it better to throw error when passing duplicate remove dimensions?
-- TODO: Is it a bug when reducing shape of [] for various settings?

-- BUG: All reduce functions: ex, where x has shape [1,1]:  reduce_mean x [0,1,3,4] True
reduce_reshape : Bool 
              -> (remove_dims : List Nat) 
              -> (shape : List Nat)
              -> List Nat
reduce_reshape b_keep_rank remove_dims shape 
  = if b_keep_rank 
    then shape 
    else removeFn remove_dims shape
      where
      removeFn : List Nat -> List Nat -> List Nat
      removeFn remove shape = foldl removeElem shape remove_ordered 
        where 
        remove_ordered : List Nat
        remove_ordered = reverse $ sort remove
        removeElem : List a -> Nat -> List a
        removeElem ls n = case splitAt n ls of (l, r) => l ++ fromMaybe Nil (tail' r)

remove_all_dims : List Nat -> List Nat
remove_all_dims shape = fromMaybe Nil $ init' [0 .. length shape]

export -- tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
reduce_mean : Tensor shape dt
           -> (remove_dims : List Nat)
           -> (b_keep_rank : Bool)
           -> Tensor (reduce_reshape b_keep_rank remove_dims shape) dt
reduce_mean (MkT x) remove_dims b_keep_rank 
  = MkT . unsafePerformIO $ tf /. "reduce_mean" $. [x, pyList remove_dims, b_keep_rank]


export -- tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
reduce_mean' : Tensor shape dt 
            -> (b_keep_rank : Bool) 
            -> Tensor (if b_keep_rank then shape else []) dt
reduce_mean' {shape=shape} (MkT x) b_keep_rank 
  = MkT . unsafePerformIO $ tf /. "reduce_mean" $. [x, pyList $ remove_all_dims shape, b_keep_rank]


export -- tf.reduce_sum(input_tensor, reduction_indices=None, keep_dims=False, name=None)
reduce_sum : Tensor shape dt
          -> (remove_dims : List Nat)
          -> (b_keep_rank : Bool)
          -> Tensor (reduce_reshape b_keep_rank remove_dims shape) dt
reduce_sum (MkT x) remove_dims b_keep_rank 
  = MkT . unsafePerformIO $ tf /. "reduce_sum" $. [x, pyList remove_dims, b_keep_rank]

export -- tf.reduce_sum(input_tensor, reduction_indices=None, keep_dims=False, name=None)
reduce_sum' : Tensor shape dt 
            -> (b_keep_rank : Bool) 
            -> Tensor (if b_keep_rank then shape else []) dt
reduce_sum' {shape=shape} (MkT x) b_keep_rank 
  = MkT . unsafePerformIO $ tf /. "reduce_sum" $. [x, pyList $ remove_all_dims shape, b_keep_rank]


-------------------------
-- Shapes and shaping

{-
export -- tf.concat(concat_dim, values, name='concat')
concat : TensorList a -> Tensor xs dt
cocnat = _
-}

-------------------------
-- Constants, Sequences, and Random Values:

export  -- tf.ones(shape, dtype=tf.float32, name=None)
ones : Tensor xs dt
ones {xs=xs} = MkT . unsafePerformIO $ tf /. "ones" $. [pyList xs]

export  -- tf.zeros(shape, dtype=tf.float32, name=None)
zeros : Tensor xs dt
zeros {xs=xs} = MkT . unsafePerformIO $ tf /. "zeros" $. [pyList xs]

{-
 (dt : DType a) -> Vect r (Vect c a) -> Matrix r c dt
tf.constant(value, dtype=None, shape=None, name='Const')
constant : Int -> ElemType -> Shape -> Maybe String -> Matrix
constant i  _ _ = _
-}


-------------------------
-- Comparison operators

export -- tf.greater(x, y, name=None)
greater : Tensor xs dt -> Tensor xs dt -> Bool
greater (MkT x) (MkT y) = unsafePerformIO $ tf /. "greater" $. [x, y]

export
(>) : Tensor xs dt -> Tensor xs dt -> Bool
(>) = greater

export -- tf.greater_equal(x, y, name=None)
greater_equal : Tensor xs dt -> Tensor xs dt -> Bool
greater_equal (MkT x) (MkT y) = unsafePerformIO $ tf /. "greater_equal" $. [x, y]

export
(>=) : Tensor xs dt -> Tensor xs dt -> Bool
(>=) = greater_equal

export -- tf.less(x, y, name=None)
less : Tensor xs dt -> Tensor xs dt -> Bool
less (MkT x) (MkT y) = unsafePerformIO $ tf /. "less" $. [x, y]

export
(<) : Tensor xs dt -> Tensor xs dt -> Bool
(<) = less

export -- tf.less_equal(x, y, name=None)
less_equal : Tensor xs dt -> Tensor xs dt -> Bool
less_equal (MkT x) (MkT y) = unsafePerformIO $ tf /. "less_equal" $. [x, y]

export
(<=) : Tensor shape type -> Tensor shape type -> Bool
(<=) = less_equal



-------------------------
-- Control flow
{-
test_ {a=a} = toL {a=a} (unsafePerformIO $ tf /. "test_" $. []) 
  where
  toL : {a : List (Shape, ElemType)} -> List (Obj Tensor) -> TensorList a -- {a : List (Shape, ElemType)} -> a -> List (Obj Tensor) -> TensorList a
  toL {a=a} (l::ls) = case a of
    (s,t) :: ts  => (the (Tensor s t) (MkT l)) :: (toL {a=ts} ls)
    Nil          => Nil
  toL {a=a} Nil  = case a of
    (s,t) :: ts  => believe_me 'a'
    Nil          => Nil
-- -}

{-
export
test : TensorList ts
test 
  = toTensorList $ unsafePerformIO $ kludge /. "test" $. []
  where
  kludge : Obj TensorFlowKludge
  kludge = unsafePerformIO $ importModule "kludge"
  toTensorList : List (Obj Tensor) -> TensorList ts
  toTensorList {ts=(shape,type)::ts'} (x::xs) = (the (Tensor shape type) (MkT x)) :: (the (TensorList ts') (toTensorList xs))
  toTensorList {ts=Nil} _ = Nil
  toTensorList _ _ impossible
-}

{-
export -- tf.while_loop(cond, body, loop_vars, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)
while_loop : TFCondition a -> TFBody a -> TensorList a -> Int -> Bool -> Bool -> TensorList a
while_loop cond body vars parallelIterations backProp swapMemory 
  = believe_me $ unsafePerformIO $ kludge /. "while_loop" $. [cond, body, vars, parallelIterations, backProp, swapMemory]
-}

-------------------------
-- Datatypes
toTfType : ElemType -> Obj TensorElemType_PS
toTfType dt = case dt of
         Float64 => unsafePerformIO $ tf /. "float64" $. []
         Float32 => unsafePerformIO $ tf /. "float32" $. []
         Float16 => unsafePerformIO $ tf /. "float16" $. []
         Int32   => unsafePerformIO $ tf /. "int32" $. []

-------------------------
-- Placeholders
export -- tf.placeholder(dtype, shape=None, name=None)
placeholder : Tensor xs dt
placeholder {dt=dt} = MkT . unsafePerformIO $ tf /. "placeholder" $. [toTfType dt]


----------------------------------------------------------------------------------------------------
implementation Num (Tensor [] dt) where
  (+) = add
  (*) = mul
  fromInteger = believe_me -- TODO/BUG: Consider how to handle this case. There is no Integer type (abitrary precision integer) in tensorflow.
                           -- TODO/BUG:  fromInteger : Num ty => Integer -> ty

implementation Neg (Tensor [] dt) where
  negate = neg
  (-) = sub
  abs = Python.Lib.TensorFlow.Matrix.abs
