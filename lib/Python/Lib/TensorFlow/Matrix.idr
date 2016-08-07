module Python.Lib.TensorFlow.Matrix

import Python
import Python.Prim
import Python.Lib.TensorFlow
import Python.Lib.Numpy
import Python.Lib.Numpy.Matrix as Np


import Data.List.Quantifiers
import Data.Vect.Quantifiers
import Data.Vect

import Control.Monad.State

%access public export
%default total

-- Operator Precedence
infixl 8 +.
infixl 8 -.
infixl 9 *.
-- infixl 9 /. TODO: Fix this
-- infixl 9 *> TODO: Fix this

-- This library does not support DT_STRING (variable length byte array).
-- Python TensorFlow may not fully support Float16
data ElemType 
  = Float16
  | Float32
  | Float64
  | Int8
  | Int16
  | Int32
  | Int64
  | UInt8
  | TFBool
  | Complex64
  | Complex128
  | QInt8
  | QInt32
  | QUInt8

-- Tensor types
Shape : Type
Shape = List Nat


{-
public export 
record Tensor (shape : Shape) (dtype : ElemType) where
  constructor MkT
  tensor_p : Tensor_P
-}

data Tensor : (shape : Shape) -> (dtype : ElemType) -> Type where
    TensorCon      : Tensor_P   -> Tensor shape dtype
    PlaceholderCon : Tensor_P   -> Tensor shape dtype
    VariableCon    : Variable_P -> Tensor shape dtype

toPyType : Tensor s dt -> Tensor_P
toPyType (TensorCon x) = x
toPyType (PlaceholderCon x) = x
toPyType (VariableCon x) = believe_me x

{-
public export 
record Placeholder (shape : Shape) (dtype : ElemType) where
  constructor MkPlc
  placeholder : Tensor shape dtype
-}

Tensors : {n : Nat} -> (dtTs : Vect n (Shape, ElemType)) -> Type
Tensors {n=n} _ = Vect n Tensor_P

export 
record Op where
  constructor MkOp
  op_p : Op_P

-- Type Session
export 
record Session where
  constructor MkS
  session_p : Session_P


-- Fn
private
tf : Obj TensorFlow
tf = unsafePerformIO TensorFlow.import_

private
op1 : (f : String)
  -> {auto pf : TensorFlow f = [Tensor_P] ~~> Tensor_P}
  -> Tensor ls dt -> Tensor ls dt
op1 f x = TensorCon . unsafePerformIO $ tf /. f $. [toPyType x]


private
op2 : (f : String)
  -> {auto pf : TensorFlow f = [Tensor_P, Tensor_P] ~~> Tensor_P}
  -> Tensor ls dt -> Tensor ls dt -> Tensor ls dt
op2 f x y = TensorCon . unsafePerformIO $ tf /. f $. [toPyType x, toPyType y]

private 
unsafeTfSession : PIO (Session_P) -> Session
unsafeTfSession = MkS . unsafePerformIO

-------------------------


implementation Show (Tensor xs dt) where
  show (TensorCon x) = unsafePerformIO $ x /. "__str__" $. []
  show (PlaceholderCon x) = unsafePerformIO $ x /. "__str__" $. []
  show (VariableCon x) = unsafePerformIO $ x /. "__str__" $. []


-------------------------
-- Helper functions (not tf api)

-- BUG: Make broadcast work
-- Broadcast example
-- [100,1,100,1] [100,1,100] => [100,100,100,100]
broadcast_dim : List Nat -> List Nat -> List Nat
broadcast_dim x y = y

toTfType : ElemType -> TensorElemType
toTfType dt = case dt of
         Float16 => "float16"
         Float32 => "float32"
         Float64 => "float64"
         Int8    => "int8"
         Int16   => "int16"
         Int32   => "int32"
         Int64   => "int64"
         UInt8   => "uint8"
         TFBool  => "bool"
         Complex64  => "complex64"
         Complex128 => "complex128"
         QInt8   => "qint8"
         QInt32  => "qint32"
         QUInt8  => "quint8"

----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
-- Tensorflow Api


implementation Cast ElemType NpElemType where
  cast t = case t of
    Float16 => Np.Float16
    Float32 => Np.Float32
    Float64 => Np.Float64
    Int8 => Np.Int8
    Int16 => Np.Int16
    Int32 => Np.Int32
    Int64 => Np.Int64
    UInt8 => Np.Uint8
    TFBool => Np.Bool
    Complex64 => Np.Complex64
    Complex128 => Np.Complex128
    -- NOTE: For the types below, there are probably corner cases where this won't work
    QInt8 => Np.Float32
    QInt32 => Np.Float32
    QUInt8 => Np.Float32

implementation Cast (Shape, ElemType) (Shape, NpElemType) where
  cast (s,e) = (s, cast e)

implementation Cast (Vect n (Shape, ElemType)) (Vect n (Shape, NpElemType)) where
  cast v = map cast v

implementation Cast (Vect n a) (List a) where
  cast (x::xs) = the (List a) $ x :: cast xs
  cast [] = []

-------------------------
-- Session
tensorToMatrix : List (Shape, ElemType) -> List (Shape, NpElemType)
tensorToMatrix = map convert
  where convert : (Shape, ElemType) -> (Shape, NpElemType)
        convert (s, dt) = (s, cast dt)


toSessionParam : (Shape, ElemType, NpElemType) -> Type
toSessionParam (shape, dtT, dtM) = (Tensor shape dtT, MatrixN shape dtM)

Session_Params : Vect n (Shape, ElemType, NpElemType) -> Type
Session_Params xs = All toSessionParam xs

    
export
session : PIO Session
session = do sess <- tf /. "Session" $. []
             return $ MkS sess

||| tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None)
export 
run : PIO Session 
   -> (fetches   : Tensor xs dt)
   -> (feed_dict : (Tensors dtTs, Matrices dtMs))
   -> PIO (MatrixN xs $ cast dt)
run sessM fetch phs = 
  do sess <- sessM
     m <- (session_p sess) /. "run" $. [toPyType fetch, feed_dict]  -- BUG: fetch should not be a placeholder
     return $ MkM' m
  where
  tensorList : List Tensor_P
  matrixList : List Arr
  feed_dict : Dictionary_P (Tensor_P, Arr) 

  tensorList = cast $ fst phs
  matrixList = cast $ snd phs
  feed_dict = dict $ pyDictionary $ zip tensorList matrixList

export 
close : PIO Session -> PIO ()
close sM = do s <- sM
              m <- (session_p s) /. "close" $. []
              return ()

-------------------------
-- Variable

------------
data PlaceholderParam (shape : Shape) (dt : ElemType) where
  tensorPlaceholder : Maybe $ Tensor shape dt
  matrixPlaceholder : MatrixN shape (cast dt)

data TFP where
  x : (Maybe $ Tensor [2] Float32, MatrixN [2] NpFloat32)
  y : (Maybe $ Tensor []  Float32, MatrixN []  NpFloat32)

TFV : (variables : Vect n (Shape, ElemType)) 
   -> Type
TFV {n=n} _ = Vect n Tensor_P

{-
TFV : {n : Nat} -> (dtTs : Vect n (Shape, ElemType)) -> Type
TFV {n=n} _ = Vect n Tensor_P
-- Tensors {n=n} _ = Vect n Tensor_P
data TFV (List (Shape, ElemType)) where
  variables : List (Tensor_P, (Shape, ElemType))
  x : PlaceholderParam [] Float32
data TF_PVs where
--  placeholders : List ((Maybe Tensor_P, Matrix_P), (Shape, ElemType))
  placeholders : Placeholders
  variables : List (Tensor_P, (Shape, ElemType))
Tensors : List (Shape, ElemType) 
       -> Type
Tensors _ = List Tensor_P
-}

------
https://eb.host.cs.st-andrews.ac.uk/drafts/eff-tutorial.pdf
https://github.com/idris-lang/Idris-dev/blob/master/libs/effects/Effect/StdIO.idr
http://docs.idris-lang.org/en/latest/effects/state.html
https://vimeo.com/123606435
ERROR THIS LINE OF CODE!!!!        -> PyState (TFP, TFV) (Tensor xs dt)

PyState : (state : Type)
       -> (type  : Type)
       -> Type
PyState s a = StateT s PIO a

||| tf.Variable.__init__(initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None)
||| @ initial_value can be placeholder, variable, or regular tensor
export 
variable : (initial_value : Tensor xs dt) 
        -> PyState (TFP, TFV) (Tensor xs dt)
variable initial_value = pure $ VariableCon . unsafePerformIO $ tf /. "Variable" $. [toPyType initial_value]
-- variable : (initial_value : Tensor xs dt) -> PyState (Tensor xs dt)
-- variable initial_value = pure $ VariableCon . unsafePerformIO $ tf /. "Variable" $. [toPyType initial_value]

export -- tf.initialize_all_variables()
initialize_all_variables : Op
initialize_all_variables = MkOp . unsafePerformIO $ tf /. "initialize_all_variables" $. []

-- TensorFlow ops should all be in the PIO monad b/c they are all potentially mutable with Variable tensor
-- it0 *
-- while_loop
--  

{-
How to handle immutable and mutalbe tensors
 * value       imut - ones
 * placeholder mut  - t <- pure ones
 * variable    mut  - t <- pure ones
Is a Variable just a pointer?
-}



{-
Value       - ImmutTensor : T
Variable    - MutabTensor : t <- (PIO T)
Placeholder - ParamTensor : t <- (M -> PIO T) matrix
-}


data VariableTensor : (x : Tensor shape dt) -> Type where
    IsVariableTensor : VariableTensor (VariableCon x)

fn : PIO (ls : List t ** NonEmpty ls) -> PIO ()  
fn = believe_me

||| tf.Variable.assign(value, use_locking=False)
export 
assign : PIO (variable : (Tensor xs dt) ** VariableTensor variable)
      ->     (value    : Tensor xs dt) 
      -> PIO ()
  = var /. "assign" $. [toPyType val] 
assign _ _ {ok=p} impossible



--  = pure $ VariableCon . unsafePerformIO $ var /. "assign" $. [toPyType val] 

{-
assign : PIO $ Tensor xs dt
      -> value : Tensor xs dt) 
      -> {auto ok  : VariableTensor variable}
      -> PIO ()
-}

{-
data Tensor_ : (shape : Shape) -> (dtype : ElemType) -> Type where
    Tensor'      : Tensor_P -> Tensor_ shape dtype
    Placeholder' : Tensor_P -> Tensor_ shape dtype
    Variable     : Tensor_P -> Tensor_ shape dtype
-}
{-
 -The fetches argument may be 
 - * a single graph element 
 - * a list of graph elements
 - * a dictionary whose values are the above
 -The type of fetches determines the return value of this method. 
 -A graph element can be one of the following types:
 - * If an element of fetches is an Operation, the corresponding fetched value will be None.
 - * If an element of fetches is a Tensor, the corresponding fetched value will be a numpy ndarray containing the value of that tensor.
 - * If an element of fetches is a SparseTensor, the corresponding fetched value will be a SparseTensorValue containing the value of that sparse tensor.
 - * If an element of fetches is produced by a get_tensor_handle op, the corresponding fetched value will be a numpy ndarray containing the handle of that tensor.
 -data GraphElem = GEOp | GET (Tensor xs dt) | GEST (STensor xs dt) | 
 -data Fetches = FetchL (List GraphElem) | Fetch GraphElem | Fetch
 -data FetchesIn  = FIO Op              | FIT (Tensor xs dt) | FIS (STensor xs dt)  | get_tensor_handle op
 -data FetchesOut = Nil                 | FOT (MatrixN xs dt)| FOS (STensorV xs dt) | handle of tensor (MatrixN xs dt)
 -List (Just )
 --}

{-
  do sess <- sessM
     m <- (session_p sess) /. "run" $. [fetch, feed_dict] 
     return $ MkM' m

export -- tf.gradients(ys, xs, grad_ys=None, name='gradients', colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None)
gradients : 
tf.gradients(ys, xs, grad_ys=None, name='gradients', colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None)
-- -}

-------------------------
-- Tensor transformations
export -- tf.cast(x, dtype, name=None)
cast : Tensor xs dt1 -> Tensor xs dt2
cast (MkT x) {dt2=dt2} = MkT . unsafePerformIO $ tf /. "cast" $. [x, toTfType dt2]


-------------------------
-- Math

export -- tf.abs(x, name=None)
abs : Tensor xs dt -> Tensor xs dt
abs = op1 "abs"

export -- tf.add(x, y, name=None)
add : Tensor xs dt -> Tensor xs dt -> Tensor xs dt
add  = op2 "add"

export
(+) : Tensor xs dt -> Tensor xs dt -> Tensor xs dt
(+) = add

export -- Broadcast add
(+.) : Tensor xs dt -> Tensor ys dt -> Tensor (broadcast_dim xs ys) dt
(+.) (MkT x) (MkT y) = MkT . unsafePerformIO $ tf /. "add" $. [x, y]

export -- tf.div(x, y, name=None)
div : Tensor xs dt -> Tensor xs dt -> Tensor xs dt
div = op2 "div"

export
(/) : Tensor xs dt -> Tensor xs dt -> Tensor xs dt
(/) = div

export -- Broadcast division
(/.) : Tensor xs dt -> Tensor ys dt -> Tensor (broadcast_dim xs ys) dt
(/.) (MkT x) (MkT y) = MkT . unsafePerformIO $ tf /. "div" $. [x, y]

export -- tf.log(x, name=None)
log : Tensor xs dt -> Tensor xs dt
log = op1 "log"

export -- tf.matmul(a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None) 
matmul : Tensor [b, k] dt -> Tensor [k, a] dt -> Tensor [b, a] dt
matmul (MkT x) (MkT y) = MkT . unsafePerformIO $ tf /. "matmul" $. [x, y]

export
(*>) : Tensor [b, k] dt -> Tensor [k, a] dt -> Tensor [b, a] dt
(*>) = matmul

export -- tf.mul(x, y, name=None)
mul : Tensor xs dt -> Tensor xs dt -> Tensor xs dt
mul = op2 "mul"

export
(*) : Tensor xs dt -> Tensor xs dt -> Tensor xs dt
(*) = mul

export -- Broadcast multipliation
(*.) : Tensor xs dt -> Tensor ys dt -> Tensor (broadcast_dim xs ys) dt
(*.) (MkT x) (MkT y) = MkT . unsafePerformIO $ tf /. "mul" $. [x, y]

export -- tf.neg(x, name=None)
neg : Tensor xs dt -> Tensor xs dt
neg = op1 "neg"

export -- tf.sub(x, y, name=None)
sub : Tensor xs dt -> Tensor xs dt -> Tensor xs dt
sub = op2 "sub"

export
(-) : Tensor xs dt -> Tensor xs dt -> Tensor xs dt
(-) = sub

export -- Broadcast subtract
(-.) : Tensor xs dt -> Tensor ys dt -> Tensor (broadcast_dim xs ys) dt
(-.) (MkT x) (MkT y) = MkT . unsafePerformIO $ tf /. "sub" $. [x, y]


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
ones {xs=xs} {dt=dt} = MkT . unsafePerformIO $ tf /. "ones" $. [pyList xs, toTfType dt]

export  -- tf.zeros(shape, dtype=tf.float32, name=None)
zeros : Tensor xs dt
zeros {xs=xs} = MkT . unsafePerformIO $ tf /. "zeros" $. [pyList xs, toTfType dt]


-------------------------
-- Comparison operators

export -- tf.equal(x, y, name=None)
equal : Tensor xs dt -> Tensor xs dt -> Tensor xs TFBool
equal (MkT x) (MkT y) = MkT . unsafePerformIO $ tf /. "equal" $. [x, y]

export
(==) : Tensor xs dt -> Tensor xs dt -> Tensor xs TFBool
(==) = equal

export -- tf.not_equal(x, y, name=None)
not_equal : Tensor xs dt -> Tensor xs dt -> Tensor xs TFBool
not_equal (MkT x) (MkT y) = MkT . unsafePerformIO $ tf /. "not_equal" $. [x, y]

infixl 5 !=
export
(!=) : Tensor xs dt -> Tensor xs dt -> Tensor xs TFBool
(!=) = not_equal

export -- tf.greater(x, y, name=None)
greater : Tensor xs dt -> Tensor xs dt -> Tensor xs TFBool
greater (MkT x) (MkT y) = MkT . unsafePerformIO $ tf /. "greater" $. [x, y]

export
(>) : Tensor xs dt -> Tensor xs dt -> Tensor xs TFBool
(>) = greater

export -- tf.greater_equal(x, y, name=None)
greater_equal : Tensor xs dt -> Tensor xs dt -> Tensor xs TFBool
greater_equal (MkT x) (MkT y) = MkT . unsafePerformIO $ tf /. "greater_equal" $. [x, y]

export
(>=) : Tensor xs dt -> Tensor xs dt -> Tensor xs TFBool
(>=) = greater_equal

export -- tf.less(x, y, name=None)
less : Tensor xs dt -> Tensor xs dt -> Tensor xs TFBool
less (MkT x) (MkT y) = MkT . unsafePerformIO $ tf /. "less" $. [x, y]

export
(<) : Tensor xs dt -> Tensor xs dt -> Tensor xs TFBool
(<) = less

export -- tf.less_equal(x, y, name=None)
less_equal : Tensor xs dt -> Tensor xs dt -> Tensor xs TFBool
less_equal (MkT x) (MkT y) = MkT . unsafePerformIO $ tf /. "less_equal" $. [x, y]

export
(<=) : Tensor xs dt -> Tensor xs dt -> Tensor xs TFBool
(<=) = less_equal


-------------------------
-- Sequence Comparison and Indexing

-- TODO: Validate argmax for all input tensor shapes
-- TODO: Consider if reduction arg type is better with (Tensor s1 Int32) or (List Nat)
export -- tf.argmax(input, dimension, name=None)
argmax : Tensor s0 dt -> (reduce_dims : List Nat) -> Tensor (reduce_reshape False reduce_dims s0) dt
argmax (MkT x) reduce_dims = MkT . unsafePerformIO $ tf /. "argmax" $. [x, pyList reduce_dims]

-------------------------
-- Control flow

{-
export -- tf.while_loop(cond, body, loop_vars, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)
while_loop : TFCondition a -> TFBody a -> TensorList a -> Int -> Bool -> Bool -> TensorList a
while_loop cond body vars parallelIterations backProp swapMemory 
  = believe_me $ unsafePerformIO $ kludge /. "while_loop" $. [cond, body, vars, parallelIterations, backProp, swapMemory]
-}

-------------------------
-- Placeholders
export -- tf.placeholder(dtype, shape=None, name=None)
placeholder : (accessor : (TFP -> Tensorxs xs dt))
           -> PyState (TFP, TFV) (Tensor xs dt)
placeholder {xs=xs} {dt=dt} accessor = 
  do t <- lift $ pure $ MkT . unsafePerformIO $ tf /. "placeholder" $. [toTfType dt, pyList xs]
     updatePlaceholders accessor t   
     pure t
  where
  updatePlaceholders : (TFP -> Tensorxs xs dt) 
                    -> Tensor xs dt
                    -> PyState (TFP, TFV) (Tensor xs dt)
  updatePlaceholders accessor t = 
    do pvs <- get 
       let ps = record {accessor = Just t} $ fst pvs
           vs = snd pvs
       put (ps, vs)
      

-- placeholder : PIO (Tensor xs dt)
-- placeholder {xs=xs} {dt=dt} = pure $ MkT . unsafePerformIO $ tf /. "placeholder" $. [toTfType dt, pyList xs]



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
 
