module Python.Lib.TensorFlow.Matrix

import Effects
import Effect.State

import Data.List.Quantifiers
import Data.Vect.Quantifiers
import Data.Vect

-- import Control.Monad.State

import Python
import Python.PyIO
import Python.Prim
import Python.Lib.TensorFlow
import Python.Lib.Numpy
import Python.Lib.Numpy.Matrix as Np


%access public export
%default total

-- Operator Precedence
infixl 8 +.
infixl 8 -.
infixl 9 *.
-- infixl 9 /. -- TODO: Fix this
-- infixl 9 *> -- TODO: Fix this

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
  MkT : Tensor_P -> Tensor shape dtype


data Tensors : (xs : List (Shape, ElemType)) -> Type where
  MkTs : List Tensor_P -> Tensors xs

data Op : Type where
  MkOp : Op_P -> Op

data Session : Type where
  MkS : Session_P -> Session

unwrapSess : Session -> Session_P
unwrapSess s = case s of 
               (MkS s') => s'


-- private 
-- unsafeTfSession : PIO (Session_P) -> Session
-- unsafeTfSession = MkS . unsafePerformIO

-------------------------


{-
implementation Show (Tensor xs dt) where
  show (TensorCon x) = unsafePerformIO $ x /. "__str__" $. []
  show (PlaceholderCon x) = unsafePerformIO $ x /. "__str__" $. []
  show (VariableCon x) = unsafePerformIO $ x /. "__str__" $. []
-}

implementation Show (Tensor xs dt) where
  show (MkT x) = unsafePerformIO $ x /. "__str__" $. []


-------------------------
-- Helper functions (not tf api)

-- BUG: Make broadcast work
-- Broadcast example
-- [100,1,100,1] [100,1,100] => [100,100,100,100]
broadcast_dim : List Nat -> List Nat -> List Nat
broadcast_dim x y = y

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

-- Placeholders
record Placeholder (shape : Shape) (dt : ElemType) where
  constructor    MkPh
  tPlaceholder : Maybe $ Tensor shape dt
  mPlaceholder : MatrixN shape (cast dt)

infixr 5 #
||| Help convert the Placeholder record to a list. 
||| Ex: toListTuples (MkPhs x1 x2) = x1 # x2 # []
export
(#) : Placeholder xs dt -> List (Tensor_P, Arr) -> List (Tensor_P, Arr)
(#) (MkPh (Just (MkT t)) (MkM' m)) ls = (t, m) :: ls
(#) (MkPh Nothing _) ls = ls


export
interface ToPyPlaceholders a where                 -- TODO: Consider doing this instead: ToListTuples Placeholders (Tensor_P, Arr)
  toPyPlaceholders : a -> List (Tensor_P, Arr)         -- Obj Tensor_PS  cannot be a parameter of ToListTuples
                                                   --     (Implementation arguments must be type or data constructors)
implementation ToPyPlaceholders () where
  toPyPlaceholders () = []

-- Fn
private
tf : Obj TensorFlow
tf = unsafePerformIO TensorFlow.import_

private
op1 : (f : String)
  -> {auto pf : TensorFlow f = [Tensor_P] ~~> Tensor_P}
  -> Tensor ls dt -> Tensor ls dt
op1 f (MkT x) = MkT . unsafePerformIO $ tf /. f $. [x]


private
op2 : (f : String)
  -> {auto pf : TensorFlow f = [Tensor_P, Tensor_P] ~~> Tensor_P}
  -> Tensor ls dt -> Tensor ls dt -> Tensor ls dt
op2 f (MkT x) (MkT y) = MkT . unsafePerformIO $ tf /. f $. [x, y]

toTfType : ElemType -> TensorElemType_P
toTfType dt = case dt of
         Float16 => tf /. "float16"
         Float32 => tf /. "float32"
         Float64 => tf /. "float64"
         Int8    => tf /. "int8"
         Int16   => tf /. "int16"
         Int32   => tf /. "int32"
         Int64   => tf /. "int64"
         UInt8   => tf /. "uint8"
         TFBool  => tf /. "bool"
         Complex64  => tf /. "complex64"
         Complex128 => tf /. "complex128"
         QInt8   => tf /. "qint8"
         QInt32  => tf /. "qint32"
         QUInt8  => tf /. "quint8"

-- variable {dt=dt} (MkT initial_value) = MkT <$> (tf /. "Variable" $> [initial_value, toTfType dt])


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
session : Eff Session [PYIO]
session = MkS <$> (tf /. "Session" $> [])


{- Fetches
Op -> ()
Tensor -> Matrix
SparseTensor -> SparseTensorValue
?? get_tensor_handle op. The corresponding fetched value will be a numpy ndarray containing the handle of that tensor.
String -> Matrix -- A string which is the name of a tensor or operation in the graph.
-}

namespace t
  ||| tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None)
  ||| feed_dict (phs) is passed as a parameter. See runM
  ||| @ phs (feed_dict) Passed as a parameter. In the TF python API it is called feed_dict. For now it is refered to as placeholders.
  export 
  run : (ToPyPlaceholders phs) 
     => Session
     -> (fetch : Tensor xs dt)
     -> phs
     -> Eff (MatrixN xs $ cast dt) [PYIO]
  run (MkS sess) (MkT fetch) placeholders = 
    do m <- sess /. "run" $> [believe_me fetch, feed_dict placeholders]  -- NOTE: fetch should not be a placeholder
       return $ MkM' (believe_me  m)
    where
    feed_dict : (ToPyPlaceholders phs) => phs -> Dictionary_P (Tensor_P, Arr) 
    feed_dict phs = dict $ pyDictionary $ toPyPlaceholders phs

  ||| tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None)
  ||| feed_dict (phs) is passed as effect State. See run
  ||| @ phs (feed_dict) Passed as effect State. In the TF python API it is called feed_dict. For now it is refered to as placeholders.
  export 
  runM : (ToPyPlaceholders phs) 
     => Session
     -> (fetch : Tensor xs dt)
     -> Eff (MatrixN xs $ cast dt) [STATE phs, PYIO]
  runM (MkS sess) (MkT fetch) = 
    do placeholders <- get
       m <- sess /. "run" $> [believe_me fetch, feed_dict placeholders]  -- BUG: fetch should not be a placeholder
       return $ MkM' (believe_me m)
    where
    feed_dict : (ToPyPlaceholders phs) => phs -> Dictionary_P (Tensor_P, Arr) 
    feed_dict phs = dict $ pyDictionary $ toPyPlaceholders phs

namespace op
  export 
  run : Session -> Op -> Eff () [PYIO]
  run (MkS sess) (MkOp fetch) = 
    do sess /. "run" $> [believe_me fetch, feed_dict] 
       return ()
    where
    feed_dict : Dictionary_P (Tensor_P, Arr) 
    feed_dict = dict $ pyDictionary $ toPyPlaceholders ()


export 
close : Session
     -> Eff () [PYIO] 
close (MkS sess) = sess /. "close" $> []

-------------------------
-- Ops
export
initialize_all_variables : Op
initialize_all_variables = MkOp . unsafePerformIO $ tf /. "initialize_all_variables" $. []

-------------------------
-- Variable

||| tf.Variable.__init__(initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None)
||| @ initial_value can be placeholder, variable, or regular tensor
export 
variable : (initial_value : Tensor xs dt) 
        -> Eff (Tensor xs dt) [PYIO]
variable {dt=dt} (MkT initial_value) = MkT <$> (tf /. "Variable" $> [initial_value, toTfType dt])


-- NOTE: minval/maxval can be python scalar (eg Double) or a TF Tensor
-- TODO: Constrain type to floating point
-- tf.random_uniform_initializer(minval=0.0, maxval=1.0, seed=None, dtype=tf.float32)
export
random_uniform_initializer : (minval : Double)
                          -> (maxval : Double)
                          -> (seed   : Int)
                          -> Tensor shape dt
random_uniform_initializer minval maxval seed {shape=shape} {dt=dt}
  = let fn = (tf /. "random_uniform_initializer" $. [minval, maxval, seed, toTfType dt])
    in  MkT . unsafePerformIO $ fn $: [pyList shape]

------------

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

private
getUpdatedPlaceholder : Tensor xs dt 
                     -> (getPh : (phs -> Placeholder xs dt))
                     -> Eff (Placeholder xs dt) [STATE phs] [STATE phs]
getUpdatedPlaceholder t getPh = 
  do phs <- get
     pure $ case (getPh phs) of
               MkPh _ m => MkPh (Just t) m

export -- tf.placeholder(dtype, shape=None, name=None)
placeholder : (getPh : (phs -> Placeholder xs dt))
           -> (setPh : (Placeholder xs dt -> phs -> phs))
           -> Eff (Tensor xs dt) [STATE phs, PYIO] [STATE phs, PYIO]
placeholder {xs=xs} {dt=dt} getPh setPh =
  do tfPlaceholder <- pyGetTFPlaceholder
     ph <- getUpdatedPlaceholder tfPlaceholder getPh
     updateM $ setPh ph
     pure tfPlaceholder
  where
  pyGetTFPlaceholder : Eff (Tensor xs dt) [PYIO]
  pyGetTFPlaceholder = MkT <$> (tf /. "placeholder" $> [toTfType dt, pyList xs])

-------------------------
-- Training
-- gradients
--  tf.gradients(ys, xs, grad_ys=None, name='gradients', colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None)
--   ys: A Tensor or list of tensors to be differentiated.
--   xs: A Tensor or list of tensors to be used for differentiation.
--   grad_ys: Optional. A Tensor or list of tensors the same size as ys and holding the gradients computed for each y in ys.
--   name: Optional name to use for grouping all the gradient ops together. defaults to 'gradients'.
--   colocate_gradients_with_ops: If True, try colocating gradients with the corresponding op.
--   gate_gradients: If True, add a tuple around the gradients returned for an operations. This avoids some race conditions.
--   aggregation_method: Specifies the method used to combine gradient terms. Accepted values are constants defined in the class AggregationMethod.
--  A list of sum(dy/dx) for each x in xs.
||| returns list of sum(dy/dx) for each x in xs.
||| @ ys tensors to be differentiated
||| @ xs tensors used to differentiate
export
gradients : {ysT : List (Shape, ElemType)}
         -> {xsT : List (Shape, ElemType)}
         -> (ys : Tensors ysT)
         -> (xs : Tensors xsT)
         -> Tensors varTypes
gradients (MkTs ys) (MkTs xs) 
  = MkTs . unsafePerformIO $ tf /. "gradients" $. [ys, xs, Nothing, "gradients", True, False]


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
 
