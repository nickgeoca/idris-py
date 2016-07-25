module Python.Lib.Numpy.Matrix

import Python
import Python.Prim
import Python.Lib.Numpy

import Data.List.Quantifiers
import Data.Vect

%access public export
%default total

private
np : Obj Numpy
np = unsafePerformIO Numpy.import_

private
nda : Obj NDArrayT
nda = np /. "ndarray"


data NpElemType =
  ||| Boolean (True or False) stored as a byte
  Bool |
  ||| Default integer type (same as C long; normally either int64 or int32)
  Int | 
  ||| Identical to C int (normally int32 or int64)
  Intc |
  ||| Integer used for indexing (same as C ssize_t; normally either int32 or int64)
  Intp |
  ||| Byte (-128 to 127)
  Int8 | 
  ||| Integer (-32768 to 32767)
  Int16 | 
  ||| Integer (-2147483648 to 2147483647)
  Int32 | 
  ||| Integer (-9223372036854775808 to 9223372036854775807)
  Int64 | 
  ||| Unsigned integer (0 to 255)
  Uint8 | 
  ||| Unsigned integer (0 to 65535)
  Uint16 |
  ||| Unsigned integer (0 to 4294967295)
  Uint32 |
  ||| Unsigned integer (0 to 18446744073709551615)
  Uint64 |
  ||| Shorthand for float64.
  Float |
  ||| Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
  Float16 |
  ||| Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
  Float32 |
  ||| Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
  Float64 |
  ||| Shorthand for complex128.
  Complex |
  ||| Complex number, represented by two 32-bit floats (real and imaginary components)
  Complex64 |
  ||| Complex number, represented by two 64-bit floats (real and imaginary components)
  Complex128 

toNpType : NpElemType -> NpElemType_P
toNpType dt = case dt of
  Bool => unsafePerformIO $ np /. "bool" $. []
  Matrix.Int  => unsafePerformIO $ np /. "int" $. [] 
  Intc => unsafePerformIO $ np /. "intc" $. [] 
  Intp => unsafePerformIO $ np /. "intp" $. [] 
  Int8 => unsafePerformIO $ np /. "int8" $. [] 
  Int16 => unsafePerformIO $ np /. "int16" $. [] 
  Int32 => unsafePerformIO $ np /. "int32" $. [] 
  Int64 => unsafePerformIO $ np /. "int64" $. [] 
  Uint8 => unsafePerformIO $ np /. "uint8" $. [] 
  Uint16 => unsafePerformIO $ np /. "uint16" $. [] 
  Uint32 => unsafePerformIO $ np /. "uint32" $. [] 
  Uint64 => unsafePerformIO $ np /. "uint64" $. [] 
  Matrix.Float => unsafePerformIO $ np /. "float" $. [] 
  Float16 => unsafePerformIO $ np /. "float16" $. [] 
  Float32 => unsafePerformIO $ np /. "float32" $. [] 
  Float64 => unsafePerformIO $ np /. "float64" $. [] 
  Matrix.Complex => unsafePerformIO $ np /. "complex" $. [] 
  Complex64 => unsafePerformIO $ np /. "complex64" $. [] 
  Complex128 => unsafePerformIO $ np /. "complex128" $. [] 


export
record Matrix (rows : Nat) (cols : Nat) (dtype : NpElemType) where
  constructor MkM
  arr : Obj NDArray

public export
record MatrixN (shape : List Nat) (dtype : NpElemType) where
  constructor MkM'
  arr' : Obj NDArray


MatrixList : List (List Nat, NpElemType) -> Type 
MatrixList ts = All fn ts
  where fn : (List Nat, NpElemType) -> Type
        fn (xs,dt) = MatrixN xs dt


Matrices : {n : Nat} -> (dtMs : Vect n (List Nat, NpElemType)) -> Type
Matrices {n=n} _ = Vect n Arr

export -- Different
unsafeNp : PIO (Obj NDArray) -> Matrix r c dt
unsafeNp = MkM . unsafePerformIO

private
op : (f : String)
  -> {auto pf : NDArrayT f = [Obj NDArray, Obj NDArray] ~~> Obj NDArray}
  -> Matrix r c dt -> Matrix r c dt -> Matrix r c dt
op f (MkM x) (MkM y) = unsafeNp $ nda /. f $. [x, y]

export -- numpy.full(shape, fill_value, dtype=None, order='C')
full : Double -> MatrixN shape dt
full {shape=shape} {dt} x = MkM' . unsafePerformIO $ np /. "full" $. [pyList shape, x, toNpType dt]

{-
export
fill : {dt : NpElemType} -> a -> MatrixN shape dt
fill {shape=shape} x = unsafeNp $ np /. "tile" $. [toDyn x, shape]

export
singleton : {dt : NpElemType} -> a -> Matrix 1 1 dt
singleton {a=a} {dt=dt} x =
  unsafeNp $
    np //. FP "array" a $. [pyList [pyList [x]], dtName dt]
-}

export
dot : Matrix r c dt -> Matrix c k dt -> Matrix r k dt
dot (MkM x) (MkM y) = unsafeNp $ np /. "dot" $. [x,y]

export
transpose : Matrix r c dt -> Matrix c r dt
transpose (MkM x) = unsafeNp $ np /. "transpose" $. [x]

{-
export
array : (dt : DType a) -> Vect r (Vect c a) -> Matrix r c dt
array {a=a} dt xs = unsafeNp $ np //. FP "array" a $. [c (map c xs), dtName dt]
  where
    c : {a : Type} -> Vect n a -> Obj (PyList a)
    c xs = pyList $ toList xs
-}

export
reshape : Matrix r c dt -> {auto pf : r*c = r'*c'} -> Matrix r' c' dt
reshape {r'=r'} {c'=c'} (MkM x) =
  unsafeNp $
    np /. "reshape" $. [x, (r', c')]

export
(/) : Matrix r c dt -> Matrix r c dt -> Matrix r c dt
(/) = op "__div__"

export
minus : Matrix r c dt -> Matrix r c dt -> Matrix r c dt
minus = op "__sub__"

export
abs : Matrix r c dt -> Matrix r c dt
abs (MkM x) = unsafeNp $ np /. "abs" $. [x]

implementation Num (Matrix r c dt) where
  (+) = op "__add__"
  (*) = op "__mul__"
  -- fromInteger = Matrix.fromInteger
  fromInteger = believe_me -- BUG:

implementation Show (Matrix r c dt) where
  -- show (MkM x) = show x
  show (MkM x) = unsafePerformIO $ x /. "__str__" $. []

implementation Show (MatrixN xs dt) where
  -- show (MkM' x) = show x
  show (MkM' x) = unsafePerformIO $ x /. "__str__" $. []
