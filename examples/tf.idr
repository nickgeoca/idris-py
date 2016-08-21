import Effects
import Effect.State

import Data.Vect

import Python
import Python.Prim
import Python.IO
import Python.PyIO
import Python.Lib.TensorFlow
import Python.Lib.Numpy
import Python.Lib.TensorFlow.Matrix
import Python.Lib.Numpy.Matrix

%default total


--------------------------------------------------
-- Placeholders

-- Minimal code to make placeholders
-- 1. Create: record Placeholders where
-- 2. Create: implementation ToPyPlaceholders Placeholders where

record Placeholders where
  constructor    MkPhs
  X1 : Placeholder [2,2] Float32
  X2 : Placeholder [] Float32

implementation ToPyPlaceholders Placeholders where
  toPyPlaceholders (MkPhs x1 x2) = x1 # x2 # []

--------------------------------------------------
-- Miscelaneous functions
empty : (Tensors [], Matrices [])
empty = ([],[])


printOp : ToPyPlaceholders phs => Tensor xs dt -> Eff () [STATE phs, PYIO] 
printOp op = do rslt <- run session op
                printLn' rslt

{-
--------------------------------------------------
-- basics test
test_basics : Eff () [PYIO]
test_basics = do printLn' x
                 printOp x
  where
    x : Tensor [3,4] Float32
    x = ones
{-
Tensor("ones:0", shape=(3, 4), dtype=float32)
[[ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]]
-}


--------------------------------------------------
-- arithmetic test
test_arithmetic : Eff () [PYIO]
test_arithmetic =
  do printOp $ z
     printOp $ x + x
  where
    x : Tensor [3,4] Float32
    y : Tensor [4,3] Float32
    z : Tensor [3,3] Float32

    x = ones
    y = ones
    z = matmul x y
{-
[[ 4.  4.  4.]
 [ 4.  4.  4.]
 [ 4.  4.  4.]]
[[ 2.  2.  2.  2.]
 [ 2.  2.  2.  2.]
 [ 2.  2.  2.  2.]]
-}


--------------------------------------------------
-- reduce_<fns> test
test_reduce : Eff () [PYIO]
test_reduce = 
  do printOp $ reduce_mean x [0,1] True
     printOp $ reduce_mean x [0] False
     printOp $ reduce_mean' x True
     printOp $ reduce_mean' x False
     printOp $ reduce_sum x [0,1] True
     printOp $ reduce_sum x [0] False
     printOp $ reduce_sum' x False
     printOp $ reduce_sum' x True
  where 
     x : Tensor [3,4] Float32
     x = ones
{-
[[ 1.]]
[ 1.  1.  1.  1.]
[[ 1.]]
1.0
[[ 12.]]
[ 3.  3.  3.  3.]
12.0
[[ 12.]]
-}

-}

--------------------------------------------------
-- arithmetic run
test_run : Eff () [STATE Placeholders, PYIO] --  [STATE phs, PYIO]
test_run =
  do x1 <- placeholder X1 set_X1
     x2 <- placeholder X2 set_X2
     printOp (x2 +. (ones + x1))

{-
[[ 5.14000034  5.14000034]
 [ 5.14000034  5.14000034]]
-}


--------------------------------------------------
-- main

main : PIO ()
main = runInit [phs, ()] (do test_run)
  where
  phs : Placeholders
  phs = MkPhs (MkPh Nothing $ full 3.14)
              (MkPh Nothing $ full 1)


{-
main = run (do test_basics
               test_arithmetic
               test_reduce
               test_run)

-}

--------------------------------------------------
{-
fn2 : PIO (MatrixN [3,3] DDouble)
fn2 = run session z

exports : FFI_Export FFI_Py "tf.py" []
exports =
    Fun greet2 "greet2" $
--    Fun fn "fn" $
    End
  where
    greet2 : String -> PIO String
    greet2 name = return $ "Hello " ++ name ++ "!"
    greet : String -> PIO ()
    greet name = putStrLn' $ "Hello " ++ name ++ "!"
-}
