import Data.Vect
import Python
import Python.Prim
import Python.Lib.TensorFlow
import Python.Lib.Numpy
import Python.Lib.TensorFlow.Matrix
import Python.Lib.Numpy.Matrix

%default total


empty : (Tensors [], Matrices [])
empty = ([],[])

-- Miscelaneous functions
printOp : Tensor xs dt -> PIO ()
printOp op = do rslt <- run {dtTs=v} {dtMs=cast v} session op empty 
                printLn' rslt
  where
  v : Vect 0 (Shape, ElemType)
  v = []


--------------------------------------------------
-- basics test
test_basics : PIO ()
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
test_arithmetic : PIO ()
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
test_reduce : PIO ()
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

--------------------------------------------------
-- arithmetic run
test_run : PIO ()
test_run =
  do tp <- pure $ placeholder 
     rslt <- run {dtTs=v} {dtMs=cast v} session (ones + tp) (params tp)
     printLn' rslt
  where
  m0 : MatrixN [2,2] Float32
  v : Vect 1 (Shape, ElemType)
  params : Tensor [2,2] Float32 -> (Tensors v, Matrices (cast v))

  m0 = full 3.14
  v = [([2,2], Float32)]
  params t = ([tensor_p t],[arr' m0])

{-
[[ 4.14000034  4.14000034]
 [ 4.14000034  4.14000034]]
-}


--------------------------------------------------
-- main

main : PIO ()
main = do -- test_basics
          -- test_arithmetic
          -- test_reduce
          test_run


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
