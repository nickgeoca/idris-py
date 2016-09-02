import Effects
import Effect.State

import Data.Vect
import Python.IO
import Python.PyIO
import Python
import Python.Lib.TensorFlow
import Python.Lib.TensorFlow.Matrix
import Python.Lib.Numpy
import Python.Lib.Numpy.Matrix
import Python.Lib.Kanu

%default total

--------------------------------------------------
-- Miscelaneous functions
printOpParam : ToPyPlaceholders phs => Tensor xs dt -> Eff () [STATE phs, PYIO] 
printOpParam t = do sess <- session
                    op.run sess initialize_all_variables
                    printLn' !(runM sess t)
  

-- NOTE: Ran this w/ placeholder code and threw runtime error
printOp : Tensor xs dt -> Eff () [PYIO] 
printOp op = do sess <- session
                run sess initialize_all_variables
                printLn' !(run sess op ())

--------------------------------------------------
-- Model
model : Tensor [batchDim, 784] dt
     -> NN     [batchDim, 10]  dt
model x = 
  do start x
     dense 10
     y1 <- end

     start x
     dense 10
     y2 <- end

     y1 * y2

--------------------------------------------------
-- Placeholders
record Parameters where
  constructor MkParams
  X : Placeholder [2, 784] Float32

implementation ToPyPlaceholders Parameters where
  toPyPlaceholders (MkParams x1) = x1 # []

--------------------------------------------------
-- basics test
main : PIO ()
main = runInit [phs, (), ()] (do x <- placeholder X set_X 
                                 y <- model x
                                 printOpParam y)
  where
  phs : Parameters
  phs = MkParams (MkPh Nothing (full 3.14))





{-
-- Model
model : Tensor [batchDim, 784] dt
    -> PIO $ Tensor [batchDim, 10] dt
model x = runInit [(), ()] $ 
  do start x
     dense 10
     y1 <- end

     start x
     dense 10
     y2 <- end

     y1 * y2
-}
