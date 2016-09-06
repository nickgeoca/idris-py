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
printOpParam : ToPyPlaceholders phs => Session -> Tensor xs dt -> Eff () [STATE phs, PYIO] 
printOpParam sess t = printLn' !(runM sess t)
  

-- NOTE: Ran this w/ placeholder code and threw runtime error
printOp : Session -> Tensor xs dt -> Eff () [PYIO] 
printOp sess op = printLn' !(run sess op ())

--------------------------------------------------
-- Model
{-
TODO
get list of vars from this
get model fn from this
run contain omdel state in this function
-}

-- {-
model : Tensor [batchDim, 784] dt
     -> Eff         (NNData [batchDim, 10] dt [([10], dt),
                                               ([784, 10], dt),
                                               ([10], dt),
                                               ([784, 10], dt)]) 
            [STATE $ Tensors [], PYIO] 
            [STATE $ NNData [batchDim, 10] dt [([10], dt),
                                              ([784, 10], dt),
                                              ([10], dt),
                                              ([784, 10], dt)], PYIO]
model x = 
  do start x
     dense 10
     y1 <- stop

     start x
     dense 10
     y2 <- stop

     y1 * y2
     end
-- -}

{-
modelx : Tensor s dt
  -> Eff () [STATE $ Tensors ws, PYIO] [STATE $ NNData s dt ws, PYIO]
modelx x = do start x
              pure ()
              
-- -}

-- Documentation
-- Idris comm tutorial
{-
C-c C-s  -- Do on decleration to get patterns
C-c C-c  -- Case split on pattern variable
C-c C-a  -- Try to solve hole
C-c C-z  -- Go to repl
-}
  
--------------------------------------------------
-- Placeholders
record Parameters where
  constructor MkParams
  X : Placeholder [2, 784] Float32

implementation ToPyPlaceholders Parameters where
  toPyPlaceholders (MkParams x1) = x1 # []

--------------------------------------------------
-- basics test

-- {-
main : PIO ()
main = runInit [phs, (MkTs []), ()] (
  do sess <- initSess
     x <- placeholder X set_X 
     -- NOTE: Probably runnig slow so need to tag to improve speed
     (MkNND y ws) <- model x
     updateM (\(MkNND _ _) => ())
     printOpParam y
     sgd ws ones -- (cross_entropy y (constant(full 0.5)))
     printOpParam y
  )
  where
  phs : Parameters
  phs = MkParams (MkPh Nothing (full 3.14))
  initSess : Eff Session [PYIO]
  initSess = do sess <- session
                op.run sess initialize_all_variables
                return sess
-- -}

{-
NN shape dt = Eff (Tensor shape dt) [STATE (), PYIO] [STATE $ Tensor shape dt, PYIO]

runModel
 run model placeholders

train : Optimizer opt 
     => Eff () [PYIO]
 runModel
 opt weights loss

data Model : Type where
  model : NN s dt
  opt   : (weights : Tensors wTys) 
       -> (loss    : Tensor [] dtl)
       -> Eff () [PYIO]
  
interface Model s dt where
  run : matricies -> Tensor s dt
  train : matricies -> Eff () [PYIO]

   
sgd : (weights : Tensors wTys)
   -> (loss    : Tensor [] dt) -- Be careful if change this line of code
-}

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
