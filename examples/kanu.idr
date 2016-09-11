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
printOpParam : ToPyPlaceholders phs => Session -> Tensor xs dt -> Eff () ['Phs ::: (STATE phs), PYIO] 
printOpParam sess t = printLn' !(runM sess t)
  

-- NOTE: Ran this w/ placeholder code and threw runtime error
printOp : Session -> Tensor xs dt -> Eff () [PYIO] 
printOp sess op = printLn' !(run sess op ())

--------------------------------------------------
-- Model

weights : ElemType -> List (Shape, ElemType)
weights dt = [([10], dt),([784, 10], dt),([10], dt),([784, 10], dt)]


model : Tensor [batchDim, 784] dt
     -> Eff (NNData [batchDim, 10] dt (weights dt)) 
            ['NN ::: STATE (Tensors []), PYIO] 
            ['NN ::: STATE (NNData [batchDim, 10] dt (weights dt)), PYIO]
model x = 
  do start x
     dense 10
     y1 <- stop

     start x
     dense 10
     y2 <- stop

     y1 * y2
     end


modelx : Tensor s dt
  -> TransEff.Eff () ['NN ::: STATE (Tensors ws)    , PYIO] 
            ['NN ::: STATE (NNData s dt ws), PYIO]
modelx x = do start x
              pure ()
  
--------------------------------------------------
-- Placeholders
record Parameters where
  constructor MkParams
  X : Placeholder [2, 784] Float32

implementation ToPyPlaceholders Parameters where
  toPyPlaceholders (MkParams x1) = x1 # []

--------------------------------------------------
-- Main


main : PIO ()
main =  
  do runInit ['NN := (the (Tensors []) (MkTs [])), 'Phs := phs, ()]( -- ['Phs := phs, 'NN := (MkTs []), ()] (
       do sess <- initSess
          x <- placeholder X set_X 
          (MkNND y ws) <- model x
          -- printOpParam y
          -- sgd ws (believe_me ones) -- (cross_entropy y (constant(full 0.5)))  
          -- printOpParam y
          pure ()
       )
     pure ()
  where
  phs : Parameters
  phs = MkParams (MkPh Nothing (full 3.14))
  initSess : Eff Session [PYIO]
  initSess = do sess <- session
                op.run sess initialize_all_variables
                return sess

