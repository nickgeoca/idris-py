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
import Python.Lib.Numpy.Matrix as Np

%default total

--------------------------------------------------
-- Miscelaneous functions
printOpParam : ToPyPlaceholders phs => Session -> Tensor s dt -> Eff () ['Phs ::: STATE phs, PYIO] 
printOpParam {s} {dt} sess t = 
  do result <- runM sess ts
     printLn' result
  where
  ts : Tensors [(s,dt)]
  ts = t #> (MkTs [])
  
{-
-- NOTE: Ran this w/ placeholder code and threw runtime error
printOp : Session -> Tensor s dt -> Eff () [PYIO] 
printOp {s} {dt} sess t = 
  do ms <- t.run sess ts ()
     case ms of
       (MkMs m :: _) => printLn' (the (MatrixN s (cast dt)) $MkM' m)
       _ => pure ()
  where
  ts : Tensors [(s,dt)]
  ts = t #> (MkTs [])
-}

-- NOTE: Ran this w/ placeholder code and threw runtime error
runOps : Session -> List Op -> Eff () [PYIO] 
runOps sess ops = op.run sess ops

--------------------------------------------------
-- Model

weights : ElemType -> List (Shape, ElemType)
weights dt = [([10], dt),([784, 10], dt),([10], dt),([784, 10], dt)]


model : Tensor [batchDim, 784] dt
     -> Eff (NNData [batchDim, 10] dt (weights dt)) 
            ['NN ::: STATE (Tensors []), PYIO] 
            ['NN ::: STATE (), PYIO]
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

mainEff : TransEff.Eff () ['NN ::: STATE (Tensors []), 'Phs ::: STATE Parameters, PYIO]
                          ['NN ::: STATE ()          , 'Phs ::: STATE ()        , PYIO]
mainEff =        
  do x <- placeholder X set_X 
     (MkNND y ws) <- model x

     sess <- initSess -- NOTE: This line must go below the model

     let loss = cross_entropy y half
     printOpParam sess y
     ops <- sgd ws loss
     runOps sess ops
     printOpParam sess y

     'Phs :- putM ()
     pure ()
  where
  initSess : Eff Session [PYIO]
  initSess = do sess <- session
                op.run sess [initialize_all_variables]
                return sess
  half : Tensor [2,10] Float32
  half = constant $ the (MatrixN [2,10] Np.Float32) $ full 0.5

main : PIO ()
main =  
  do runInit ['NN := (the (Tensors []) (MkTs [])), 'Phs := phs, ()] mainEff
     pure ()
  where
  phs : Parameters
  phs = MkParams (MkPh Nothing (full 3.14))

-- -}

{-
mainEff : Eff () [PYIO]
mainEff = printOp !session $ !(variable (the (Tensor [2,2] Float32) !glorot_uniform)) 


main : PIO ()
main =  
  do runInit [()] mainEff
     pure ()
  where
  phs : Parameters
  phs = MkParams (MkPh Nothing (full 3.14))
-- -}
