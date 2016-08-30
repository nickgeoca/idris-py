module Python.Lib.Kanu

import Python
import Python.Prim
-- import Python.Lib.TensorFlow
-- import Python.Lib.Numpy as Np
import Python.Lib.Numpy.Matrix
import Python.Lib.TensorFlow.Matrix
import Python.Lib.TensorFlow.NN
-- import Python.Lib.TensorFlow

import Python.PyIO
import Effects
import Effect.State

import Data.Vect

%access public export
%default total


-- Misc functions
putMRet : a -> Eff a [STATE _] [STATE a]
putMRet z = do putM z
               return z

NN : (os  : Shape)
  -> (odt : ElemType)
  -> Type
NN shape dt = Eff (Tensor shape dt) [STATE (), PYIO] [STATE $ Tensor shape dt, PYIO]

NNLayer : (i : (Shape, ElemType)) 
       -> (o : (Shape, ElemType))  
       -> Type
NNLayer (iShape, iDt) (oShape, oDt) 
  = Eff (Tensor oShape oDt) [STATE $ Tensor iShape iDt, PYIO] [STATE $ Tensor oShape oDt, PYIO]
--       -> Eff layer [STATE layer, PYIO]
-- ex: Input(batch_shape=(batchSize, timesteps, cnnWidth, featureDim))

export
(*) : Tensor xs dt
   -> Tensor xs dt
   -> Eff (Tensor xs dt) [STATE a, PYIO] [STATE $ Tensor xs dt, PYIO]
(*) x y = putMRet $ mul x y

--------------------------------------------------
-- ex: Input(batch_shape=(batchSize, timesteps, cnnWidth, featureDim))
input : Tensor shape dt
     -> NNLayer (_, _) (shape, dt)
input t = putMRet t
start : Tensor shape dt
     -> Eff (Tensor shape dt) [STATE a, PYIO] [STATE $ Tensor shape dt, PYIO]
start t = putMRet t
end : Eff (Tensor shape dt) [STATE $ Tensor shape dt, PYIO] [STATE (), PYIO] 
end = do t <- get
         putM ()
         return t


-- keras.layers.core.Dense(output_dim, init='glorot_uniform', activation='linear', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None)
-- {-
dense : (outputDim : Nat) 
     -> NNLayer ([batchDim, inputDim] , dt)  
                ([batchDim, outputDim], dt)
dense {dt=dt} {batchDim=batchDim} {inputDim=inputDim} outputDim =  
  do x <- get
     b <- variable $ zeros
     w <- variable $ zeros
     putMRet $ layer x b w
  where
  layer : (x : Tensor [batchDim, inputDim] dt)
       -> (b : Tensor [outputDim] dt)  
       -> (w : Tensor [inputDim, outputDim] dt)
       ->      Tensor [batchDim, outputDim] dt
  layer x b w = softmax $ b +. (x *> w) 
