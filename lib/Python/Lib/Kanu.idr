module Python.Lib.Kanu

import Python
import Python.Prim
import Python.Lib.Numpy.Random as Random
import Python.Lib.Numpy.Matrix
import Python.Lib.TensorFlow.Matrix
import Python.Lib.TensorFlow.NN
import Python.Lib.TensorFlow

import Python.PyIO
import Effects
import Effect.State

import Data.Vect

%access public export
%default total

----------------------------------------------------------------------------------------------------
-- Initializers
-- TODO: Add length constraint to shape? Keras only uses ranks: 2,4,5.
-- NOTE: This function is partial in a sense
-- In keras initializations.py 
private 
getFans : Shape -> (Double, Double)
getFans [] = (0, 0)              -- NOTE: Invalid case
getFans [i] = (cast i, cast i)   -- NOTE: Invalid case
getFans [i, o] = (cast i, cast o)
getFans [d0,d1,d2,d3] 
  = let receptiveFieldSize = d0 * d1
        i = cast $ d2 * receptiveFieldSize
        o = cast $ d3 * receptiveFieldSize
    in  (i, o)
getFans [d0,d1,d2,d3,d4] 
  = let receptiveFieldSize = d0 * d1
        i = cast $ d3 * receptiveFieldSize
        o = cast $ d4 * receptiveFieldSize
    in  (i, o)
getFans ls = (0,0) -- NOTE: Invalid case

-- NOTE: Must be constrainted to floating types. also can use Tensor as low/high inputs. This is for when calling random_uniform_initializer 
--TODO: randint : Int -> Int
random_uniform : Double -> Double -> Eff (Tensor shape dt) [PYIO]
random_uniform low high =
  do seed <- Random.randint 100000000 -- 10e8
     return $ random_uniform_initializer low high seed -- (shape)


-- uniform(shape, scale=0.05, name=None)
export
uniform : Double
       -> Eff (Tensor shape dt) [PYIO]
uniform scale = random_uniform (-1 * scale) scale


-- TODO: Consider two rand strategies: running TensorFlow session with seed, vs no seed
export
glorot_uniform : Eff (Tensor shape dt) [PYIO]
glorot_uniform {shape=shape} = uniform scale
  where
  scale : Double
  scale = case getFans shape of
               (fanIn, fanOut) => sqrt $ 6 / (fanIn + fanOut)
----------------------------------------------------------------------------------------------------
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
-- Layers
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
export
dense : (outputDim : Nat) 
     -> NNLayer ([batchDim, inputDim] , dt)  
                ([batchDim, outputDim], dt)
dense {dt=dt} {batchDim=batchDim} {inputDim=inputDim} outputDim =  
  do x <- get
     b <- variable !glorot_uniform
     w <- variable !glorot_uniform
     putMRet $ layer x b w
  where
  layer : (x : Tensor [batchDim, inputDim] dt)
       -> (b : Tensor [outputDim] dt)  
       -> (w : Tensor [inputDim, outputDim] dt)
       ->      Tensor [batchDim, outputDim] dt
  layer x b w = softmax (b +. (x *> w)) -- TODO: Fix precedence of operators and remove parans


--------------------------------------------------
-- Loss functions
export
cross_entropy : (y : Tensor s dt) 
             -> (t : Tensor s dt) 
             -> Tensor [] dt
cross_entropy y t = reduceMeanAll $ -1 *. reduce_sum (t * log y) [1] False
  where
  reduceMeanAll : Tensor _ dt -> Tensor [] dt
  reduceMeanAll x = reduce_mean' x False

--------------------------------------------------
-- Optimizers
-- https://github.com/fchollet/keras/blob/master/keras/optimizers.py#L120
-- 'clipnorm', 'clipvalue'
-- lr=0.01, momentum=0., decay=0., nesterov=False,

assign : Tensor xs dt -> Tensor xs dt -> Eff () [PYIO]
assign = believe_me

-- {-
--  tf.gradients(loss, variables, colocate_gradients_with_ops=True)
export -- optimizer
sgd : (weights : Tensors wTys)
   -> (loss    : Tensor [] dt)
   -> Eff () [PYIO] [PYIO]
sgd (MkTs weightsPy) loss = map getNewWeight
  do (do (wPy, (ws, wdt)) <- zip weightsPy wTys
         let w = MkT wPy
             wGrad = gradients loss w
             -- position = (-1 * lr) *. wGrad -- BUG: this operation must involve float math, but permitting any ElemType
             position = (-1 / 100) *. wGrad -- BUG: this operation must involve float math, but permitting any ElemType
             wNew = w + position
         assign w wNew)
     return ()
  where


{-
  getNewWeight {s=s} {dt=dt} t = w + position
    where
    w : Tensor s dt
    w = MkT wPy
    wGrad = gradients loss w
    position = (-1 / 100) *. wGrad -- BUG: this operation must involve float math, but permitting any ElemType
    -- position = (-1 * lr) *. wGrad -- BUG: this operation must involve float math, but permitting any ElemType
-}
 
  -- lr : Tensor [] Float32
  -- lr = 0.01

-- -}

double : Tensors ds -> Tensors ds
double {ds=ds} (MkTs ts) = MkTs $
  do ((s,dt),tp) <- zip ds ts
     let t = the (Tensor s dt) (MkT tp)
     case t + t of
       (MkT t') => return t'

map : Tensors tys -> (Tensor s dt -> ) -> 

-- run
-- train
  -- loss
  -- opt


{-
model : Tensor [batchDim, 784] dt
     -> NN shape dt



transform : NN shape dt -> MatrixN shape (cast dt)
transform 
"model name"
train

model
loss
opt
 -}
 
 
 
 
