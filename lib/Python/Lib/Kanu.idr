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
     pure $ random_uniform_initializer low high seed -- (shape)


-- uniform(shape, scale=0.05, name=None)
export
uniform : Double
       -> Eff (Tensor shape dt) [PYIO]
uniform scale = random_uniform (-1 * scale) scale


-- TODO: Consider two rand strategies: running TensorFlow session with seed, vs no seed
export
glorot_uniform : Eff (Tensor shape dt) [PYIO]
glorot_uniform {shape} = uniform scale
  where
  scale : Double
  scale = case getFans shape of
               (fanIn, fanOut) => sqrt $ 6 / (fanIn + fanOut)
----------------------------------------------------------------------------------------------------
-- Misc functions
export 
record NNData (oS : Shape) (oDt : ElemType) (wTys : List (Shape, ElemType))  where
  constructor    MkNND
  nnOutput  : Tensor oS oDt
  nnWeights : Tensors wTys

-- TODO: Consider removing PYIO type from effect where there are no side effects
-- TODO: Consider- NN Open Close (s,dt) wtys
-- TODO: Consider- NN Close Open (s,dt) wtys
-- TODO: Consider- NN Open Open (s,dt) wtys
-- TODO: Or- NN () (s,dt) wtys1 wtys2
-- TODO: Or- NN (iS,iDt) (oS,oDt) wtys1 wtys2



fn : Eff () ['Tag ::: STATE Int] ['Tag ::: STATE Char]
fn = 'Tag :- putM 'a'



public export 
NNStart : (outTy : (Shape, ElemType))
       -> (wTys  : List (Shape, ElemType))
       -> Type
NNStart (oS,oDt) wTys = Eff (Tensor oS oDt) 
                            ['NN ::: STATE (Tensors wTys), PYIO] 
                            ['NN ::: STATE (NNData oS oDt wTys), PYIO]


||| End of sequential
public export
NNStop : (outTy : (Shape, ElemType))
      -> (wTys  : List (Shape, ElemType))
      -> Type
NNStop (oS,oDt) wTys = Eff (Tensor oS oDt) 
                           ['NN ::: STATE (NNData oS oDt wTys), PYIO] 
                           ['NN ::: STATE (Tensors wTys), PYIO]


public export
NNLayer : (wiTys : List (Shape, ElemType))
       -> (iTy : (Shape, ElemType)) 
       -> (woTys : List (Shape, ElemType))
       -> (oTy : (Shape, ElemType))  
       -> Type
NNLayer wiTys (iS, iDt) woTys (oS, oDt) 
  = Eff (Tensor oS oDt) ['NN ::: STATE (NNData iS iDt wiTys), PYIO] 
                        ['NN ::: STATE (NNData oS oDt woTys), PYIO]

public export
NN : (ty : (Shape, ElemType)) 
  -> (wTys : List (Shape, ElemType))
  -> Type
NN (s, dt) wTys
  = Eff (NNData s dt wTys) ['NN ::: STATE (NNData s dt wTys), PYIO] 
                           ['NN ::: STATE (NNData s dt wTys), PYIO] 



{- Did not parameterize w/ layer's weights b/c following error when running in functions
 |                   Type mismatch between
 |                           wTys
 |                   and
 |                           wTys ++ []
-}


--------------------------------------------------
-- Util functions

-- NOTE: See addWeights note
private
getNNOutput : NNLayer wTys (s,dt) wTys (s, dt)
getNNOutput = nnOutput <$> ('NN :- get)


namespace StartState
  -- NOTE: See addWeights note
  private
  setNNOutput : Tensor s dt
             -> NNStart (s, dt) wTys
  setNNOutput x = do 'NN :- updateM (\ws => MkNND x ws)
                     pure x

namespace LayerState
  -- NOTE: See addWeights note
  private
  setNNOutput : Tensor s dt
             -> NNLayer wTys (_,_) wTys (s, dt)
  setNNOutput x = do 'NN :- updateM (\(MkNND _ ws) => MkNND x ws)
                     pure x


-- NOTE: adding weights has type NNLayer. Maybe UtilFn type alias would make more sense
private
addWeights : Tensors newWTys
          -> NNLayer wTys (s,dt) (wTys ++ newWTys) (s,dt)
addWeights (MkTs newWsPy) = 
  do 'NN :- updateM (\(MkNND x (MkTs wsPy)) => MkNND x (MkTs $ wsPy ++ newWsPy))
     nnOutput <$> 'NN :- get


--------------------------------------------------
-- Layer helpers
-- Could call this start or input
-- ex: Input(batch_shape=(batchSize, timesteps, cnnWidth, featureDim))
export
start : Tensor s dt
     -> NNStart (s, dt) wTys 
start x = do 'NN :- updateM (\ws => MkNND x ws)
             pure x

export
stop : NNStop (s,dt) wTys
stop = do (MkNND t ws) <- 'NN :- get
          'NN :- putM ws
          pure t

export
end : NN (s,dt) wTys
end = 'NN :- get

--------------------------------------------------
-- Math functions

export
(*) : Tensor s dt
   -> Tensor s dt
   -> NNStart (s,dt) wTys
(*) {s} {dt} x y = setNNOutput z
  where
  z = x * y

{- TODO: Important to do this fn to help get types down
export 
concat : Tensor s1 dt
      -> Tensor s2 dt
      -> NNLayer wTys (_,_) wTys (s1,dt)
concat x y = 
-- 'sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max'.
-}


--------------------------------------------------
-- Layers

-- keras.layers.core.Dense(output_dim, init='glorot_uniform', activation='linear', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None)
export
dense : (outputDim : Nat) 
     -> NNLayer wiTys
                ([batchDim, inputDim] , dt)
                (wiTys ++ [([outputDim], dt), ([inputDim, outputDim], dt)])
                ([batchDim, outputDim], dt)
dense {dt} {batchDim} {inputDim} outputDim =  
  do x <- getNNOutput
     b <- variable !glorot_uniform
     w <- variable !glorot_uniform
     addWeights $ b #> w #> MkTs []
     setNNOutput (layer x b w)
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

private
zipWith : (fn : {s : Shape} -> {dt : ElemType}
             -> Tensor s dt 
             -> Tensor s dt 
             -> Eff () [PYIO]
          )
     -> Tensors tys
     -> Tensors tys 
     -> Eff () [PYIO]
zipWith {tys=(s,dt)::tys} fn (MkTs (t1Py::ts1Py)) (MkTs (t2Py::ts2Py)) = 
  do fn t1 t2
     zipWith fn ts1 ts2
  where 
  t1 : Tensor s dt
  t2 : Tensor s dt
  ts1 : Tensors tys
  ts2 : Tensors tys

  t1 = MkT t1Py
  t2 = MkT t2Py
  ts1 = MkTs ts1Py
  ts2 = MkTs ts2Py
zipWith _ _ _ = pure ()


export
sgd : (weights : Tensors wTys)  -- TODO: Is it better to constrain weights or something to variables only????
   -> (loss    : Tensor [] dt) -- Be careful if change this line of code
   -> Eff () [PYIO]
sgd {wTys} weights (MkT lossPy) = zipWith update weights weightGrads
  where
  loss : Tensors [([],dt)]
  loss = MkTs [lossPy]
  weightGrads : Tensors wTys
  weightGrads = gradients weights loss  -- QUESTION: Diff between passing list of tensors vs single tensor? Seems to be time vs mem tradeoff
  
  update : {s : Shape} -> {dt : ElemType} -> Tensor s dt -> Tensor s dt -> Eff () [PYIO]
  update w g = assign w newWeight
    where newWeight = w + (-1 *. (1 / 100) *. g) -- BUG: ? this operation must involve float math, but permitting any ElemType


{-
<Melvar> linman: Not exactly, but a pattern-matching bind becomes a case, 
  and for whatever reason it commits to a rigid variable as the output type 
  of that and then can’t unify that with the () actually returned from sgd.
-}


double : Tensors ds -> Tensors ds
double {ds} (MkTs ts) = MkTs $
  do ((s,dt),tp) <- zip ds ts
     let t = the (Tensor s dt) (MkT tp)
     case t + t of
       (MkT t') => pure t'



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
 
 
 

