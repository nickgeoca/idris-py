module Python.Lib.TensorFlow.NN

import Python
import Python.Prim
import Python.Lib.TensorFlow.Matrix
import Python.Lib.TensorFlow

import Data.List.Quantifiers
import Data.Vect

%access public export
%default total


private
nn : Obj TensorFlowNN
nn = unsafePerformIO $ importModule "tensorflow.python.ops.nn"

-- TODO: Constrain to: Float16, Float32, Float64
export -- tf.nn.softmax(logits, name=None)
softmax : Tensor [b, x] dt -> Tensor [b, x] dt
softmax (MkT x) = MkT . unsafePerformIO $ nn /. "softmax" $. [x]

export
sigmoid : Tensor s dt -> Tensor s dt
sigmoid (MkT x) = MkT . unsafePerformIO $ nn /. "sigmoid" $. [x]
