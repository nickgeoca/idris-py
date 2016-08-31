module Python.Lib.Numpy.Random

import Effects

import Python
import Python.Prim
import Python.Lib.Numpy
import Python.PyIO

import Data.List.Quantifiers
import Data.Vect

%access public export
%default total


npRand : Obj NumpyRandom
npRand = unsafePerformIO $ importModule "numpy.random"

export
randint : Int -> Eff Int [PYIO]
randint high = return $ unsafePerformIO $ npRand /. "randint" $. [high]
