module Python.PyIO

import Effects

import Python.IO
import Python.Functions

import Python.Objects
import Python.Fields
import Python.Telescope


%default total
%access public export


{-
Python.importModule
Python.Exceptions
Python.Functions
Python.Prim
-}

--------------------------------------------------
-- Setup Effects for PIO
-- 1
export
data PyIO : Effect where
  DuckTypeFnCall : (f : Obj pySig)
                -> {auto pf : pySig "__call__" = Call t}
                -> (args : a)
                -> sig PyIO (retTy t args)

-- 2
export
implementation Handler PyIO PIO where
  handle () (DuckTypeFnCall f args) k = do x <- f $. args
                                           k x ()

-- 3
export
PYIO : EFFECT
PYIO = MkEff () PyIO

--------------------------------------------------
-- Functions
export
($.) : (f : Obj pySig)
    -> {auto pf : pySig "__call__" = Call t}
    -> (args : a)
    -> Eff (retTy t args) [PYIO]
($.) f a = call $ DuckTypeFnCall f a



{-
locally : x -> (Eff t [STATE x]) -> Eff t [STATE y]
export
($:) : 
     {t : Telescope a}
  -> (f : PIO (Obj sig))
  -> {auto pf : sig "__call__" = Call t}
  -> (args : a)
  -> PIO $ retTy t args
($:) meth args = meth >>= \m => m $. args
-}

