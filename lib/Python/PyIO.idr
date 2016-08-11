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
data PyIO : Effect where
  DuckTypeFnCall : (f : Obj pySig)
                -> {auto pf : pySig "__call__" = Call t}
                -> (args : a)
                -> sig PyIO (retTy t args)

  PutStr'        : String
                -> sig PyIO ()

-- 2
implementation Handler PyIO PIO where
  handle () (DuckTypeFnCall f args) k = do x <- f $. args
                                           k x ()
  handle () (PutStr' s) k             = do putStrLn' s
                                           k () ()


-- 3
PYIO : EFFECT
PYIO = MkEff () PyIO


namespace PYIO
  infixl 4 $>  -- TODO: Is it better to give this the ($.) name? Or stick w/ alternate ($>)?
  ||| This is the PIO equivilent to $.
  ($>) : (f : Obj pySig)
      -> {auto pf : pySig "__call__" = Call t}
      -> (args : a)
      -> Eff (retTy t args) [PYIO]
  ($>) f a = call $ DuckTypeFnCall f a
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

putStr' : String -> Eff () [PYIO]
putStr' s = call $ PutStr' s

putStrLn' : String -> Eff () [PYIO]
putStrLn' s = putStr' (s ++ "\n")

||| Output something showable to stdout, with a trailing newline, for any FFI
||| descriptor
printLn' : Show ty => ty -> Eff () [PYIO]
printLn' a = putStrLn' (show a)

