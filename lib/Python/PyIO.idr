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
  ||| Duck-typed function call. Equivilent to PIO ($.)
  ($>) : (f : Obj pySig)
      -> {auto pf : pySig "__call__" = Call t}
      -> (args : a)
      -> Eff (retTy t args) [PYIO]
  ($>) f a = call $ DuckTypeFnCall f a

  infixl 4 $<
  ||| Duck-typed function call, useful for chaining. Equivilent to PIO ($:)
  ($<) : {t : Telescope a}
      -> (f : Eff (Obj pySig) [PYIO])
      -> {auto pf : pySig "__call__" = Call t}
      -> (args : a)
      -> Eff (retTy t args) [PYIO]
  ($<) meth args = meth >>= \m => m $> args


putStr' : String -> Eff () [PYIO]
putStr' s = call $ PutStr' s

putStrLn' : String -> Eff () [PYIO]
putStrLn' s = putStr' (s ++ "\n")

||| Output something showable to stdout, without a trailing newline, for any FFI
||| descriptor
print' : Show ty => ty -> Eff () [PYIO]
print' a = putStr' (show a)

||| Output something showable to stdout, with a trailing newline, for any FFI
||| descriptor
printLn' : Show ty => ty -> Eff () [PYIO]
printLn' a = putStrLn' (show a)

