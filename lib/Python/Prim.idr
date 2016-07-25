module Python.Prim

import Python
import Python.Telescope
import Python.Fields
import Python.Functions

%access public export
%default total

||| The actual state of iteration.
Iterator : Type -> Signature
Iterator a f = case f of
  "next" => [] ~~> a
  _ => Object f

||| Something that can produce an iterator.
Iterable : Type -> Signature
Iterable a f = case f of
  "__iter__" => [] ~~> Obj (Iterator a)
  _ => Object f

||| Python string as object.
PyString : Signature
PyString f = case f of
  "join" => [Obj $ Iterable String] ~~> String
  _ => Iterable Char f

||| Python list as object.
PyList : Type -> Signature
PyList a = Iterable a  -- nothing else yet

||| Python dictionary
{-
Dictionary_PS : {key : Type} -> {value : Type} -> (key, value) -> Signature
Dictionary_PS (k,v) f = case f of
  _ => Object f
  
Dictionary_P : (Type, Type) -> Type
Dictionary_P (k,v) = Obj $ Dictionary_PS (k,v) 
-- -}

-- TODO: Enforce the 'Type' parameter to be a tuple
Dictionary_PS : Type -> Signature
Dictionary_PS a f = case f of
  "__str__" => [] ~~> String
  _ => Object f
  
Dictionary_P : Type -> Type
Dictionary_P a = Obj $ Dictionary_PS a

public export
record Dictionary a where
  constructor MkDict
  dict : Obj $ Dictionary_PS a


implementation Show (Dictionary a) where
  show (MkDict d) = unsafePerformIO $ d /. "__str__" $. []



{-
implementation Show a => Show (Obj $ Dictionary_PS a) where
  show = \x=>"abc" -- believe_me  -- unsafePerformIO $ d /. "__str__" $. []
-- -}

||| Primitives promotable to objects.
data PythonPrim : Type -> Signature -> Type where
  PPString : PythonPrim String PyString

namespace Builtins
  Builtins : Signature
  Builtins f = case f of
    "list" => fun $
      forall a : Type .
        pi xs : (List a) .
          Return $ Obj (PyList a)
    "dict" => fun $
      forall a : Type .
        pi xs : (List a) .
          Return $ Dictionary_P a
    _ => Module f

  builtins : Obj Builtins
  builtins = getGlobal "__builtins__"


pyList : List a -> Obj $ PyList a
pyList {a=a} xs = unsafePerformIO (builtins /. "list" $. [Erase a, xs])

pyDictionary : List (k,v) -> Dictionary (k,v)
pyDictionary {k=k} {v=v} xs = MkDict $ unsafePerformIO (builtins /. "dict" $. [Erase (k,v), xs])

{-
    -- "dict" => [Obj $ PyList (a,b)] ~~> Dictionary_P
pyDictionary : List (k,v) -> Dictionary_P (k,v)
pyDictionary {k=k} {v=v} xs = unsafePerformIO (builtins /. "dict" $. [Erase (k,v), xs])
pyDictionary : List (k,v) -> Dictionary_P (k,v)
pyDictionary {k=k} {v=v} xs = unsafePerformIO (builtins /. "dict" $. [Erase (k,v), xs])
pyDictionary : List kv -> Dictionary_P kv
pyDictionary {kv=kv} xs = unsafePerformIO (builtins /. "dict" $. [pyList xs])
-}

||| Promote a primitive to an object. Note that this is a no-oop,
||| all primitives already are objects in Python.
obj : (x : a) -> {auto pf : PythonPrim a sig} -> Obj sig
obj x = believe_me x

||| Get the next value from an iterator.
next : Obj (Iterator a) -> PIO (Maybe a)
next {a = a} it = do
  OK x <- try (it /. "next" $. [])
    | Except StopIteration e => return Nothing
    | Except _ e => raise e
  return $ Just x

||| A left-fold over an iterable object, implemented in Idris.
||| This is not very efficient (TCO does not kick in here)
||| but it's a demonstration we can do it easily.
|||
||| See also `foreach`, which is a native Python for-loop.
partial
iterate : (o : Obj sig)
  -> (st : b)
  -> (f : b -> a -> PIO b)
  -> {auto pf : sig "__iter__" = [] ~~> Obj (Iterator a)}
  -> PIO b
iterate iterable st f = do
    iterator <- iterable /. "__iter__" $. []
    iter iterator st f
  where
    partial
    iter : Obj (Iterator a) -> (st : b) -> (f : b -> a -> PIO b) -> PIO b
    iter it st f = do
      Just x <- next it | Nothing => return st
      st' <- f st x
      iter it st' f

||| A left-fold over an iterable object, implemented as a for-loop in Python.
|||
||| @ o The iterable object.
||| @ st Initial state.
||| @ f  PIO action called for every element, transforms the state.
partial 
foreach :
  (o : Obj sig)
  -> (st : b)
  -> (f : b -> a -> PIO b)
  -> {auto pf : sig "__iter__" = [] ~~> Obj (Iterator a)}
  -> PIO b
foreach {a=a} {b=b} {sig=sig} iterable st f = do
  iterator <- iterable /. "__iter__" $. []
  unRaw <$>
    foreign FFI_Py "_idris_foreach"
      (Obj sig -> Raw b -> Raw (b -> a -> PIO b) -> PIO (Raw b))
      iterable
      (MkRaw st)
      (MkRaw f)

||| Collect all elements of an iterator into a list.
partial
collect : (it : Obj sig) -> {auto pf : sig "__iter__" = [] ~~> Obj (Iterator a)} -> PIO (List a)
collect it = reverse <$> foreach it List.Nil (\xs, x => return (x :: xs))


-- TODO: Is using constraint (Show a) correct? Some things in Idris that are part of show may not translate to python



           
--implementation Show Tensor_P where
--  show x = unsafePerformIO $ x /. "__str__" $. []


{-
implementation Show a => Show (Dictionary_P a) where
  show d = show d
-}
--     Can't find implementation for Show (Obj (Dictionary_PS a))

{-
When checking right hand side 
of (Dictionary_P a) implementation of Show, 
method showPrec with expected type
   Dictionary_P a -> String
     
Can't find implementation for Show (Obj (Dictionary_PS a))
-- -}
