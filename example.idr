import Python
import Python.Prim
import Python.Exceptions

-- These modules contain signatures for Python libraries.
import Python.Lib.Os
import Python.Lib.Requests
import Python.Lib.BeautifulSoup

%default total

infixr 1 =<<
(=<<) : Monad m => (a -> m b) -> m a -> m b
(=<<) f x = x >>= f

-- Even though field names are strings,
-- everything is typechecked according to the signatures imported above.

partial
main : PIO ()
main = do
  reqs <- Requests.import_

  -- (/) extracts the named attribute
  -- ($) calls a function
  -- (/.) and ($.) work with pure LHS
  -- (/:) and ($:) work with monadic LHS (useful for chaining)
  --
  -- equivalent to: session = reqs.Session()
  session <- reqs /. "Session" $: []

  -- equivalent to: html = session.get("http://idris-lang.org").text
  html <- session /. "get" $: ["http://idris-lang.org"] /: "text"

  -- import Beautiful Soup
  bs4 <- BeautifulSoup.import_

  -- construct soup from HTML
  soup <- bs4 /. "BeautifulSoup" $: [html]

  -- get the iterator over <li> elements, given by CSS selector
  features <- soup /. "select" $: ["div.entry-content li"] >: Iterable (Obj Element)

  -- print all <li> elements as features
  putStrLn $ "Idris has got the following exciting features:"
  count <- iterate features 0 $ \i : Int, li => do
    -- collect : Iterator a -> PIO (List a)
    line <- map concat . collect =<< li /. "strings" >: Iterable String
    putStrLn $ show (i+1) ++ ". " ++ line
    return $ i + 1

  putStrLn $ "Total number of features: " ++ show count
  putStrLn ""

  -- test some exceptions
  os <- Os.import_
  putStrLn "And now, let's fail!"
  OK ret <- try $ os /. "mkdir" $: ["/root/hello"]
    | Catch OSError e => putStrLn ("  -> OSError as expected: " ++ show e)
    | Catch _ e => putStrLn ("  -> some other error: " ++ show e)
  putStrLn $ "Your root could probably use some security lessons!"
