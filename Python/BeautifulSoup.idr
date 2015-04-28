module Python.BeautifulSoup

import Python

data Element : PySig where
  Element_string : Element "string" String

data BeautifulSoup : PySig where
  Soup_select : BeautifulSoup "select" (FMethod [String] $ Iterator (Object Element))

data Bs4 : PySig where
  Bs4_Soup : Bs4 "BeautifulSoup" (FMethod [String] $ Object BeautifulSoup)

instance Importable Bs4 where
  moduleName _ = "bs4"