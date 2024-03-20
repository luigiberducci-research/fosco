import dreal
import z3
import sympy as sp

Z3SYMBOL = z3.ArithRef
DRSYMBOL = dreal.Variable | dreal.Expression
SPSYMBOL = sp.Expr

SYMBOL = Z3SYMBOL | DRSYMBOL | SPSYMBOL
