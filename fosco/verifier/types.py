import dreal
import z3

Z3SYMBOL = z3.ArithRef
DRSYMBOL = dreal.Expression

SYMBOL = Z3SYMBOL | DRSYMBOL