import sympy as sp

SYMPY_FNS = {
    "And": sp.And,
    "Or": sp.Or,
    "If": None,  # todo: do we really need If? Or max is enough?
    "Not": sp.Not,
    "False": False,
    "True": True,
    "Exists": None,
    "ForAll": None,
    "Substitute": sp.substitution,
    "Check": None,
    "RealVal": lambda x: float(x),
    "Sqrt": sp.sqrt,
}
