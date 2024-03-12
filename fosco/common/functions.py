import numpy as np

from fosco.common.consts import VerifierType
from fosco.verifier.dreal_verifier import VerifierDR
from fosco.verifier.z3_verifier import VerifierZ3

NP_FUNCTIONS = {
    # symbolic functions
    "Exists": None,
    "ForAll": None,
    "Substitute": None,
    "Check": None,
    # numerical functions
    "RealVal": lambda real: real,
    "Sqrt": np.sqrt,
    "Sin": np.sin,
    "Cos": np.cos,
    "Exp": np.exp,
    "And": np.logical_and,
    "Or": np.logical_or,
    "If": np.where,
    "Not": np.logical_not,
    "False": False,
    "True": True,
}


FUNCTIONS = {
    "numerical": NP_FUNCTIONS,
    VerifierType.Z3: VerifierZ3.solver_fncts(),
    VerifierType.DREAL: VerifierDR.solver_fncts(),
}
