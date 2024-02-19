from typing import Callable

from fosco.verifier.types import SYMBOL, Z3SYMBOL, DRSYMBOL


def get_solver_fns(x: list[SYMBOL]) -> dict[str, Callable]:
    """
    Given a list of symbolic variables, it returns the functions supported for the type of variables.

    Args:
        x: list of symbolic variables

    Returns:
        dict: dictionary of functions supported for the type of variables, as {function_name: function}
    """
    if all([isinstance(xi, Z3SYMBOL) for xi in x]):
        from fosco.verifier.z3_verifier import VerifierZ3
        return VerifierZ3.solver_fncts()
    elif all([isinstance(xi, DRSYMBOL) for xi in x]):
        from fosco.verifier.dreal_verifier import VerifierDR
        return VerifierDR.solver_fncts()
    else:
        raise NotImplementedError(f"Unsupported type {type(x)}")
