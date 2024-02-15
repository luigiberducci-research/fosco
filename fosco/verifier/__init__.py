from typing import Type

from fosco.common.consts import VerifierType
from fosco.verifier.verifier import Verifier


def make_verifier(type: VerifierType) -> Type[Verifier]:
    if type == VerifierType.Z3:
        from fosco.verifier.z3_verifier import VerifierZ3
        return VerifierZ3
    elif type == VerifierType.DREAL:
        from fosco.verifier.dreal_verifier import VerifierDR
        return VerifierDR
    else:
        raise ValueError(f"Unknown verifier type {type}")