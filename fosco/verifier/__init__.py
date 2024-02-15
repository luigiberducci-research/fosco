from typing import Type

from fosco.common.consts import VerifierType
from fosco.verifier.verifier import Verifier
from fosco.verifier.z3_verifier import VerifierZ3


def make_verifier(type: VerifierType) -> Type[Verifier]:
    if type == VerifierType.Z3:
        return VerifierZ3
    elif type == VerifierType.DREAL:
        raise NotImplementedError("Dreal not implemented")
    else:
        raise ValueError(f"Unknown verifier type {type}")