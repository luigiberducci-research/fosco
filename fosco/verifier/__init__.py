from typing import Type

from fosco.common.consts import VerifierType
from fosco.verifier.verifier import Verifier


def make_verifier(type: str | VerifierType) -> Type[Verifier]:
    if isinstance(type, str):
        type = VerifierType[type.upper()]

    if type == VerifierType.Z3:
        from fosco.verifier.z3_verifier import VerifierZ3

        return VerifierZ3
    elif type == VerifierType.DREAL:
        from fosco.verifier.dreal_verifier import VerifierDR

        return VerifierDR
    else:
        raise ValueError(f"Unknown verifier type {type}")
