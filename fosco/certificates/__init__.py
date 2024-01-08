from typing import Type

from fosco.certificates.cbf import ControlBarrierFunction
from fosco.certificates.certificate import Certificate
from fosco.certificates.rcbf import RobustControlBarrierFunction

from fosco.common.consts import CertificateType





def make_certificate(
        certificate_type: CertificateType
) -> Type[Certificate]:
    """
    Factory function for certificates.
    """
    if certificate_type == CertificateType.CBF:
        from fosco.certificates.cbf import ControlBarrierFunction
        return ControlBarrierFunction
    elif certificate_type == CertificateType.RCBF:
        from fosco.certificates.rcbf import RobustControlBarrierFunction
        return RobustControlBarrierFunction
    else:
        raise NotImplementedError(
            f"Certificate type {certificate_type} not implemented"
        )