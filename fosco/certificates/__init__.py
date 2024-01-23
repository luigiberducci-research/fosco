from typing import Type

from fosco.certificates.certificate import Certificate
from fosco.common.consts import CertificateType


def make_certificate(certificate_type: CertificateType) -> Type[Certificate]:
    """
    Factory function for certificates.
    """
    if certificate_type == CertificateType.CBF:
        from fosco.certificates.cbf import TrainableCBF

        return TrainableCBF
    elif certificate_type == CertificateType.RCBF:
        from fosco.certificates.rcbf import TrainableRCBF

        return TrainableRCBF
    elif certificate_type == CertificateType.CBFgt:
        raise NotImplementedError()
    else:
        raise NotImplementedError(
            f"Certificate type {certificate_type} not implemented"
        )
