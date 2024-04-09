from typing import Type

from fosco.certificates.certificate import Certificate, TrainableCertificate
from fosco.common.consts import CertificateType


def make_certificate(certificate_type: str | CertificateType) -> Type[TrainableCertificate]:
    """
    Factory function for certificates.
    """
    if isinstance(certificate_type, str):
        certificate_type = CertificateType[certificate_type.upper()]

    if certificate_type == CertificateType.CBF:
        from fosco.certificates.cbf import TrainableCBF
        return TrainableCBF
    elif certificate_type == CertificateType.RCBF:
        from fosco.certificates.rcbf import TrainableRCBF
        return TrainableRCBF
    else:
        raise NotImplementedError(
            f"Certificate type {certificate_type} not implemented"
        )
