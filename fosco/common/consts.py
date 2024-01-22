from enum import Enum, auto

import torch


class DomainNames(Enum):
    XD = "lie"
    XU = "unsafe"
    XI = "init"
    UD = "input"
    ZD = "uncertainty"


class CertificateType(Enum):
    CBF = auto()
    RCBF = auto()

    @classmethod
    def get_certificate_sets(
        cls, certificate_type
    ) -> tuple[list[DomainNames], list[DomainNames]]:
        dn = DomainNames
        if certificate_type == CertificateType.CBF:
            domains = [dn.XD, dn.UD, dn.XI, dn.XU]
            data = [dn.XD, dn.UD, dn.XI, dn.XU]
        elif certificate_type == CertificateType.RCBF:
            domains = [dn.XD, dn.UD, dn.ZD, dn.XI, dn.XU]
            data = [dn.XD, dn.UD, dn.ZD, dn.XI, dn.XU]
        else:
            raise NotImplementedError(
                f"Certificate type {certificate_type} not implemented"
            )
        return domains, data


class TimeDomain(Enum):
    CONTINUOUS = auto()
    DISCRETE = auto()


class ActivationType(Enum):
    IDENTITY = auto()
    RELU = auto()
    LINEAR = auto()
    SQUARE = auto()
    POLY_2 = auto()
    RELU_SQUARE = auto()
    REQU = auto()
    POLY_3 = auto()
    POLY_4 = auto()
    POLY_5 = auto()
    POLY_6 = auto()
    POLY_7 = auto()
    POLY_8 = auto()
    EVEN_POLY_4 = auto()
    EVEN_POLY_6 = auto()
    EVEN_POLY_8 = auto()
    EVEN_POLY_10 = auto()
    RATIONAL = auto()


class VerifierType(Enum):
    Z3 = auto()


class LossReLUType(Enum):
    RELU = auto()
    SOFTPLUS = auto()

    def __call__(self, x):
        if self == LossReLUType.RELU:
            return torch.relu(x)
        elif self == LossReLUType.SOFTPLUS:
            return torch.nn.functional.softplus(x)
        else:
            raise NotImplementedError(f"LossReLUType {self} not implemented")
