from enum import Enum, auto

import torch


class DomainName(Enum):
    XD = "lie"
    XU = "unsafe"
    XI = "init"
    UD = "input"
    ZD = "uncertainty"


class CertificateType(Enum):
    CBF = "cbf"
    RCBF = "rcbf"

    @classmethod
    def get_certificate_sets(
        cls, certificate_type
    ) -> tuple[list[DomainName], list[DomainName]]:
        dn = DomainName
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
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class ActivationType(Enum):
    IDENTITY = "identity"
    RELU = "relu"
    LINEAR = "linear"
    SQUARE = "square"
    POLY_2 = "poly_2"
    RELU_SQUARE = "relu_square"
    REQU = "requ"
    POLY_3 = "poly_3"
    POLY_4 = "poly_4"
    POLY_5 = "poly_5"
    POLY_6 = "poly_6"
    POLY_7 = "poly_7"
    POLY_8 = "poly_8"
    EVEN_POLY_4 = "even_poly_4"
    EVEN_POLY_6 = "even_poly_6"
    EVEN_POLY_8 = "even_poly_8"
    EVEN_POLY_10 = "even_poly_10"
    RATIONAL = "rational"
    # dReal only from here
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTPLUS = "softplus"
    COSH = "cosh"


class VerifierType(Enum):
    Z3 = "z3"
    DREAL = "dreal"


class LossReLUType(Enum):
    RELU = "relu"
    SOFTPLUS = "softplus"

    def __call__(self, x):
        if self == LossReLUType.RELU:
            return torch.relu(x)
        elif self == LossReLUType.SOFTPLUS:
            return torch.nn.Softplus()(x)
        else:
            raise NotImplementedError(f"LossReLUType {self} not implemented")
