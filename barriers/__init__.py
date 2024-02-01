from barriers.single_integrator import SingleIntegratorCompensatorAdditiveBoundedUncertainty
from barriers.single_integrator import SingleIntegratorTunableCompensatorAdditiveBoundedUncertainty
from models.torchsym import TorchSymDiffModel
from systems import ControlAffineDynamics


def make_barrier(system: ControlAffineDynamics, uncertainty: str, **kwargs) -> dict[str, TorchSymDiffModel]:
    """
    TODO: we need to use uncertainty_type here to choose corresponding compensator
    """
    if system.id == "single_integrator":
        from barriers.single_integrator import SingleIntegratorCBF
        barrier = SingleIntegratorCBF(system=system)
        if uncertainty == "additive_bounded":
            compensator = SingleIntegratorCompensatorAdditiveBoundedUncertainty(h=barrier, system=system)
        if uncertainty == "tunable_additive_bounded":
            compensator = SingleIntegratorTunableCompensatorAdditiveBoundedUncertainty(h=barrier, system=system)
        return {"barrier": barrier, "compensator": compensator}
    else:
        raise NotImplementedError(f"barrier {id} not implemented")