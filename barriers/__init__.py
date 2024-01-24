from barriers.single_integrator import SingleIntegratorCompensatorAdditiveBoundedUncertainty
from models.torchsym import TorchSymModel
from systems import ControlAffineDynamics


def make_barrier(system: ControlAffineDynamics, **kwargs) -> dict[str, TorchSymModel]:
    if system.id == "single_integrator":
        from barriers.single_integrator import SingleIntegratorCBF
        barrier = SingleIntegratorCBF(system=system)
        compensator = SingleIntegratorCompensatorAdditiveBoundedUncertainty(h=barrier, system=system)
        return {"barrier": barrier, "compensator": compensator}
    else:
        raise NotImplementedError(f"barrier {id} not implemented")