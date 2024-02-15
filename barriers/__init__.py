from models.torchsym import TorchSymDiffModel
from systems import ControlAffineDynamics


def make_barrier(
    system: ControlAffineDynamics, **kwargs
) -> dict[str, TorchSymDiffModel]:
    if system.id == "SingleIntegrator":
        from barriers.single_integrator import SingleIntegratorCBF

        barrier = SingleIntegratorCBF(system=system)
        compensator = None
        return {"barrier": barrier, "compensator": compensator}
    if system.id == "SingleIntegrator_AdditiveBounded":
        from barriers.single_integrator import SingleIntegratorCBF
        from barriers.single_integrator import (
            SingleIntegratorCompensatorAdditiveBoundedUncertainty,
        )

        barrier = SingleIntegratorCBF(system=system)
        compensator = SingleIntegratorCompensatorAdditiveBoundedUncertainty(
            h=barrier, system=system
        )
        return {"barrier": barrier, "compensator": compensator}
    else:
        raise NotImplementedError(f"barrier for {system.id} not implemented")
