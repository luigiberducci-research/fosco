from models.torchsym import TorchSymModel
from systems import ControlAffineDynamics


def make_barrier(system: ControlAffineDynamics, **kwargs) -> dict[str, TorchSymModel]:
    if system.id == "single_integrator":
        from barriers.single_integrator import SingleIntegratorCBF
        return {
            "barrier": SingleIntegratorCBF(system=system),
            "compensator": None,
        }

    else:
        raise NotImplementedError(f"barrier {id} not implemented")