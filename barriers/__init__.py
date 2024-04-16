import pathlib
from typing import Optional

from fosco.models import TorchSymDiffModel, TorchSymModel
from fosco.models.utils import load_model
from fosco.systems import ControlAffineDynamics, UncertainControlAffineDynamics


def make_barrier(
    system: ControlAffineDynamics,
    model_to_load: Optional[str | pathlib.Path] = "default",
) -> TorchSymDiffModel:
    assert isinstance(system, ControlAffineDynamics), f"got {type(system)}"
    assert isinstance(model_to_load, str) or isinstance(
        model_to_load, pathlib.Path
    ), f"got {type(model_to_load)}"

    if system.id.startswith("SingleIntegrator") and model_to_load == "default":
        from barriers.single_integrator import SingleIntegratorCBF

        return SingleIntegratorCBF(system=system)

    if system.id.startswith("DoubleIntegrator") and model_to_load == "default":
        from barriers.double_integrator import DoubleIntegratorCBF

        return DoubleIntegratorCBF(system=system)

    if str(model_to_load).endswith(".yaml"):
        assert pathlib.Path(
            model_to_load
        ).exists(), f"model path {model_to_load} does not exist"
        return load_model(config_path=model_to_load)

    raise NotImplementedError(
        f"barrier {model_to_load} for {system.id} not implemented"
    )


def make_compensator(
    system: UncertainControlAffineDynamics,
    model_to_load: Optional[str | pathlib.Path] = "default",
) -> TorchSymModel:
    assert isinstance(system, UncertainControlAffineDynamics), f"got {type(system)}"
    assert isinstance(model_to_load, str) or isinstance(
        model_to_load, pathlib.Path
    ), f"got {type(model_to_load)}"

    if str(model_to_load).endswith(".yaml"):
        assert pathlib.Path(
            model_to_load
        ).exists(), f"model path {model_to_load} does not exist"
        return load_model(config_path=model_to_load)

    if system.id == "SingleIntegrator_AdditiveBounded" and model_to_load == "default":
        from barriers.single_integrator import SingleIntegratorCBF
        from barriers.single_integrator import (
            SingleIntegratorCompensatorAdditiveBoundedUncertainty,
        )

        barrier = SingleIntegratorCBF(system=system)
        compensator = SingleIntegratorCompensatorAdditiveBoundedUncertainty(
            h=barrier, system=system
        )
        return compensator

    if system.id == "SingleIntegrator_AdditiveBounded" and model_to_load == "tunable":
        from barriers.single_integrator import SingleIntegratorCBF
        from barriers.single_integrator import (
            SingleIntegratorTunableCompensatorAdditiveBoundedUncertainty,
        )

        barrier = SingleIntegratorCBF(system=system)
        compensator = SingleIntegratorTunableCompensatorAdditiveBoundedUncertainty(
            h=barrier, system=system
        )
        return compensator

    raise NotImplementedError(
        f"barrier models of type {model_to_load} for {system.id} not implemented"
    )
