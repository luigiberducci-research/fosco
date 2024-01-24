from models.torchsym import TorchSymModel


def make_barrier(id: str, **kwargs) -> TorchSymModel:
    if id == "single_integrator":
        from barriers.single_integrator import SingleIntegratorCBF
        return SingleIntegratorCBF(**kwargs)
    else:
        raise NotImplementedError(f"barrier {id} not implemented")