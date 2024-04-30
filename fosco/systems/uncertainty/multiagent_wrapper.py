import numpy as np
import torch

from fosco.common import domains
from fosco.common.domains import Set
from fosco.systems import ControlAffineDynamics, MultiParticle
from fosco.systems.uncertainty.uncertainty_wrapper import UncertaintyWrapper, register


@register
class DynamicAgents(UncertaintyWrapper):

    def __init__(self, system: ControlAffineDynamics):
        assert isinstance(system, MultiParticle)
        super(DynamicAgents, self).__init__(system)

    @property
    def uncertainty_id(self) -> str:
        return self.__class__.__name__

    @property
    def uncertain_vars(self) -> tuple[str, ...]:
        variables = []
        # _aid suffix for the agent id, excluding 0 for the first agent (ego)
        for aid in range(1, self._base_system._n_agents):
            variables.extend(
                [f"z{var}_{aid}" for var in self._base_system._single_agent_dynamics.controls]
            )
        return tuple(variables)

    @property
    def uncertainty_domain(self) -> Set:
        # assume the uncertain variables are the input domains
        lb = np.array(
            self._base_system._single_agent_dynamics.input_domain.lower_bounds * (self._base_system._n_agents - 1)
        )
        ub = np.array(
            self._base_system._single_agent_dynamics.input_domain.upper_bounds * (self._base_system._n_agents - 1)
        )
        return domains.Rectangle(vars=self.uncertain_vars, lb=tuple(lb), ub=tuple(ub),)

    def fz_torch(
        self, x: np.ndarray | torch.Tensor, z: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        n_vars_i = self._base_system._single_agent_dynamics.n_vars
        n_ctrl_i = self._base_system._single_agent_dynamics.n_controls
        n_agents = self._base_system._n_agents

        if isinstance(x, np.ndarray):
            fz = np.zeros((self.n_vars, self.n_uncertain))
            fz[n_vars_i:, :] = np.eye((n_agents - 1) * n_ctrl_i)
            fz_batch = np.tile(fz, (x.shape[0], 1, 1))
        else:
            fz = torch.zeros((self.n_vars, self.n_uncertain))
            fz[n_vars_i:, :] = torch.eye((n_agents-1) * n_ctrl_i)
            fz_batch = torch.tile(fz, (x.shape[0], 1, 1)).to(x.device)
        return fz_batch @ z

    def fz_smt(self, x: list, z: list) -> np.ndarray | torch.Tensor:
        n_vars_i = self._base_system._single_agent_dynamics.n_vars
        n_ctrl_i = self._base_system._single_agent_dynamics.n_controls
        n_agents = self._base_system._n_agents

        fz = np.zeros((self.n_vars, self.n_uncertain))
        fz[n_vars_i:, :] = np.eye((n_agents - 1) * n_ctrl_i)
        return fz @ z

    def gz_torch(
        self, x: np.ndarray | torch.Tensor, z: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if isinstance(x, np.ndarray):
            gx = np.zeros((self.n_vars, self.n_controls))[None].repeat(
                x.shape[0], axis=0
            )
        else:
            gx = torch.zeros((self.n_vars, self.n_controls))[None].repeat(
                (x.shape[0], 1, 1)
            ).to(x.device)
        return gx

    def gz_smt(self, x: list, z: list) -> np.ndarray | torch.Tensor:
        return np.zeros((self.n_vars, self.n_controls))

    def render_state_with_objects(self, **kwargs):
        return self._base_system.render_state_with_objects(**kwargs)