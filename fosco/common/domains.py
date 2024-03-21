import math
from functools import partial

import numpy as np
import torch

from fosco.common.utils import round_init_data, square_init_data
from fosco.verifier.utils import get_solver_fns
from fosco.verifier.verifier import SYMBOL


class Set:
    def __init__(self, vars: list[str]) -> None:
        self.vars = vars
        self.dimension = len(self.vars)

    def generate_domain(self, x) -> SYMBOL:
        raise NotImplementedError

    def generate_data(self, batch_size) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        try:
            return self.to_latex()
        except TypeError:
            return self.__class__.__name__

    def _generate_data(self, batch_size) -> callable:
        """
        Lazy version of generate_data, returns a function that generates data when called
        param batch_size: number of data points to generate
        returns: data points generated in relevant domain according to shape
        """
        # return partial to deal with pickle
        return partial(self.generate_data, batch_size)

    def check_containment(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SumToOneSet(Set):
    """
    A set whose elements are positive and sum up to one.
    """
    def __repr__(self):
        """
        Return a string representation of the domain.
        e.g., print(SumToOneSet()) => 'SumToOneSet()'
        """
        return f"SumToOneSet{self.vars}"

    def generate_domain(self, z) -> verifier.SYMBOL:
        """
        Generates symbolic domain.

        :param z: symbolic variables
        :return: symbolic expression
        """
        f = verifier.FUNCTIONS
        z_dim = [i for i, vz in enumerate(z) if str(vz) in self.vars]
        positivity = f["And"](*[0.0 <= z[v_id] for v_id in z_dim])
        z_sum = 0.0
        for v_id in z_dim:
            z_sum += z[v_id]
        return f["And"](positivity, z_sum == 1)

    def generate_data(self, batch_size) -> torch.Tensor:
        """
        Generates data samples in the domain.

        :param batch_size: number of samples
        :return: batch of data
        """
        sum_to_one_data = torch.rand(batch_size, len(self.vars))
        sum_to_one_data /= sum_to_one_data.sum(dim=-1).unsqueeze(-1)
        return sum_to_one_data

    def check_containment(self, z: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Checks if the given batch of samples is contained in the domain.

        :param z: batch of data samples
        :return: boolean batch
        """
        batch_size = z.shape[0]
        assert z.shape[1] == len(self.vars)
        contain_validity = torch.sum(z, sum=1) == torch.ones(batch_size)
        return contain_validity


class Rectangle(Set):
    def __init__(
        self,
        lb: tuple[float, ...],
        ub: tuple[float, ...],
        vars: list[str],
        dim_select=None,
    ):
        self.name = "box"
        self.lower_bounds = lb
        self.upper_bounds = ub
        self.dim_select = dim_select
        self.volume = np.prod([ub_i - lb_i for lb_i, ub_i in zip(lb, ub)])
        super().__init__(vars=vars)

    def __repr__(self):
        return f"Rectangle{self.lower_bounds, self.upper_bounds}"

    def generate_domain(self, x):
        """
        param x: data point x
        returns: symbolic formula for domain
        """
        dim_selection = [i for i, vx in enumerate(x) if str(vx) in self.vars]
        fns = get_solver_fns(x=x)
        lower = fns["And"](
            *[x[v_id] >= self.lower_bounds[i] for i, v_id in enumerate(dim_selection)]
        )
        upper = fns["And"](
            *[x[v_id] <= self.upper_bounds[i] for i, v_id in enumerate(dim_selection)]
        )
        return fns["And"](lower, upper)

    def generate_data(self, batch_size):
        """
        param x: data point x
        returns: data points generated in relevant domain according to shape
        """
        return square_init_data(
            [self.lower_bounds, self.upper_bounds], batch_size
        ).float()

    def get_vertices(self):
        """Returns vertices of the rectangle

        Returns:
            List: vertices of the rectangle
        """
        spaces = [
            np.linspace(lb, ub, 2)
            for lb, ub in zip(self.lower_bounds, self.upper_bounds)
        ]
        vertices = np.meshgrid(*spaces)
        vertices = np.array([v.flatten() for v in vertices]).T
        return vertices

    def check_containment(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 2, f"Expected x to be 2D, got {x.shape}"
        if self.dim_select:
            x = np.array([x[:, i] for i in self.dim_select])
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        all_constr = torch.logical_and(
            torch.tensor(self.upper_bounds) >= x, torch.tensor(self.lower_bounds) <= x
        )
        ans = torch.zeros((x.shape[0]))
        for idx in range(all_constr.shape[0]):
            ans[idx] = all_constr[idx, :].all()

        return ans.bool()


class Sphere(Set):
    def __init__(
        self,
        center,
        radius,
        vars: list[str],
        dim_select=None,
        include_boundary: bool = True,
    ):
        self.center = center
        self.radius = radius
        self.include_boundary = include_boundary
        super().__init__(vars=vars)
        self.dim_select = dim_select
        self.volume = math.pi ** (self.dimension / 2) / math.gamma(
            self.dimension / 2 + 1
        )

    def __repr__(self) -> str:
        return f"Sphere{self.center, self.radius}"

    def generate_domain(self, x):
        """
        param x: data point x
        returns: symbolic formula for domain
        """
        if self.dim_select:
            x = [x[i] for i in self.dim_select]

        if self.include_boundary:
            domain = (
                sum([(x[i] - self.center[i]) ** 2 for i in range(self.dimension)])
                <= self.radius ** 2
            )
        else:
            domain = (
                sum([(x[i] - self.center[i]) ** 2 for i in range(self.dimension)])
                < self.radius ** 2
            )
        return domain

    def generate_data(self, batch_size):
        """
        param batch_size: number of data points to generate
        returns: data points generated in relevant domain according to shape
        """
        return round_init_data(self.center, self.radius ** 2, batch_size)

    def check_containment(
        self, x: np.ndarray | torch.Tensor, epsilon: float = 1e-6
    ) -> torch.Tensor:
        """
        Check if the points in x are contained in the sphere.

        Args:
            x: batch of points to check
            epsilon: tolerance for the checking up to numerical precision

        Returns:
            torch.Tensor: boolean tensor with True for points contained in the sphere
        """
        assert len(x.shape) == 2, f"Expected x to be 2D, got {x.shape}"
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if self.dim_select:
            x = x[:, self.dim_select]
        c = torch.tensor(self.center).reshape(1, -1).to(x.device)
        return (x - c).norm(2, dim=-1) - self.radius ** 2 <= epsilon


class Union(Set):
    """
    Set formed by union of sets.
    """

    def __init__(self, sets: list[Set]) -> None:
        assert all(
            [set(sets[0].vars) == set(s.vars) for s in sets]
        ), "Sets must have the same variables"
        super().__init__(vars=sets[0].vars)
        self.sets = sets

    def __repr__(self) -> str:
        return "(" + " | ".join([f"({s})" for s in self.sets]) + ")"

    def generate_domain(self, x):
        fns = get_solver_fns(x=x)
        return fns["Or"](*[s.generate_domain(x) for s in self.sets])

    def generate_data(self, batch_size):
        """
        Sample data from each set, proportially to the volume if available.
        """
        volumes = [s.volume for s in self.sets if hasattr(s, "volume") or None]
        nonz_volumes = [v for v in volumes if v is not None]

        if len(nonz_volumes) > 0:
            # if volume is not available, use average of other volumes
            volumes = [v if v is not None else np.mean(nonz_volumes) for v in volumes]
            # normalize volumes to value in 0..1 that sum up to 1
            norm_volumes = np.array([v / sum(volumes) for v in volumes])
            # compute the number of samples to take from each set
            n_per_set = np.ceil(norm_volumes * batch_size).astype(int)
        else:
            # if no volume available, sample uniformly from each set
            n_per_set = np.ceil([batch_size / len(self.sets)] * len(self.sets)).astype(
                int
            )

        s = torch.empty(0, self.dimension)
        for set_i, n_set_i in zip(self.sets, n_per_set):
            s = torch.cat([s, set_i.generate_data(n_set_i)])
        return s[:batch_size]


class Intersection(Set):
    """
    Set formed by intersection of S1 and S2
    """

    def __init__(self, sets: list[Set]) -> None:
        assert all(
            [set(sets[0].vars) == set(s.vars) for s in sets]
        ), "Sets must have the same variables"
        super().__init__(vars=sets[0].vars)
        self.sets = sets

    def __repr__(self) -> str:
        return "(" + "&".join([f"({s})" for s in self.sets]) + ")"

    def generate_domain(self, x):
        fns = get_solver_fns(x=x)
        return fns["And"](*[s.generate_domain(x) for s in self.sets])

    def generate_data(self, batch_size: int, max_iter: int = 1000) -> torch.Tensor:
        """
        Rejection sampling to generate data in the intersection of S1 and S2.

        Args:
            batch_size: number of data points to generate
            max_iter: maximum number of iterations for rejection sampling

        Returns:
            torch.Tensor: data points generated in the intersection of S1 and S2
        """
        samples = torch.empty(0, self.dimension)
        while len(samples) < batch_size and max_iter > 0:
            rnd_set_id = np.random.randint(0, len(self.sets))
            s = self.sets[rnd_set_id].generate_data(batch_size=batch_size)
            for s_i in self.sets:
                s = s[s_i.check_containment(s)]
            samples = torch.cat([samples, s])
            max_iter -= 1
        return samples[:batch_size]
