from functools import partial

import numpy as np
import torch

from fosco import verifier
from fosco.common.utils import round_init_data, square_init_data
from fosco.verifier.utils import get_solver_fns
from fosco.verifier.verifier import SYMBOL


class Set:
    def __init__(self, vars: list[str] = None) -> None:
        if vars is None:
            vars = [f"x{i}" for i in range(self.dimension)]
        self.vars = vars

    def generate_domain(self, x) -> SYMBOL:
        raise NotImplementedError

    def generate_data(self, batch_size) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        try:
            return self.to_latex()
        except TypeError:
            return self.__class__.__name__

    def generate_complement(self, x) -> SYMBOL:
        """Generates complement of the set as a symbolic formulas

        Args:
            x (list): symbolic data point

        Returns:
            SMT variable: symbolic representation of complement of the rectangle
        """
        raise NotImplementedError("access to fns to be fixed")
        f = verifier.functions(x)
        return f["Not"](self.generate_domain(x))

    def _generate_data(self, batch_size) -> callable:
        """
        Lazy version of generate_data, returns a function that generates data when called
        param batch_size: number of data points to generate
        returns: data points generated in relevant domain according to shape
        """
        # return partial to deal with pickle
        return partial(self.generate_data, batch_size)

    def sample_border(self, batch_size) -> torch.Tensor:
        raise NotImplementedError

    def _sample_border(self, batch_size) -> callable:
        """
        Lazy version of sample_border, returns a function that generates data when called
        param batch_size: number of data points to generate
        returns: data points generated in relevant domain according to shape
        """
        # return partial to deal with pickle
        return partial(self.sample_border, batch_size)

    def check_containment(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Rectangle(Set):
    def __init__(
        self,
        lb: tuple[float, ...],
        ub: tuple[float, ...],
        vars: list[str] = None,
        dim_select=None,
    ):
        self.name = "square"
        self.lower_bounds = lb
        self.upper_bounds = ub
        self.dimension = len(lb)
        self.dim_select = dim_select
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
            *[self.lower_bounds[i] <= x[v_id] for i, v_id in enumerate(dim_selection)]
        )
        upper = fns["And"](
            *[x[v_id] <= self.upper_bounds[i] for i, v_id in enumerate(dim_selection)]
        )
        return fns["And"](lower, upper)

    def generate_boundary(self, x):
        """Returns boundary of the rectangle

        Args:
            x (List): symbolic data point

        Returns:
            symbolic formula for boundary of the rectangle
        """

        fns = get_solver_fns(x=x)
        lower = fns["Or"](
            *[self.lower_bounds[i] == x[i] for i in range(self.dimension)]
        )
        upper = fns["Or"](
            *[x[i] == self.upper_bounds[i] for i in range(self.dimension)]
        )
        return fns["Or"](lower, upper)

    def generate_interior(self, x):
        """Returns interior of the rectangle

        Args:
            x (List): symbolic data point
        """
        fns = get_solver_fns(x=x)
        lower = fns["And"](
            *[self.lower_bounds[i] < x[i] for i in range(self.dimension)]
        )
        upper = fns["And"](
            *[x[i] < self.upper_bounds[i] for i in range(self.dimension)]
        )
        return fns["And"](lower, upper)

    def generate_data(self, batch_size):
        """
        param x: data point x
        returns: data points generated in relevant domain according to shape
        """
        return square_init_data([self.lower_bounds, self.upper_bounds], batch_size)

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
        if self.dim_select:
            x = np.array([x[:, i] for i in self.dim_select])
        x = torch.from_numpy(x)
        all_constr = torch.logical_and(
            torch.tensor(self.upper_bounds) >= x, torch.tensor(self.lower_bounds) <= x
        )
        ans = torch.zeros((x.shape[0]))
        for idx in range(all_constr.shape[0]):
            ans[idx] = all_constr[idx, :].all()

        return ans.bool()

    def check_containment_grad(self, x: torch.Tensor) -> torch.Tensor:
        # check containment and return a tensor with gradient
        if self.dim_select:
            x = [x[:, i] for i in self.dim_select]

        # returns 0 if it IS contained, a positive number otherwise
        return torch.relu(
            torch.sum(x - torch.tensor(self.upper_bounds), dim=1)
        ) + torch.relu(torch.sum(torch.tensor(self.lower_bounds) - x, dim=1))


class Sphere(Set):
    def __init__(
        self,
        centre,
        radius,
        vars: list[str] = None,
        dim_select=None,
        include_boundary: bool = True,
    ):
        self.center = centre
        self.radius = radius
        self.dimension = len(centre)
        self.include_boundary = include_boundary
        super().__init__(vars=vars)
        self.dim_select = dim_select

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
                <= self.radius**2
            )
        else:
            domain = (
                sum([(x[i] - self.center[i]) ** 2 for i in range(self.dimension)])
                < self.radius**2
            )
        return domain

    def generate_data(self, batch_size):
        """
        param batch_size: number of data points to generate
        returns: data points generated in relevant domain according to shape
        """
        return round_init_data(self.center, self.radius ** 2, batch_size)

    def sample_border(self, batch_size):
        """
        param batch_size: number of data points to generate
        returns: data points generated on the border of the set
        """
        return round_init_data(
            self.center, self.radius ** 2, batch_size, on_border=True
        )

    def check_containment(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if self.dim_select:
            x = np.array([x[:, i] for i in self.dim_select])
        x = torch.from_numpy(x)
        c = torch.tensor(self.center).reshape(1, -1)
        return (x - c).norm(2, dim=-1) <= self.radius**2

    def check_containment_grad(self, x: torch.Tensor) -> torch.Tensor:
        # check containment and return a tensor with gradient
        c = torch.tensor(self.center).reshape(1, -1)
        if self.dim_select:
            x = x[:, :, self.dim_select]
            c = [self.center[i] for i in self.dim_select]
            c = torch.tensor(c).reshape(1, -1)
        # returns 0 if it IS contained, a positive number otherwise
        return torch.relu((x - c).norm(2, dim=-1) - self.radius**2)
