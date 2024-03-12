import unittest

from fosco.common.domains import Rectangle
from fosco.models import TorchMLP
from fosco.plotting.utils import plot_func_and_domains


class TestPlottingUtils(unittest.TestCase):
    def test_plot_random_network_2d(self):
        from plotly.graph_objs import Figure

        n_dim = 2
        domain = Rectangle(
            vars=[f"x{i}" for i in range(n_dim)],
            lb=(-1.0,) * n_dim,
            ub=(1.0,) * n_dim,
        )
        model = TorchMLP(
            input_size=domain.dimension,
            output_size=1,
            hidden_sizes=(32, 32),
            activation=("ReLU", "ReLU"),
        )

        fig = plot_func_and_domains(
            func=model,
            in_domain=domain,
            levels=None,  # test default value
            domains={"domain": domain},
            dim_select=None,  # test default value
        )
        self.assertTrue(
            isinstance(fig, Figure), "plot should return a figure, got {fig}"
        )

    def test_plot_random_network_5d(self):
        from plotly.graph_objs import Figure

        n_dim = 3
        domain = Rectangle(
            vars=[f"x{i}" for i in range(n_dim)],
            lb=(-1.0,) * n_dim,
            ub=(1.0,) * n_dim,
        )
        model = TorchMLP(
            input_size=domain.dimension,
            output_size=1,
            hidden_sizes=(32, 32),
            activation=("ReLU", "ReLU"),
        )

        for i0 in range(domain.dimension):
            for i1 in range(i0 + 1, domain.dimension):
                fig = plot_func_and_domains(
                    func=model,
                    in_domain=domain,
                    levels=[0.0],
                    domains={"domain": domain},
                    dim_select=(i0, i1),
                )
                self.assertTrue(
                    isinstance(fig, Figure), "plot should return a figure, got {fig}"
                )
