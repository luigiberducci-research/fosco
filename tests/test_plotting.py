import unittest

import matplotlib.pyplot as plt
import numpy as np

from fosco.common.domains import Rectangle, Sphere, Intersection
from fosco.models import TorchMLP
from fosco.plotting.domains import plot_rectangle, plot_sphere, plot_domain
from fosco.plotting.functions import plot_func_and_domains


class TestPlottingUtils(unittest.TestCase):
    def test_plot_random_network_2d(self):
        from plotly.graph_objs import Figure

        n_dim = 2
        domain = Rectangle(
            vars=[f"x{i}" for i in range(n_dim)], lb=(-1.0,) * n_dim, ub=(1.0,) * n_dim,
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
            vars=[f"x{i}" for i in range(n_dim)], lb=(-1.0,) * n_dim, ub=(1.0,) * n_dim,
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

    def test_plot_domains(self):
        from plotly.graph_objs import Figure
        import matplotlib.pyplot as plt

        box = Rectangle(vars=["x0", "x1", "x2"], lb=[0, 1, 0], ub=[2, 3, 4])
        sphere = Sphere(vars=["x0", "x1", "x2"], center=[0, 1, 3], radius=1.333)
        intersect = Intersection(sets=[box, sphere])

        fig = Figure()
        for domain, color in zip([box, sphere, intersect], ["red", "green", "blue"]):
            fig = plot_domain(domain, fig, color=color)
        self.assertTrue(
            isinstance(fig, Figure), f"plot should return a plotly figure, got {fig}"
        )
        # fig.show()

        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for domain, color in zip([box, sphere, intersect], ["red", "green", "blue"]):
            fig = plot_domain(domain, fig, color=color)
        self.assertTrue(
            isinstance(fig, plt.Figure),
            f"plot should return a matplotlib figure, got {fig}",
        )
        # plt.show()

    def test_plot_datasets(self):
        from plotly.graph_objs import Figure
        import matplotlib.pyplot as plt
        from fosco.plotting.utils2d import scatter_datasets

        from fosco.systems import make_system

        system = make_system("SingleIntegrator")()
        domains = system.domains

        datas = {}
        for name, domain in domains.items():
            data = domain.generate_data(100)
            datas[name] = data

        fig = Figure()
        fig = scatter_datasets(datas, counter_examples={}, fig=fig)
        self.assertTrue(
            isinstance(fig, Figure), f"plot should return a plotly figure, got {fig}"
        )
        # fig.show()

        fig, ax = plt.subplots()
        fig = scatter_datasets(datas, counter_examples={}, fig=fig)
        self.assertTrue(
            isinstance(fig, plt.Figure),
            f"plot should return a matplotlib figure, got {fig}",
        )
        # plt.show()

    def test_plot_functions(self):
        from plotly.graph_objs import Figure
        import matplotlib.pyplot as plt
        from fosco.plotting.utils3d import plot_surface

        def func(x):
            assert (
                len(x.shape) == 2 and x.shape[1] == 2
            ), "x must be a batch of 2d points"
            return np.sin(x[:, 0])  # + np.cos(x[:, 1])

        fig = Figure()
        fig = plot_surface(func, (-10, 10), (-5, 5), levels=[0], label="test", fig=fig)
        self.assertTrue(
            isinstance(fig, Figure), f"plot should return a plotly figure, got {fig}"
        )
        # fig.show()

        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        fig = plot_surface(func, (-10, 10), (-5, 5), levels=[0], label="test", fig=fig)
        self.assertTrue(
            isinstance(fig, plt.Figure),
            f"plot should return a matplotlib figure, got {fig}",
        )
        # plt.show()
