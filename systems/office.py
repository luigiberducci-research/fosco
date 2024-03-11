from fosco.common import domains
from fosco.common.domains import Set
from systems import SingleIntegrator
from systems.system import register


@register
class OfficeSingleIntegrator(SingleIntegrator):
    """
    Extends SingleIntegrator to include obstacle in a static office environment.
    """

    @property
    def id(self) -> str:
        return self.__class__.__name__

    @property
    def state_domain(self) -> Set:
        return domains.Rectangle(
            vars=self.vars, lb=(-10.0,) * self.n_vars, ub=(10.0,) * self.n_vars
        )

    @property
    def input_domain(self) -> Set:
        return domains.Rectangle(
            vars=self.controls,
            lb=(-5.0,) * self.n_controls,
            ub=(5.0,) * self.n_controls,
        )

    @property
    def init_domain(self) -> Set:
        return domains.Rectangle(
            vars=self.vars, lb=(-10.0,) * self.n_vars, ub=(-9.0,) * self.n_vars
        )

    @property
    def unsafe_domain(self) -> Set:
        human_radius = 0.36
        plant_radius = 0.4

        humans = [
            domains.Sphere(vars=self.vars, centre=(0.44956875, 3.6988103), radius=human_radius, include_boundary=False),
            domains.Sphere(vars=self.vars, centre=(4.0826755, -1.6512532), radius=human_radius, include_boundary=False),
            domains.Sphere(vars=self.vars, centre=(3.0811677, -1.0228925), radius=human_radius, include_boundary=False),
            domains.Sphere(vars=self.vars, centre=(3.03672, -2.3108509), radius=human_radius, include_boundary=False),
            domains.Sphere(vars=self.vars, centre=(-0.69713455, 1.9523418), radius=human_radius, include_boundary=False)
        ]
        plants = [
            domains.Sphere(vars=self.vars, centre=(2.2344275, 3.9294362), radius=plant_radius, include_boundary=False),
            domains.Sphere(vars=self.vars, centre=(-2.9687812, -4.31725), radius=plant_radius, include_boundary=False),
        ]

        table_cx = -0.6971345718589101
        table_cy = 0.7023418244845709
        table_w = 1.5
        table_l = 3.0
        tables = [
            domains.Rectangle(vars=self.vars,
                              lb=(table_cx - table_w / 2, table_cy - table_l / 2),
                              ub=(table_cx + table_w / 2, table_cy + table_l / 2))
        ]

        return domains.Union(sets=humans + plants + tables)


