"""Minimal reproducible load-case configurations.

Three cases are provided:
1) Pure heat conduction
2) Pure thermo-elastic
3) Coupled thermo-mechanical damage
"""

from project.config import Config


def pure_heat_case() -> Config:
    cfg = Config()
    cfg.train.num_time_steps = 2
    cfg.train.n_domain = 256
    cfg.train.n_boundary = 128
    cfg.train.n_initial = 128
    cfg.train.n_quadrature = 256

    cfg.train.w_T = 1.0
    cfg.train.w_u = 0.0

    cfg.load.thermal.initial_temperature = 293.15
    cfg.load.thermal.heating_curve = ((0.0, 293.15), (1.0, 473.15))
    cfg.load.thermal.thermal_bc_tags = ("top",)
    cfg.load.thermal.boundary_temperature = {"top": 293.15}
    cfg.load.thermal.boundary_temperature_amplitude = {"top": 1.0}

    cfg.load.mechanical.displacement_constraints = {
        "left": (True, False),
        "bottom": (False, True),
    }
    cfg.load.mechanical.boundary_displacement = {"left": (0.0, 0.0), "bottom": (0.0, 0.0)}
    return cfg


def pure_thermoelastic_case() -> Config:
    cfg = Config()
    cfg.train.num_time_steps = 2
    cfg.train.w_T = 1.0
    cfg.train.w_u = 1.0

    cfg.load.thermal.initial_temperature = 293.15
    cfg.load.thermal.heating_curve = ((0.0, 293.15), (1.0, 393.15))
    cfg.load.thermal.thermal_bc_tags = ("top", "right")
    cfg.load.thermal.boundary_temperature = {"top": 293.15, "right": 293.15}

    cfg.load.mechanical.displacement_constraints = {
        "left": (True, False),
        "bottom": (False, True),
        "right": (True, False),
    }
    cfg.load.mechanical.boundary_displacement = {
        "left": (0.0, 0.0),
        "bottom": (0.0, 0.0),
        "right": (2e-5, 0.0),
    }
    cfg.load.mechanical.boundary_displacement_amplitude = {"right": (1.0, 0.0)}
    return cfg


def coupled_case() -> Config:
    cfg = Config()
    cfg.train.num_time_steps = 3
    cfg.train.w_T = 1.0
    cfg.train.w_u = 1.0

    cfg.load.time_function.points = ((0.0, 0.0), (0.3, 0.4), (1.0, 1.0))
    cfg.load.time_function.scale = 1.0
    cfg.load.time_function.offset = 0.0

    cfg.load.thermal.initial_temperature = 293.15
    cfg.load.thermal.heating_curve = ((0.0, 293.15), (1.0, 573.15))
    cfg.load.thermal.thermal_bc_tags = ("top", "left")
    cfg.load.thermal.boundary_temperature = {"top": 293.15, "left": 293.15}
    cfg.load.thermal.boundary_temperature_amplitude = {"top": 1.0, "left": 0.25}

    cfg.load.mechanical.displacement_constraints = {
        "left": (True, False),
        "bottom": (False, True),
        "right": (True, True),
    }
    cfg.load.mechanical.boundary_displacement = {
        "left": (0.0, 0.0),
        "bottom": (0.0, 0.0),
        "right": (2e-4, -1e-4),
    }
    cfg.load.mechanical.boundary_displacement_amplitude = {"right": (1.0, 0.5)}
    return cfg
