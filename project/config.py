from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class MaterialConfig:
    # COMSOL baseline (SI units)
    E: float = 70e9
    nu: float = 0.33
    alpha: float = 23e-6
    rho: float = 2700.0
    c_p: float = 900.0
    k0: float = 180.0
    T_ref: float = 273.15

    # Gf0 = 0.0024 MPa*mm = 2.4 N/m (SI)
    GcI: float = 2.4
    xi: float = 1.0
    GcII: float | None = None
    l0: float = 1e-3
    eta_pf: float = 1e-4
    kappa: float = 1e-8
    eps_r: float = 1e-5


    def __post_init__(self) -> None:
        if self.GcII is None:
            self.GcII = 2.0 * (1.0 + self.nu) * (self.xi**2) * self.GcI


@dataclass
class TrainConfig:
    t0: float = 0.0
    tf: float = 1.0
    num_time_steps: int = 20

    adam_lr: float = 1e-3
    adam_epochs_tu: int = 1000
    adam_epochs_d: int = 1000
    lbfgs_iters_tu: int = 300
    lbfgs_iters_d: int = 300

    w_T: float = 1.0
    w_u: float = 1.0
    mech_mode: str = "stress_correction"

    n_domain: int = 5000
    n_boundary: int = 1000
    n_initial: int = 2000
    n_quadrature: int = 4000
    irreversibility: str = "output_transform"


@dataclass
class LoadConfig:
    Q: float = 0.0


@dataclass
class RuntimeConfig:
    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class ThermalLoadConfig:
    initial_temperature: float = 573.15
    # Piecewise-linear temperature-time curve [(t, T), ...]
    heating_curve: Tuple[Tuple[float, float], ...] = ((0.0, 573.15), (1.0, 573.15))
    thermal_bc_tags: Tuple[str, ...] = ("top",)


@dataclass
class MechanicalLoadConfig:
    # Boundary tag -> constrained directions (x, y)
    displacement_constraints: Dict[str, Tuple[bool, bool]] = field(
        default_factory=lambda: {"left": (True, False), "bottom": (False, True)}
    )
    prescribed_displacement: Tuple[float, float] = (0.0, 0.0)


@dataclass
class LoadConfig:
    thermal: ThermalLoadConfig = field(default_factory=ThermalLoadConfig)
    mechanical: MechanicalLoadConfig = field(default_factory=MechanicalLoadConfig)


@dataclass
class Config:
    material: MaterialConfig = field(default_factory=MaterialConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    load: LoadConfig = field(default_factory=LoadConfig)
