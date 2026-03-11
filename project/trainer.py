from dataclasses import dataclass
from typing import Callable, Dict


import torch

from .config import Config
from .geometry_sampling import RectDomain, Sampler
from .history_update import update_history_fields
from .losses_phasefield import phasefield_loss
from .losses_thermo_mech import thermo_mech_total_loss
from .networks import PhaseFieldNet, ThermoMechNet


@dataclass
class TimeStepState:
    t: float
    d_q: torch.Tensor
    HI: torch.Tensor
    HII: torch.Tensor
    He: torch.Tensor


class CoupledTrainer:
    def __init__(self, cfg: Config, domain: RectDomain):
        self.cfg = cfg
        self.device = torch.device(cfg.runtime.device)

        self.net_tu = ThermoMechNet().to(self.device)
        self.net_d = PhaseFieldNet().to(self.device)

        self.sampler = Sampler(domain, self.device)
        self.dt = (cfg.train.tf - cfg.train.t0) / cfg.train.num_time_steps
        self.irreversibility = cfg.train.irreversibility

    def _interpolate_curve(self, t: torch.Tensor, curve) -> torch.Tensor:
        if len(curve) == 0:
            return torch.zeros_like(t)

        t_scalar = t[:, 0:1]
        t_out = torch.empty_like(t_scalar)

        if len(curve) == 1:
            t_out.fill_(curve[0][1])
            return t_out

        t0, v0 = curve[0]
        t_out[t_scalar <= t0] = float(v0)

        for i in range(len(curve) - 1):
            ta, va = curve[i]
            tb, vb = curve[i + 1]
            seg = (t_scalar >= ta) & (t_scalar <= tb)
            ratio = (t_scalar[seg] - ta) / max(tb - ta, 1e-12)
            t_out[seg] = va + ratio * (vb - va)

        t_last, v_last = curve[-1]
        t_out[t_scalar >= t_last] = float(v_last)
        return t_out

    def _load_time_factor(self, t: torch.Tensor) -> torch.Tensor:
        tf_cfg = self.cfg.load.time_function
        values = self._interpolate_curve(t, tf_cfg.points)
        return tf_cfg.offset + tf_cfg.scale * values

    def _interpolate_heating_curve(self, t: torch.Tensor) -> torch.Tensor:
        curve = self.cfg.load.thermal.heating_curve
        if len(curve) == 0:
            return torch.full_like(t, self.cfg.load.thermal.initial_temperature)
        return self._interpolate_curve(t, curve)

    def _default_temperature_bc_fn(self, xyt_bc: torch.Tensor, bc_labels: torch.Tensor) -> torch.Tensor:
        target = torch.zeros((xyt_bc.shape[0], 1), device=self.device)
        t_curve = self._interpolate_heating_curve(xyt_bc[:, 2:3])
        load_factor = self._load_time_factor(xyt_bc[:, 2:3])

        for tag in self.cfg.load.thermal.thermal_bc_tags:
            tag_id = self.sampler.BOUNDARY_TAG_TO_ID[tag]
            mask = bc_labels.squeeze(-1) == tag_id
            base = self.cfg.load.thermal.boundary_temperature.get(tag, 0.0)
            amp = self.cfg.load.thermal.boundary_temperature_amplitude.get(tag, 1.0)
            target[mask] = base + amp * load_factor[mask] * t_curve[mask]

        t_min, t_max = self.cfg.load.temperature_bounds
        return torch.clamp(target, min=t_min, max=t_max)

    def _default_temperature_init_fn(self, xyt_init: torch.Tensor) -> torch.Tensor:
        t0 = torch.full((xyt_init.shape[0], 1), self.cfg.load.thermal.initial_temperature, device=self.device)
        t_min, t_max = self.cfg.load.temperature_bounds
        return torch.clamp(t0, min=t_min, max=t_max)

    def _default_displacement_bc_fn(self, xyt_bc: torch.Tensor, bc_labels: torch.Tensor) -> torch.Tensor:
        u_target = torch.zeros((xyt_bc.shape[0], 2), device=self.device)
        u0 = self.cfg.load.mechanical.prescribed_displacement
        load_factor = self._load_time_factor(xyt_bc[:, 2:3])

        for tag, constrained in self.cfg.load.mechanical.displacement_constraints.items():
            tag_id = self.sampler.BOUNDARY_TAG_TO_ID[tag]
            mask = bc_labels.squeeze(-1) == tag_id
            base = self.cfg.load.mechanical.boundary_displacement.get(tag, u0)
            amp = self.cfg.load.mechanical.boundary_displacement_amplitude.get(tag, (1.0, 1.0))
            ux = base[0] + amp[0] * load_factor[mask, 0]
            uy = base[1] + amp[1] * load_factor[mask, 0]
            if constrained[0]:
                u_target[mask, 0] = ux
            if constrained[1]:
                u_target[mask, 1] = uy

        (ux_min, uy_min), (ux_max, uy_max) = self.cfg.load.displacement_bounds
        u_target[:, 0] = torch.clamp(u_target[:, 0], min=ux_min, max=ux_max)
        u_target[:, 1] = torch.clamp(u_target[:, 1], min=uy_min, max=uy_max)
        return u_target

    def _build_batch(
        self,
        t: float,
        temperature_bc_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        temperature_init_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        displacement_bc_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        c = self.cfg.train
        bc_T, bc_T_labels = self.sampler.sample_boundary(c.n_boundary, t)
        bc_u, bc_u_labels = self.sampler.sample_boundary(c.n_boundary, t)
        split_bc_T = self.sampler.sample_split_boundaries(c.n_boundary, t, self.cfg.load.thermal.thermal_bc_tags)
        split_bc_u = self.sampler.sample_split_boundaries(
            c.n_boundary,
            t,
            tuple(self.cfg.load.mechanical.displacement_constraints.keys()),
        )
        xyt_init = self.sampler.sample_initial(c.n_initial, self.cfg.train.t0)

        batch = {
            "domain": self.sampler.sample_domain(c.n_domain, t),
            "bc_T": bc_T,
            "bc_T_labels": bc_T_labels,
            "bc_u": bc_u,
            "bc_u_labels": bc_u_labels,
            "bc_T_split": split_bc_T,
            "bc_u_split": split_bc_u,
            "init": xyt_init,
        }
        q, w_q = self.sampler.sample_quadrature(c.n_quadrature, t)
        batch["quad"] = q
        batch["w_q"] = w_q

        temperature_bc_fn = temperature_bc_fn or self._default_temperature_bc_fn
        temperature_init_fn = temperature_init_fn or self._default_temperature_init_fn
        displacement_bc_fn = displacement_bc_fn or self._default_displacement_bc_fn

        batch["T_bar"] = temperature_bc_fn(batch["bc_T"], batch["bc_T_labels"])
        batch["T0"] = temperature_init_fn(batch["init"])
        batch["u_bar"] = displacement_bc_fn(batch["bc_u"], batch["bc_u_labels"])

        batch["T_bar_split"] = {
            tag: temperature_bc_fn(pts, labels)
            for tag, (pts, labels) in batch["bc_T_split"].items()
        }
        batch["u_bar_split"] = {
            tag: displacement_bc_fn(pts, labels)
            for tag, (pts, labels) in batch["bc_u_split"].items()
        }
        return batch

    def _run_adam(self, optimizer, closure_fn, epochs: int):
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = closure_fn()
            loss.backward()
            optimizer.step()

    def _run_lbfgs(self, optimizer, closure_fn):
        def closure():
            optimizer.zero_grad()
            loss = closure_fn()
            loss.backward()
            return loss

        optimizer.step(closure)

    def train(self):
        mat = self.cfg.material
        c = self.cfg.train

        # Background quadrature state for history variables
        q0, _ = self.sampler.sample_quadrature(c.n_quadrature, c.t0)
        HI = torch.zeros((q0.shape[0], 1), device=self.device)
        HII = torch.zeros((q0.shape[0], 1), device=self.device)
        d_prev = torch.zeros((q0.shape[0], 1), device=self.device)

        history = []

        print(f"[CoupledTrainer] irreversibility strategy: {self.irreversibility}")

        for n in range(c.num_time_steps):
            t_np1 = c.t0 + (n + 1) * self.dt
            batch = self._build_batch(t_np1)

            # Step 1: train thermo-mechanical net with frozen d^n
            d_tm = d_prev

            opt_tu_adam = torch.optim.Adam(self.net_tu.parameters(), lr=c.adam_lr)

            def tu_loss_fn():
                loss, _ = thermo_mech_total_loss(
                    self.net_tu, batch, d_tm, mat, (c.w_T, c.w_u), mode=c.mech_mode
                )
                return loss

            self._run_adam(opt_tu_adam, tu_loss_fn, c.adam_epochs_tu)
            opt_tu_lbfgs = torch.optim.LBFGS(self.net_tu.parameters(), max_iter=c.lbfgs_iters_tu)
            self._run_lbfgs(opt_tu_lbfgs, tu_loss_fn)

            # Step 2: update history on quadrature points
            HI, HII, He = update_history_fields(self.net_tu, batch["quad"], HI, HII, mat)

            # Step 3: train phase-field net with fixed H_e^{n+1}
            opt_d_adam = torch.optim.Adam(self.net_d.parameters(), lr=c.adam_lr)

            def d_loss_fn():
                loss, _ = phasefield_loss(
                    self.net_d,
                    batch,
                    d_prev,
                    He,
                    mat,
                    self.dt,
                    irreversibility=self.irreversibility,
                )
                return loss

            self._run_adam(opt_d_adam, d_loss_fn, c.adam_epochs_d)
            opt_d_lbfgs = torch.optim.LBFGS(self.net_d.parameters(), max_iter=c.lbfgs_iters_d)
            self._run_lbfgs(opt_d_lbfgs, d_loss_fn)

            with torch.no_grad():
                _, d_new = phasefield_loss(
                    self.net_d,
                    batch,
                    d_prev,
                    He,
                    mat,
                    self.dt,
                    irreversibility=self.irreversibility,
                )

            # Step 4: store and pass to next step
            d_prev = d_new.detach()
            history.append(TimeStepState(t=t_np1, d_q=d_prev, HI=HI, HII=HII, He=He))

        return history
