from dataclasses import dataclass
from inspect import signature
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


@dataclass
class QuadState:
    xyt_q: torch.Tensor
    w_q: torch.Tensor
    d_prev: torch.Tensor
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

        self.irreversibility = getattr(cfg.train, "irreversibility", True)
        self._phasefield_supports_irreversibility = "irreversibility" in signature(phasefield_loss).parameters
        self._thermo_mech_supports_mode = "mode" in signature(thermo_mech_total_loss).parameters

        # Fixed primary quadrature grid for state variables.
        q0, w0 = self.sampler.sample_quadrature(cfg.train.n_quadrature, cfg.train.t0)
        zeros = torch.zeros((q0.shape[0], 1), device=self.device)
        self.quad_state = QuadState(
            xyt_q=q0,
            w_q=w0,
            d_prev=zeros.clone(),
            HI=zeros.clone(),
            HII=zeros.clone(),
            He=zeros.clone(),
        )

    def _sample_boundary(self, n: int, t: float) -> tuple[torch.Tensor, torch.Tensor | None]:
        sample = self.sampler.sample_boundary(n, t)
        if isinstance(sample, tuple):
            return sample
        return sample, None

    def _interp_heating_curve(self, t: torch.Tensor) -> torch.Tensor:
        load_cfg = getattr(self.cfg, "load", None)
        thermal_cfg = getattr(load_cfg, "thermal", None)
        if thermal_cfg is None:
            return torch.zeros_like(t)

        curve = sorted(getattr(thermal_cfg, "heating_curve", []), key=lambda x: x[0])
        init_temp = float(getattr(thermal_cfg, "initial_temperature", 0.0))
        if len(curve) == 0:
            return torch.full_like(t, init_temp)
        if len(curve) == 1:
            return torch.full_like(t, float(curve[0][1]))

        t_out = torch.full_like(t, float(curve[0][1]))
        t_out[t >= curve[-1][0]] = float(curve[-1][1])

        for i in range(len(curve) - 1):
            ta, va = curve[i]
            tb, vb = curve[i + 1]
            seg = (t >= ta) & (t < tb)
            if torch.any(seg):
                ratio = (t[seg] - ta) / max(tb - ta, 1e-12)
                t_out[seg] = va + ratio * (vb - va)

        return t_out

    def _default_temperature_bc_fn(self, xyt_bc: torch.Tensor, bc_labels: torch.Tensor | None) -> torch.Tensor:
        load_cfg = getattr(self.cfg, "load", None)
        thermal_cfg = getattr(load_cfg, "thermal", None)
        if thermal_cfg is None:
            return torch.zeros((xyt_bc.shape[0], 1), device=self.device)

        t_curve = self._interp_heating_curve(xyt_bc[:, 2:3])
        tags = getattr(thermal_cfg, "thermal_bc_tags", None)
        tag_map = getattr(self.sampler, "BOUNDARY_TAG_TO_ID", None)

        # No labels/tags available: apply curve on all sampled boundary points.
        if bc_labels is None or not tags or tag_map is None:
            return t_curve

        target = torch.zeros((xyt_bc.shape[0], 1), device=self.device)
        for tag in tags:
            if tag not in tag_map:
                continue
            mask = bc_labels.squeeze(-1) == tag_map[tag]
            target[mask] = t_curve[mask]
        return target

    def _default_temperature_init_fn(self, xyt_init: torch.Tensor) -> torch.Tensor:
        load_cfg = getattr(self.cfg, "load", None)
        thermal_cfg = getattr(load_cfg, "thermal", None)
        init_temp = float(getattr(thermal_cfg, "initial_temperature", 0.0))
        return torch.full((xyt_init.shape[0], 1), init_temp, device=self.device)

    def _default_displacement_bc_fn(self, xyt_bc: torch.Tensor, bc_labels: torch.Tensor | None) -> torch.Tensor:
        load_cfg = getattr(self.cfg, "load", None)
        mech_cfg = getattr(load_cfg, "mechanical", None)
        constraints = getattr(mech_cfg, "displacement_constraints", None)
        tag_map = getattr(self.sampler, "BOUNDARY_TAG_TO_ID", None)

        if bc_labels is None or mech_cfg is None or constraints is None or tag_map is None:
            return torch.zeros((xyt_bc.shape[0], 2), device=self.device)

        target = torch.zeros((xyt_bc.shape[0], 2), device=self.device)
        u0 = getattr(mech_cfg, "prescribed_displacement", (0.0, 0.0))
        for tag, constrained in constraints.items():
            if tag not in tag_map:
                continue
            mask = bc_labels.squeeze(-1) == tag_map[tag]
            if constrained[0]:
                target[mask, 0] = u0[0]
            if constrained[1]:
                target[mask, 1] = u0[1]
        return target

    def _build_batch(
        self,
        t: float,
        temperature_bc_fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor] | None = None,
        temperature_init_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        displacement_bc_fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        c = self.cfg.train
        bc_T, bc_T_labels = self._sample_boundary(c.n_boundary, t)
        bc_u, bc_u_labels = self._sample_boundary(c.n_boundary, t)

        batch = {
            "domain": self.sampler.sample_domain(c.n_domain, t),
            "bc_T": bc_T,
            "bc_T_labels": bc_T_labels,
            "bc_u": bc_u,
            "bc_u_labels": bc_u_labels,
            "init": self.sampler.sample_initial(c.n_initial, self.cfg.train.t0),
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
        return batch

    def _phasefield_batch(self, t: float) -> Dict[str, torch.Tensor]:
        # Fixed primary points: do not overwrite this state grid with adaptive temp points.
        q = self.quad_state.xyt_q.clone()
        q[:, 2] = t
        return {"quad": q, "w_q": self.quad_state.w_q}

    def _thermo_mech_loss(self, batch: Dict[str, torch.Tensor], d_prev: torch.Tensor, mat, weights):
        mech_mode = getattr(self.cfg.train, "mech_mode", None)
        if mech_mode is not None and self._thermo_mech_supports_mode:
            return thermo_mech_total_loss(self.net_tu, batch, d_prev, mat, weights, mode=mech_mode)
        return thermo_mech_total_loss(self.net_tu, batch, d_prev, mat, weights)

    def _phasefield_loss(self, batch: Dict[str, torch.Tensor], d_prev: torch.Tensor, He: torch.Tensor):
        if self._phasefield_supports_irreversibility:
            return phasefield_loss(
                self.net_d,
                batch,
                d_prev,
                He,
                self.cfg.material,
                self.dt,
                irreversibility=self.irreversibility,
            )
        return phasefield_loss(self.net_d, batch, d_prev, He, self.cfg.material, self.dt)

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
        c = self.cfg.train
        mat = self.cfg.material
        history = []

        for n in range(c.num_time_steps):
            t_np1 = c.t0 + (n + 1) * self.dt
            batch = self._build_batch(t_np1)
            pf_batch = self._phasefield_batch(t_np1)

            d_tm = self.quad_state.d_prev

            opt_tu_adam = torch.optim.Adam(self.net_tu.parameters(), lr=c.adam_lr)

            def tu_loss_fn():
                loss, _ = self._thermo_mech_loss(batch, d_tm, mat, (c.w_T, c.w_u))
                return loss

            self._run_adam(opt_tu_adam, tu_loss_fn, c.adam_epochs_tu)
            self._run_lbfgs(torch.optim.LBFGS(self.net_tu.parameters(), max_iter=c.lbfgs_iters_tu), tu_loss_fn)

            HI, HII, He = update_history_fields(
                self.net_tu,
                pf_batch["quad"],
                self.quad_state.HI,
                self.quad_state.HII,
                mat,
            )
            self.quad_state.HI = HI
            self.quad_state.HII = HII
            self.quad_state.He = He

            opt_d_adam = torch.optim.Adam(self.net_d.parameters(), lr=c.adam_lr)

            def d_loss_fn():
                loss, _ = self._phasefield_loss(pf_batch, self.quad_state.d_prev, self.quad_state.He)
                return loss

            self._run_adam(opt_d_adam, d_loss_fn, c.adam_epochs_d)
            self._run_lbfgs(torch.optim.LBFGS(self.net_d.parameters(), max_iter=c.lbfgs_iters_d), d_loss_fn)

            with torch.no_grad():
                _, d_new = self._phasefield_loss(pf_batch, self.quad_state.d_prev, self.quad_state.He)

            self.quad_state.d_prev = d_new.detach()
            history.append(
                TimeStepState(
                    t=t_np1,
                    d_q=self.quad_state.d_prev,
                    HI=self.quad_state.HI,
                    HII=self.quad_state.HII,
                    He=self.quad_state.He,
                )
            )

        return history
