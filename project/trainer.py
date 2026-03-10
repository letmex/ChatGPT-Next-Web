from dataclasses import dataclass
from typing import Dict

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

        # Main state grid: sampled once and reused for all time steps.
        q0, w0 = self.sampler.sample_quadrature(cfg.train.n_quadrature, cfg.train.t0)
        zeros = torch.zeros((q0.shape[0], 1), device=self.device)
        self.quad_state = QuadState(xyt_q=q0, w_q=w0, d_prev=zeros.clone(), HI=zeros.clone(), HII=zeros.clone(), He=zeros.clone())

    def _build_batch(self, t: float) -> Dict[str, torch.Tensor]:
        c = self.cfg.train
        batch = {
            "domain": self.sampler.sample_domain(c.n_domain, t),
            "bc_T": self.sampler.sample_boundary(c.n_boundary, t),
            "bc_u": self.sampler.sample_boundary(c.n_boundary, t),
            "init": self.sampler.sample_initial(c.n_initial, self.cfg.train.t0),
        }
        q, w_q = self.sampler.sample_quadrature(c.n_quadrature, t)
        batch["quad"] = q
        batch["w_q"] = w_q

        # Placeholder BC/IC data hooks
        batch["T_bar"] = torch.zeros((batch["bc_T"].shape[0], 1), device=self.device)
        batch["T0"] = torch.zeros((batch["init"].shape[0], 1), device=self.device)
        batch["u_bar"] = torch.zeros((batch["bc_u"].shape[0], 2), device=self.device)
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

    def _phasefield_batch(self, t: float) -> Dict[str, torch.Tensor]:
        # Keep phase-field/history variables on a fixed quadrature grid.
        # If adaptive refinement is added later, temporary points should be used
        # only for loss estimation without overriding this main state grid.
        q_fixed = self.quad_state.xyt_q.clone()
        q_fixed[:, 2] = t
        return {"quad": q_fixed, "w_q": self.quad_state.w_q}

    def train(self):
        mat = self.cfg.material
        c = self.cfg.train

        history = []

        for n in range(c.num_time_steps):
            t_np1 = c.t0 + (n + 1) * self.dt
            batch = self._build_batch(t_np1)
            pf_batch = self._phasefield_batch(t_np1)

            # Step 1: train thermo-mechanical net with frozen d^n
            d_tm = self.quad_state.d_prev

            opt_tu_adam = torch.optim.Adam(self.net_tu.parameters(), lr=c.adam_lr)

            def tu_loss_fn():
                loss, _ = thermo_mech_total_loss(self.net_tu, batch, d_tm, mat, (c.w_T, c.w_u))
                return loss

            self._run_adam(opt_tu_adam, tu_loss_fn, c.adam_epochs_tu)
            opt_tu_lbfgs = torch.optim.LBFGS(self.net_tu.parameters(), max_iter=c.lbfgs_iters_tu)
            self._run_lbfgs(opt_tu_lbfgs, tu_loss_fn)

            # Step 2: update history on quadrature points
            HI, HII, He = update_history_fields(
                self.net_tu, pf_batch["quad"], self.quad_state.HI, self.quad_state.HII, mat
            )
            self.quad_state.HI = HI
            self.quad_state.HII = HII
            self.quad_state.He = He

            # Step 3: train phase-field net with fixed H_e^{n+1}
            opt_d_adam = torch.optim.Adam(self.net_d.parameters(), lr=c.adam_lr)

            def d_loss_fn():
                loss, _ = phasefield_loss(self.net_d, pf_batch, self.quad_state.d_prev, self.quad_state.He, mat, self.dt)
                return loss

            self._run_adam(opt_d_adam, d_loss_fn, c.adam_epochs_d)
            opt_d_lbfgs = torch.optim.LBFGS(self.net_d.parameters(), max_iter=c.lbfgs_iters_d)
            self._run_lbfgs(opt_d_lbfgs, d_loss_fn)

            with torch.no_grad():
                _, d_new = phasefield_loss(
                    self.net_d, pf_batch, self.quad_state.d_prev, self.quad_state.He, mat, self.dt
                )

            # Step 4: store and pass to next step
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
