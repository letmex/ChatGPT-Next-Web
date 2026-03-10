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


class CoupledTrainer:
    def __init__(self, cfg: Config, domain: RectDomain):
        self.cfg = cfg
        self.device = torch.device(cfg.runtime.device)

        self.net_tu = ThermoMechNet().to(self.device)
        self.net_d = PhaseFieldNet().to(self.device)

        self.sampler = Sampler(domain, self.device)
        self.dt = (cfg.train.tf - cfg.train.t0) / cfg.train.num_time_steps

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

    def train(self):
        mat = self.cfg.material
        c = self.cfg.train

        # Background quadrature state for history variables
        q0, _ = self.sampler.sample_quadrature(c.n_quadrature, c.t0)
        HI = torch.zeros((q0.shape[0], 1), device=self.device)
        HII = torch.zeros((q0.shape[0], 1), device=self.device)
        d_prev = torch.zeros((q0.shape[0], 1), device=self.device)

        history = []

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
                loss, _ = phasefield_loss(self.net_d, batch, d_prev, He, mat, self.dt)
                return loss

            self._run_adam(opt_d_adam, d_loss_fn, c.adam_epochs_d)
            opt_d_lbfgs = torch.optim.LBFGS(self.net_d.parameters(), max_iter=c.lbfgs_iters_d)
            self._run_lbfgs(opt_d_lbfgs, d_loss_fn)

            with torch.no_grad():
                _, d_new = phasefield_loss(self.net_d, batch, d_prev, He, mat, self.dt)

            # Step 4: store and pass to next step
            d_prev = d_new.detach()
            history.append(TimeStepState(t=t_np1, d_q=d_prev, HI=HI, HII=HII, He=He))

        return history
