from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import torch

from .config import Config, MaterialConfig
from .history_update import update_history_fields


class _SyntheticThermoMechNet(torch.nn.Module):
    def __init__(self, t_ref: float):
        super().__init__()
        self.t_ref = t_ref

    def forward(self, xyt: torch.Tensor):
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]

        T = self.t_ref + 40.0 * t
        u_x = 1.8e-3 * t * x + 6.0e-4 * t * y
        u_y = -7.5e-4 * t * x + 1.2e-3 * t * y
        u = torch.cat([u_x, u_y], dim=1)
        return T, u


def run_eps_r_sensitivity(
    eps_values: List[float] | None = None,
    n_points: int = 128,
    n_steps: int = 12,
    crack_threshold_scale: float = 0.25,
    seed: int = 0,
) -> List[Dict[str, float]]:
    if eps_values is None:
        eps_values = [1e-4, 1e-5, 1e-6]

    torch.manual_seed(seed)
    cfg = Config()
    base_mat: MaterialConfig = deepcopy(cfg.material)

    x = torch.rand((n_points, 1))
    y = torch.rand((n_points, 1))

    results = []
    crack_threshold = crack_threshold_scale * base_mat.GcI / max(base_mat.l0, 1e-12)

    for eps_r in eps_values:
        mat = deepcopy(base_mat)
        mat.eps_r = float(eps_r)

        net = _SyntheticThermoMechNet(t_ref=mat.T_ref)

        HI = torch.zeros((n_points, 1))
        HII = torch.zeros((n_points, 1))

        peak_hi = 0.0
        peak_hii = 0.0
        first_crack_t = None

        for step in range(1, n_steps + 1):
            t = step / n_steps
            xyt_q = torch.cat([x, y, torch.full_like(x, t)], dim=1)
            HI, HII, He = update_history_fields(net, xyt_q, HI, HII, mat)

            peak_hi = max(peak_hi, HI.max().item())
            peak_hii = max(peak_hii, HII.max().item())

            if first_crack_t is None and He.max().item() >= crack_threshold:
                first_crack_t = t

        results.append(
            {
                "eps_r": float(eps_r),
                "HI_peak": peak_hi,
                "HII_peak": peak_hii,
                "first_crack_t": float(first_crack_t) if first_crack_t is not None else float("nan"),
            }
        )

    return results


def format_results_table(rows: List[Dict[str, float]]) -> str:
    header = "| eps_r | HI_peak | HII_peak | first_crack_t |\n|---:|---:|---:|---:|"
    body = "\n".join(
        f"| {r['eps_r']:.0e} | {r['HI_peak']:.6e} | {r['HII_peak']:.6e} | {r['first_crack_t']:.6f} |"
        for r in rows
    )
    return f"{header}\n{body}"


if __name__ == "__main__":
    rows = run_eps_r_sensitivity()
    print(format_results_table(rows))
