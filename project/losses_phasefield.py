import torch

from .utils import degradation, grad


def irreversible_transform(d_hat: torch.Tensor, d_prev: torch.Tensor) -> torch.Tensor:
    return d_prev + (1.0 - d_prev) * torch.sigmoid(d_hat)


def history_only_transform(d_hat: torch.Tensor) -> torch.Tensor:
    return torch.clamp(d_hat, min=0.0, max=1.0)


def apply_irreversibility(d_hat: torch.Tensor, d_prev: torch.Tensor, strategy: str) -> torch.Tensor:
    if strategy == "output_transform":
        return irreversible_transform(d_hat, d_prev)
    if strategy == "history_only":
        return history_only_transform(d_hat)
    raise ValueError(f"Unsupported irreversibility strategy: {strategy}")


def phasefield_incremental_energy(net_d, xyt_q, w_q, d_prev, He, mat, dt, irreversibility: str = "output_transform"):
    xyt_q = xyt_q.requires_grad_(True)
    d_hat = net_d(xyt_q)
    d = apply_irreversibility(d_hat, d_prev, irreversibility)

    gd = degradation(d, mat.kappa)
    gd_term = gd * He

    grad_d = grad(d, xyt_q)
    grad_norm2 = grad_d[:, 0:1] ** 2 + grad_d[:, 1:2] ** 2

    crack_surface = mat.GcI * (0.5 * d**2 / mat.l0 + 0.5 * mat.l0 * grad_norm2)
    viscous = 0.5 * mat.eta_pf / dt * (d - d_prev) ** 2

    density = gd_term + crack_surface + viscous
    Pi_d = torch.sum(w_q * density)
    return Pi_d, d


def phasefield_loss(net_d, batch, d_prev, He, mat, dt, irreversibility: str = "output_transform"):
    return phasefield_incremental_energy(
        net_d,
        batch["quad"],
        batch["w_q"],
        d_prev,
        He,
        mat,
        dt,
        irreversibility=irreversibility,
    )
