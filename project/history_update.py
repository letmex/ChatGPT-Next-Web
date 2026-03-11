import torch

from .utils import elastic_strain_plane_stress, lame_constants, positive


def assert_history_consistency(
    xyt_q: torch.Tensor,
    xyt_prev: torch.Tensor | None,
    interpolated: bool,
    atol: float = 1e-8,
) -> None:
    """Ensure history tensors correspond to the current quadrature points."""
    if interpolated:
        return
    if xyt_prev is None:
        return
    if xyt_prev.shape != xyt_q.shape:
        raise ValueError("History/quad shape mismatch without interpolation.")
    if not torch.allclose(xyt_prev[:, :2], xyt_q[:, :2], atol=atol, rtol=0.0):
        raise ValueError("Quadrature spatial coordinates changed but history was not interpolated.")


def interpolate_history_to_quad(
    xyt_prev: torch.Tensor,
    HI_prev: torch.Tensor,
    HII_prev: torch.Tensor,
    xyt_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project history fields from old quadrature points to new points via nearest neighbor."""
    dist = torch.cdist(xyt_q[:, :2], xyt_prev[:, :2])
    nn_idx = torch.argmin(dist, dim=1)
    return HI_prev[nn_idx], HII_prev[nn_idx]


def update_history_fields(net_tu, xyt_q, HI_prev, HII_prev, mat):
    xyt_q = xyt_q.detach().requires_grad_(True)
    T, u = net_tu(xyt_q)
    lam, mu = lame_constants(mat.E, mat.nu)

    exx, eyy, exy, ezz = elastic_strain_plane_stress(u, T, xyt_q, mat.alpha, mat.T_ref, mat.nu)

    tr_e = exx + eyy + ezz
    tr_p = positive(tr_e)

    em = 0.5 * (exx + eyy)
    ed = 0.5 * (exx - eyy)
    r = torch.sqrt(ed**2 + exy**2 + mat.eps_r**2)
    e1 = em + r
    e2 = em - r
    e3 = ezz

    e1p = positive(e1)
    e2p = positive(e2)
    e3p = positive(e3)

    chi = ed / r
    eta = exy / r
    exxp = 0.5 * (e1p + e2p) + 0.5 * (e1p - e2p) * chi
    eyyp = 0.5 * (e1p + e2p) - 0.5 * (e1p - e2p) * chi
    exyp = 0.5 * (e1p - e2p) * eta
    ezzp = e3p

    ep2 = exxp**2 + eyyp**2 + ezzp**2 + 2.0 * exyp**2

    psi_I = 0.5 * lam * tr_p**2
    psi_II = mu * ep2

    HI = torch.maximum(HI_prev, psi_I)
    HII = torch.maximum(HII_prev, psi_II)
    He = HI + (mat.GcI / mat.GcII) * HII
    return HI.detach(), HII.detach(), He.detach()
