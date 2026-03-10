import torch

from .utils import elastic_strain_plane_stress, lame_constants, positive


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
