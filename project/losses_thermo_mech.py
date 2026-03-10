import torch

from .utils import degradation, elastic_strain_plane_stress, grad, lame_constants, positive


def heat_residual_loss(net_tu, xyt_domain, d_prev, mat):
    xyt_domain = xyt_domain.requires_grad_(True)
    T, _ = net_tu(xyt_domain)

    g = degradation(d_prev, mat.kappa)
    k_eff = g * mat.k0

    gT = grad(T, xyt_domain)
    T_t = gT[:, 2:3]
    T_x = gT[:, 0:1]
    T_y = gT[:, 1:2]

    qx = k_eff * T_x
    qy = k_eff * T_y

    div_q = grad(qx, xyt_domain)[:, 0:1] + grad(qy, xyt_domain)[:, 1:2]
    res = mat.rho * mat.c_p * T_t - div_q - mat.Q
    return torch.mean(res**2)


def temperature_bc_loss(net_tu, xyt_bc, T_bar):
    T, _ = net_tu(xyt_bc)
    return torch.mean((T - T_bar) ** 2)


def temperature_init_loss(net_tu, xyt_init, T0):
    T, _ = net_tu(xyt_init)
    return torch.mean((T - T0) ** 2)


def strain_energies(exx, eyy, exy, ezz, lam, mu, eps_r):
    tr_e = exx + eyy + ezz
    tr_p = positive(tr_e)

    em = 0.5 * (exx + eyy)
    ed = 0.5 * (exx - eyy)
    r = torch.sqrt(ed**2 + exy**2 + eps_r**2)
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

    psi_plus = psi_I + psi_II
    tr_m = tr_e - tr_p
    psi_full = 0.5 * lam * tr_e**2 + mu * (exx**2 + eyy**2 + ezz**2 + 2.0 * exy**2)
    psi_minus = psi_full - psi_plus
    return psi_plus, psi_minus


def mechanical_potential_loss(net_tu, xyt_q, w_q, d_prev, mat):
    xyt_q = xyt_q.requires_grad_(True)
    T, u = net_tu(xyt_q)
    lam, mu = lame_constants(mat.E, mat.nu)

    exx, eyy, exy, ezz = elastic_strain_plane_stress(u, T, xyt_q, mat.alpha, mat.T_ref, mat.nu)
    psi_plus, psi_minus = strain_energies(exx, eyy, exy, ezz, lam, mu, mat.eps_r)

    g = degradation(d_prev, mat.kappa)
    density = g * psi_plus + psi_minus
    Pi_u = torch.sum(w_q * density)
    return Pi_u


def displacement_bc_loss(net_tu, xyt_bc, u_bar):
    _, u = net_tu(xyt_bc)
    return torch.mean((u - u_bar) ** 2)


def thermo_mech_total_loss(net_tu, batch, d_prev, mat, weights):
    L_heat = (
        heat_residual_loss(net_tu, batch["domain"], d_prev, mat)
        + temperature_bc_loss(net_tu, batch["bc_T"], batch["T_bar"])
        + temperature_init_loss(net_tu, batch["init"], batch["T0"])
    )
    L_u = mechanical_potential_loss(net_tu, batch["quad"], batch["w_q"], d_prev, mat) + displacement_bc_loss(
        net_tu, batch["bc_u"], batch["u_bar"]
    )
    return weights[0] * L_heat + weights[1] * L_u, {"L_heat": L_heat, "L_u": L_u}
