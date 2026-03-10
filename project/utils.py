import torch


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def lame_constants(E: float, nu: float):
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return lam, mu


def degradation(d: torch.Tensor, kappa: float):
    return (1.0 - d) ** 2 + kappa


def positive(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + torch.abs(x))


def elastic_strain_plane_stress(u: torch.Tensor, T: torch.Tensor, xyt: torch.Tensor, alpha: float, T_ref: float, nu: float):
    ux = u[:, 0:1]
    uy = u[:, 1:2]
    gux = grad(ux, xyt)
    guy = grad(uy, xyt)

    exx = gux[:, 0:1] - alpha * (T - T_ref)
    eyy = guy[:, 1:2] - alpha * (T - T_ref)
    exy = 0.5 * (gux[:, 1:2] + guy[:, 0:1])
    ezz = -nu / (1 - nu) * (exx + eyy)
    return exx, eyy, exy, ezz
