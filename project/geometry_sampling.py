from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class RectDomain:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class Sampler:
    """Sampling helper for interior, boundary and quadrature points."""

    BOUNDARY_TAG_TO_ID: Dict[str, int] = {"left": 0, "right": 1, "bottom": 2, "top": 3}

    def __init__(self, domain: RectDomain, device: torch.device):
        self.domain = domain
        self.device = device

    def _rand_xy(self, n: int) -> torch.Tensor:
        x = torch.rand(n, 1, device=self.device) * (self.domain.x_max - self.domain.x_min) + self.domain.x_min
        y = torch.rand(n, 1, device=self.device) * (self.domain.y_max - self.domain.y_min) + self.domain.y_min
        return torch.cat([x, y], dim=1)

    def sample_domain(self, n: int, t: float) -> torch.Tensor:
        xy = self._rand_xy(n)
        tt = torch.full((n, 1), float(t), device=self.device)
        return torch.cat([xy, tt], dim=1)

    def sample_initial(self, n: int, t0: float) -> torch.Tensor:
        return self.sample_domain(n, t0)

    def sample_boundary(self, n: int, t: float):
        n_side = max(n // 4, 1)
        ys = torch.rand(n_side, 1, device=self.device) * (self.domain.y_max - self.domain.y_min) + self.domain.y_min
        xs = torch.rand(n_side, 1, device=self.device) * (self.domain.x_max - self.domain.x_min) + self.domain.x_min

        left = torch.cat([torch.full_like(ys, self.domain.x_min), ys], dim=1)
        right = torch.cat([torch.full_like(ys, self.domain.x_max), ys], dim=1)
        bottom = torch.cat([xs, torch.full_like(xs, self.domain.y_min)], dim=1)
        top = torch.cat([xs, torch.full_like(xs, self.domain.y_max)], dim=1)

        xy = torch.cat([left, right, bottom, top], dim=0)
        tt = torch.full((xy.shape[0], 1), float(t), device=self.device)

        labels = torch.cat(
            [
                torch.full((n_side, 1), self.BOUNDARY_TAG_TO_ID["left"], device=self.device, dtype=torch.long),
                torch.full((n_side, 1), self.BOUNDARY_TAG_TO_ID["right"], device=self.device, dtype=torch.long),
                torch.full((n_side, 1), self.BOUNDARY_TAG_TO_ID["bottom"], device=self.device, dtype=torch.long),
                torch.full((n_side, 1), self.BOUNDARY_TAG_TO_ID["top"], device=self.device, dtype=torch.long),
            ],
            dim=0,
        )
        return torch.cat([xy, tt], dim=1), labels


    def sample_boundary_by_tag(self, n: int, t: float, tag: str):
        if tag not in self.BOUNDARY_TAG_TO_ID:
            raise ValueError(f"Unknown boundary tag: {tag}")

        n_pts = max(int(n), 1)
        if tag in {"left", "right"}:
            y = torch.rand(n_pts, 1, device=self.device) * (self.domain.y_max - self.domain.y_min) + self.domain.y_min
            x_val = self.domain.x_min if tag == "left" else self.domain.x_max
            x = torch.full_like(y, x_val)
        else:
            x = torch.rand(n_pts, 1, device=self.device) * (self.domain.x_max - self.domain.x_min) + self.domain.x_min
            y_val = self.domain.y_min if tag == "bottom" else self.domain.y_max
            y = torch.full_like(x, y_val)

        xy = torch.cat([x, y], dim=1)
        tt = torch.full((n_pts, 1), float(t), device=self.device)
        labels = torch.full((n_pts, 1), self.BOUNDARY_TAG_TO_ID[tag], device=self.device, dtype=torch.long)
        return torch.cat([xy, tt], dim=1), labels

    def sample_split_boundaries(self, n: int, t: float, tags: tuple[str, ...] | None = None):
        tags = tags or tuple(self.BOUNDARY_TAG_TO_ID.keys())
        n_side = max(n // max(len(tags), 1), 1)
        out: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for tag in tags:
            out[tag] = self.sample_boundary_by_tag(n_side, t, tag)
        return out

    def sample_quadrature(self, n: int, t: float):
        # Simple MC quadrature points + unit weights scaled by area
        pts = self.sample_domain(n, t)
        area = (self.domain.x_max - self.domain.x_min) * (self.domain.y_max - self.domain.y_min)
        w = torch.full((n, 1), area / n, device=self.device)
        return pts, w

    def sample_adaptive(self, base_pts: torch.Tensor, indicator: torch.Tensor, n_refine: int) -> torch.Tensor:
        if base_pts.numel() == 0:
            return base_pts
        _, idx = torch.topk(indicator.squeeze(-1), k=min(n_refine, base_pts.shape[0]))
        seeds = base_pts[idx]
        noise = 0.01 * torch.randn_like(seeds[:, :2])
        xy = seeds[:, :2] + noise
        xy[:, 0] = torch.clamp(xy[:, 0], self.domain.x_min, self.domain.x_max)
        xy[:, 1] = torch.clamp(xy[:, 1], self.domain.y_min, self.domain.y_max)
        return torch.cat([xy, seeds[:, 2:3]], dim=1)
