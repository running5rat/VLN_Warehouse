"""v0.6 nine-mode shared conditional flow. No legacy checkpoint reinterpretation."""

import math
import torch
from torch import nn
from ..models import FlowBlock

MODES = [
    (a, b)
    for a in ("LEFT", "CENTER", "RIGHT")
    for b in ("STOP-YIELD", "FOLLOW", "PROCEED")
]


class FlowMatchingActionExpert(nn.Module):
    def __init__(self, visual_dim, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.visual = nn.Linear(visual_dim, d)
        self.bev = nn.Sequential(
            nn.Conv2d(4, 32, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, d, 3, 2, 1),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.state = nn.Linear(6, d)
        self.reference = nn.Linear(6, d)
        self.semantic = nn.Linear(d, d)
        self.delay = nn.Linear(5, d)
        self.previous = nn.Linear(3, d)
        self.types = nn.Parameter(torch.randn(7, d) * 0.02)
        self.positions = nn.Parameter(torch.randn(256, d) * 0.02)
        self.mode_tag = nn.Embedding(2, d)
        self.lateral = nn.Embedding(3, d)
        self.longitudinal = nn.Embedding(3, d)
        self.joint = nn.Linear(d, d, bias=False)
        self.action = nn.Linear(2, d)
        self.physical_time = nn.Linear(1, d)
        self.flow_time = nn.Sequential(nn.Linear(3, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(
            [FlowBlock(d, cfg.num_heads) for _ in range(cfg.num_layers)]
        )
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

    def encode(self, batch):
        rgb = self.visual(batch["visual"])
        rgb = rgb + self.positions[: rgb.shape[1]]
        bev = self.bev(batch["bev"]).flatten(2).transpose(1, 2) + self.positions[:16]
        state = torch.cat(
            (
                batch["state"],
                batch["state_mask"].float(),
                torch.full_like(batch["state"][:, :1], self.cfg.footprint_radius),
                batch["motion_direction"][:, None],
            ),
            -1,
        )
        xy = batch["reference_xy"] / self.cfg.coordinate_scale
        delta = torch.diff(xy, dim=1, prepend=xy[:, :1])
        tangent = torch.nn.functional.normalize(delta, dim=-1)
        ref = self.reference(
            torch.cat(
                (
                    xy,
                    delta.norm(dim=-1).cumsum(-1)[..., None],
                    tangent,
                    batch["reference_confidence"][:, None, None].expand(
                        -1, xy.shape[1], 1
                    ),
                ),
                -1,
            )
        )
        ref = ref + self.positions[: xy.shape[1]]
        sem = (
            self.semantic(batch["z_sem"])
            + self.mode_tag(batch["inference_mode"])[:, None]
        )
        previous = batch["previous_plan"].clone()
        previous[..., :2] /= self.cfg.coordinate_scale
        previous[..., 2] /= self.cfg.horizon * self.cfg.action_dt
        parts = [
            rgb,
            bev,
            self.state(state)[:, None],
            ref,
            sem,
            self.delay(batch["delay"])[:, None],
            self.previous(previous) + self.positions[: previous.shape[1]],
        ]
        B = len(rgb)
        masks = [
            batch["visual_mask"],
            batch["map_valid"][:, None].expand(B, 16),
            torch.ones(B, 1, dtype=torch.bool, device=rgb.device),
            batch["reference_mask"],
            batch["semantic_valid"][:, None].expand(B, sem.shape[1]),
            torch.ones(B, 1, dtype=torch.bool, device=rgb.device),
            batch["previous_plan_mask"],
        ]
        context = torch.cat([x + self.types[i] for i, x in enumerate(parts)], 1)
        valid = torch.cat(masks, 1)
        if not torch.isfinite(context).all():
            raise ValueError("Nonfinite Flow condition")
        return context, valid

    def velocity(self, x, tau, context, valid, modes):
        t = (
            torch.arange(1, self.cfg.horizon + 1, device=x.device, dtype=x.dtype)[
                None, :, None
            ]
            * self.cfg.action_dt
        )
        tokens = self.action(x) + self.physical_time(t)
        lat, lon = self.lateral(modes[:, 0]), self.longitudinal(modes[:, 1])
        cond = self.flow_time(
            torch.stack((tau, torch.sin(math.pi * tau), torch.cos(math.pi * tau)), -1)
        )
        cond = cond + lat + lon + self.joint(lat * lon)
        for block in self.blocks:
            tokens = block(tokens, context, valid, cond)
        return self.head(tokens)

    @torch.inference_mode()
    def sample(self, batch, generator=None, samples=1):
        if samples < 1:
            raise ValueError("samples must be positive")
        context, valid = self.encode(batch)
        B = len(context)
        context = context.repeat_interleave(9 * samples, 0)
        valid = valid.repeat_interleave(9 * samples, 0)
        modes = torch.tensor(
            [(i, j) for i in range(3) for j in range(3)], device=context.device
        )
        modes = modes.repeat_interleave(samples, 0).repeat(B, 1)
        x = torch.randn(
            B * 9 * samples,
            self.cfg.horizon,
            2,
            device=context.device,
            generator=generator,
        )
        for step in range(self.cfg.flow_steps):
            x += (
                self.velocity(
                    x,
                    x.new_full((len(x),), step / self.cfg.flow_steps),
                    context,
                    valid,
                    modes,
                )
                / self.cfg.flow_steps
            )
        return {
            "trajectories": (x * self.cfg.coordinate_scale).reshape(
                B, 9, samples, self.cfg.horizon, 2
            ),
            "modes": MODES,
            "quality_validated": False,
        }

    def loss(self, batch, targets, point_mask, candidate_valid, generator=None):
        """Conditional Flow Matching on verified candidates only.

        targets: [B,9,K,H,2], point_mask: [B,9,K,H], candidate_valid:
        [B,9,K]. Unknown/invalid modes are not negative action demonstrations.
        This trains the exact deployment module, without a legacy-head adapter.
        """
        expected = (len(batch["visual"]), 9, targets.shape[2], self.cfg.horizon, 2)
        if targets.shape != expected or point_mask.shape != targets.shape[:-1]:
            raise ValueError("Nine-mode target/mask shape mismatch")
        if candidate_valid.shape != targets.shape[:3] or point_mask.dtype != torch.bool or candidate_valid.dtype != torch.bool:
            raise ValueError("Candidate availability must be an explicit boolean mask")
        if not torch.isfinite(targets).all():
            raise ValueError("Nonfinite action labels")
        available = candidate_valid & point_mask.any(-1)
        index = available.nonzero()
        if not len(index):
            raise ValueError("No verified action targets; skip this minibatch")
        context, valid = self.encode(batch)
        b, mode, sample = index.unbind(-1)
        target = targets[b, mode, sample] / self.cfg.coordinate_scale
        mask = point_mask[b, mode, sample]
        noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
        tau = torch.rand(len(index), device=target.device, dtype=target.dtype, generator=generator)
        mixed = ((1-tau[:,None,None])*noise + tau[:,None,None]*target).masked_fill(~mask[...,None], 0)
        prediction = self.velocity(mixed, tau, context[b], valid[b], torch.stack((mode//3, mode%3), -1))
        error = (prediction - (target-noise)).square().sum(-1)
        loss = ((error * mask).sum(-1) / (2*mask.sum(-1))).mean()
        return loss
