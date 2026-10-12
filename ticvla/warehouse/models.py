"""Torch-only reference and action modules; no simulator or VLM imports."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def spline_basis(num_controls, num_samples, degree=3):
    """Clamped uniform B-spline basis (partition of unity, endpoint interpolation)."""
    if num_controls <= degree or num_samples < 2:
        raise ValueError("Invalid B-spline dimensions")
    knots = torch.cat(
        (
            torch.zeros(degree + 1),
            torch.linspace(0, 1, num_controls - degree + 1)[1:-1],
            torch.ones(degree + 1),
        )
    )
    t = torch.linspace(0, 1, num_samples)
    basis = ((t[:, None] >= knots[:-1]) & (t[:, None] < knots[1:])).float()
    for order in range(1, degree + 1):
        columns = len(knots) - order - 1
        next_basis = torch.zeros(num_samples, columns)
        for j in range(columns):
            left, right = (
                knots[j + order] - knots[j],
                knots[j + order + 1] - knots[j + 1],
            )
            if left > 0:
                next_basis[:, j] += (t - knots[j]) / left * basis[:, j]
            if right > 0:
                next_basis[:, j] += (knots[j + order + 1] - t) / right * basis[:, j + 1]
        basis = next_basis
    basis[-1] = 0.0
    basis[-1, -1] = 1.0
    return basis


def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)


class ReferenceDecoder(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.reference_hidden_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, d), nn.SiLU()
        )
        self.control_head = nn.Linear(d, cfg.num_control_points * 2)
        self.length_head = nn.Linear(d, cfg.num_reference_points + 1)
        self.confidence_head = nn.Linear(d, 1)
        self.register_buffer(
            "basis",
            spline_basis(
                cfg.num_control_points, cfg.num_reference_points, cfg.spline_degree
            ),
        )

    def forward(self, h_ref):
        hidden = self.trunk(h_ref.to(self.control_head.weight.dtype))
        controls = (
            self.control_head(hidden).view(-1, self.cfg.num_control_points, 2)
            * self.cfg.coordinate_scale
        )
        points = torch.einsum("nc,bcd->bnd", self.basis, controls)
        length_logits = self.length_head(hidden)
        length = length_logits.argmax(-1)
        return {
            "reference_xy": points,
            "reference_mask": torch.arange(points.shape[1], device=points.device)[None]
            < length[:, None],
            "control_points": controls,
            "length_logits": length_logits,
            "confidence_logits": self.confidence_head(hidden).squeeze(-1),
        }

    def loss(self, output, target):
        known = target["reference_known"].bool()
        mask = target["reference_mask"].bool() & known[:, None]
        points = output["reference_xy"]
        length = target["reference_mask"].sum(-1).long()
        losses = {
            "points": masked_mean(
                F.smooth_l1_loss(
                    points / self.cfg.coordinate_scale,
                    target["reference_xy"] / self.cfg.coordinate_scale,
                    reduction="none",
                ).mean(-1),
                mask,
            ),
            "length": masked_mean(
                F.cross_entropy(output["length_logits"], length, reduction="none"),
                known,
            ),
            "validity": masked_mean(
                F.binary_cross_entropy_with_logits(
                    output["confidence_logits"],
                    target["reference_valid"].float(),
                    reduction="none",
                ),
                known,
            ),
        }
        indices = (
            (
                torch.linspace(0, 1, 5, device=points.device)[None]
                * (length - 1).clamp_min(0)[:, None]
            )
            .round()
            .long()
        )
        pred = points.gather(1, indices[..., None].expand(-1, -1, 2))
        truth = target["reference_xy"].gather(1, indices[..., None].expand(-1, -1, 2))
        losses["anchors"] = masked_mean(
            F.smooth_l1_loss(
                pred / self.cfg.coordinate_scale,
                truth / self.cfg.coordinate_scale,
                reduction="none",
            ).mean((1, 2)),
            known & (length > 0) & target["text_valid"].bool(),
        )
        return losses


def pack_last_kv(cache):
    """Support tuple caches and Hugging Face DynamicCache without hardcoded heads."""
    if isinstance(cache, (tuple, list)):
        key, value = cache[-1]
    elif hasattr(cache, "layers"):
        key, value = cache.layers[-1].keys, cache.layers[-1].values
    elif hasattr(cache, "key_cache"):
        key, value = cache.key_cache[-1], cache.value_cache[-1]
    else:
        raise TypeError(f"Unsupported K/V cache {type(cache)}")
    if key.ndim != 4 or key.shape != value.shape:
        raise ValueError("Expected matching [B,KV_heads,S,head_dim] K/V")
    key = key.transpose(1, 2).flatten(2)
    value = value.transpose(1, 2).flatten(2)
    return torch.cat((key, value), -1)


class ContextCompressor(nn.Module):
    def __init__(self, kv_dim, cfg):
        super().__init__()
        d = cfg.hidden_dim
        self.project = nn.Sequential(
            nn.LayerNorm(kv_dim), nn.Linear(kv_dim, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.queries = nn.Parameter(torch.randn(cfg.context_queries, d) * 0.02)
        self.attention = nn.MultiheadAttention(d, cfg.num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d * 4), nn.SiLU(), nn.Linear(d * 4, d)
        )

    def forward(self, kv, valid):
        if kv.shape[:2] != valid.shape:
            raise ValueError("K/V mask mismatch")
        tokens = self.project(kv.to(self.queries.dtype))
        # An explicit zero null token prevents all-masked attention NaNs.
        empty = ~valid.any(-1)
        null_token = tokens.new_zeros(tokens.shape[0], 1, tokens.shape[-1])
        tokens = torch.cat((tokens, null_token), 1)
        valid = torch.cat((valid, empty[:, None]), 1)
        q = self.queries[None].expand(kv.shape[0], -1, -1)
        q = self.norm(
            q
            + self.attention(
                q, tokens, tokens, key_padding_mask=~valid, need_weights=False
            )[0]
        )
        return q + self.ffn(q)


class ConditionEncoder(nn.Module):
    def __init__(self, visual_dim, kv_dim, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.visual = nn.Linear(visual_dim, d)
        self.bev = nn.Sequential(
            nn.Conv2d(4, 32, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.bev_project = nn.Linear(64, d)
        self.bev_position = nn.Parameter(torch.randn(16, d) * 0.02)
        self.visual_position = nn.Parameter(torch.randn(256, d) * 0.02)
        self.state = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, d))
        self.reference = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, d))
        self.reference_position = nn.Parameter(
            torch.randn(cfg.num_reference_points, d) * 0.02
        )
        self.delay = nn.Sequential(nn.Linear(5, d), nn.SiLU(), nn.Linear(d, d))
        self.compressor = ContextCompressor(kv_dim, cfg)
        self.previous_plan = nn.Sequential(nn.Linear(3, d), nn.SiLU(), nn.Linear(d, d))
        self.types = nn.Parameter(torch.randn(7, d) * 0.02)

    def forward(self, batch):
        visual = batch["visual"].to(self.visual.weight.dtype)
        if visual.shape[1] > len(self.visual_position):
            raise ValueError("MVP expects one current RGB tile (<=256 tokens)")
        rgb = (
            self.visual(visual)
            + self.visual_position[: visual.shape[1]]
            + self.types[0]
        )
        bev = self.bev(batch["bev"]).flatten(2).transpose(1, 2)
        bev = self.bev_project(bev) + self.bev_position + self.types[1]
        # v, omega, their availability, footprint radius, forward-motion indicator.
        state_input = torch.cat(
            (
                batch["state"],
                batch["state_mask"].float(),
                torch.full_like(batch["state"][:, :1], self.cfg.footprint_radius),
                torch.ones_like(batch["state"][:, :1]),
            ),
            -1,
        )
        state = self.state(state_input)[:, None] + self.types[2]
        ref = batch["reference_xy"] / self.cfg.coordinate_scale
        delta = torch.diff(ref, dim=1, prepend=ref[:, :1])
        arc = delta.norm(dim=-1).cumsum(-1)
        tangent = F.normalize(delta, dim=-1)
        features = torch.cat(
            (
                ref,
                arc[..., None],
                tangent,
                batch["reference_confidence"][:, None, None].expand(
                    -1, ref.shape[1], 1
                ),
            ),
            -1,
        )
        ref_tokens = self.reference(features) + self.reference_position + self.types[3]
        sem = self.compressor(batch["kv"], batch["kv_mask"]) + self.types[4]
        delay = self.delay(batch["delay"])[:, None] + self.types[5]
        previous = batch["previous_plan"].clone()
        previous[:, :, :2] /= self.cfg.coordinate_scale
        previous[:, :, 2] /= self.cfg.horizon * self.cfg.action_dt
        previous = self.previous_plan(previous) + self.types[6]
        B = ref.shape[0]
        tokens = torch.cat((rgb, bev, state, ref_tokens, sem, delay, previous), 1)
        mask = torch.cat(
            (
                batch["visual_mask"],
                batch["map_valid"][:, None].expand(B, 16),
                torch.ones(B, 1, device=ref.device, dtype=torch.bool),
                batch["reference_mask"],
                batch["semantic_valid"][:, None].expand(B, sem.shape[1]),
                torch.ones(B, 1, device=ref.device, dtype=torch.bool),
                batch["previous_plan_mask"],
            ),
            1,
        )
        return tokens, mask


class FlowBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.norms = nn.ModuleList(
            [nn.LayerNorm(d, elementwise_affine=False) for _ in range(3)]
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        self.self_attention = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))

    def forward(self, x, context, valid, condition):
        params = self.modulation(condition).chunk(6, -1)

        def mod(index):
            return (
                self.norms[index](x) * (1 + params[2 * index][:, None])
                + params[2 * index + 1][:, None]
            )

        z = mod(0)
        x = x + self.self_attention(z, z, z, need_weights=False)[0]
        x = (
            x
            + self.cross_attention(
                mod(1), context, context, key_padding_mask=~valid, need_weights=False
            )[0]
        )
        return x + self.ffn(mod(2))


class FlowActionExpert(nn.Module):
    def __init__(self, visual_dim, kv_dim, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.context_encoder = ConditionEncoder(visual_dim, kv_dim, cfg)
        self.action = nn.Linear(2, d)
        self.physical_time = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.flow_time = nn.Sequential(nn.Linear(3, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(
            [FlowBlock(d, cfg.num_heads) for _ in range(cfg.num_layers)]
        )
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

    def velocity(self, x, tau, context, valid):
        physical = (
            torch.arange(1, self.cfg.horizon + 1, device=x.device, dtype=x.dtype)[
                None, :, None
            ]
            * self.cfg.action_dt
        )
        tokens = self.action(x) + self.physical_time(physical)
        condition = self.flow_time(
            torch.stack((tau, torch.sin(math.pi * tau), torch.cos(math.pi * tau)), -1)
        )
        for block in self.blocks:
            tokens = block(tokens, context, valid, condition)
        return self.head(tokens)

    def loss(self, batch, generator=None):
        context, valid = self.context_encoder(batch)
        target = batch["actions"] / self.cfg.coordinate_scale
        noise = torch.randn(
            target.shape, device=target.device, dtype=target.dtype, generator=generator
        )
        tau = torch.rand(len(target), device=target.device, generator=generator)
        # Missing future timesteps have no data and must not leak padded target values.
        mask = batch["action_mask"]
        target = torch.where(mask[..., None], target, torch.zeros_like(target))
        x = (1 - tau[:, None, None]) * noise + tau[:, None, None] * target
        pred = self.velocity(x, tau, context, valid)
        return masked_mean((pred - (target - noise)).square().mean(-1), mask)

    @torch.no_grad()
    def sample(self, batch, generator=None):
        context, valid = self.context_encoder(
            batch
        )  # encoded once, reused for all ODE steps
        x = torch.randn(
            len(context),
            self.cfg.horizon,
            2,
            device=context.device,
            generator=generator,
        )
        for step in range(self.cfg.flow_steps):
            tau = x.new_full((len(x),), step / self.cfg.flow_steps)
            x = x + self.velocity(x, tau, context, valid) / self.cfg.flow_steps
        return {"trajectories": (x * self.cfg.coordinate_scale)[:, None, None]}


class QueryActionExpert(nn.Module):
    """Matched-condition regression control: same perception/KV/reference encoder."""

    def __init__(self, visual_dim, kv_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = ConditionEncoder(visual_dim, kv_dim, cfg)
        self.queries = nn.Parameter(torch.randn(cfg.horizon, cfg.hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            cfg.hidden_dim,
            cfg.num_heads,
            4 * cfg.hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, cfg.num_layers)
        self.head = nn.Linear(cfg.hidden_dim, 2)

    def forward(self, batch):
        context, valid = self.context_encoder(batch)
        query = self.queries[None].expand(len(context), -1, -1)
        return (
            self.head(self.decoder(query, context, memory_key_padding_mask=~valid))
            * self.cfg.coordinate_scale
        )

    def loss(self, batch, generator=None):
        return masked_mean(
            F.smooth_l1_loss(
                self(batch) / self.cfg.coordinate_scale,
                batch["actions"] / self.cfg.coordinate_scale,
                reduction="none",
            ).mean(-1),
            batch["action_mask"],
        )

    @torch.no_grad()
    def sample(self, batch, generator=None):
        return {"trajectories": self(batch)[:, None, None]}
