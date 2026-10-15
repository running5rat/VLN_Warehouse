"""Nine-mode training adapter and no-update readiness checks.

Never manufacture zero semantic features in an actual-data readiness report.
"""

from dataclasses import asdict
from pathlib import Path
import torch
from torch import nn

from ..config import WarehouseConfig
from ..models import ContextCompressor
from ..navigation.flow import FlowMatchingActionExpert
from ..schema import file_sha256
from .core import write_json, read_json, validate_v2


class FlowTrainingAdapter(nn.Module):
    def __init__(self, visual_dim, kv_dim, cfg):
        super().__init__()
        self.compressor = ContextCompressor(kv_dim, cfg)
        self.flow = FlowMatchingActionExpert(visual_dim, cfg)
        self.cfg, self.visual_dim, self.kv_dim = cfg, visual_dim, kv_dim

    def loss(self, batch, kv, kv_valid, targets, point_mask, candidate_valid, generator=None):
        # Raw VLM K/V is frozen. The compressor and action-side projections train.
        condition = dict(batch, z_sem=self.compressor(kv.detach(), kv_valid))
        return self.flow.loss(condition, targets, point_mask, candidate_valid, generator)

    def save(self, path, provenance):
        torch.save(dict(architecture="warehouse-nine-mode-flow-v1", cfg=asdict(self.cfg),
                        visual_dim=self.visual_dim, kv_dim=self.kv_dim,
                        flow=self.flow.state_dict(), compressor=self.compressor.state_dict(),
                        provenance=provenance), path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("architecture") != "warehouse-nine-mode-flow-v1":
            raise ValueError("Not a deployment nine-mode checkpoint")
        m = cls(data["visual_dim"], data["kv_dim"], WarehouseConfig(**data["cfg"]))
        m.flow.load_state_dict(data["flow"], strict=True)
        m.compressor.load_state_dict(data["compressor"], strict=True)
        return m


def candidate_tensors(record, cfg):
    """Pack sparse, verified mode demonstrations, not nine copies of one path."""
    validate_v2(record)
    entries = record["supervision"].get("candidates", [])
    accepted = [c for c in entries if c.get("status") == "verified"]
    by_mode = [[] for _ in range(9)]
    for c in accepted:
        mode = c.get("mode_index")
        if isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode < 9:
            raise ValueError("Verified candidate requires a known mode")
        if not c.get("source_binding") == record["provenance"]["source_binding"]:
            raise ValueError("Candidate/source mismatch")
        if c.get("dt") != cfg.action_dt:
            raise ValueError("Action physical timestep mismatch")
        by_mode[mode].append(c)
    k = max(1, max(map(len, by_mode)))
    xy = torch.zeros(1, 9, k, cfg.horizon, 2)
    mask = torch.zeros(1, 9, k, cfg.horizon, dtype=torch.bool)
    valid = torch.zeros(1, 9, k, dtype=torch.bool)
    for m, entries in enumerate(by_mode):
        for j, c in enumerate(entries):
            p, v = torch.tensor(c["xy"]), torch.tensor(c["mask"], dtype=torch.bool)
            if p.shape != (cfg.horizon, 2) or v.shape != (cfg.horizon,) or not torch.isfinite(p).all():
                raise ValueError("Invalid candidate trajectory")
            xy[0,m,j], mask[0,m,j], valid[0,m,j] = p, v, bool(v.any())
    return xy, mask, valid


def readiness(output, cache=None):
    report = dict(optimizer_updates=0, formal_training_started=False,
                  actual_data_forward_pass=False, nine_mode_gradient_check=False,
                  reference_text_loss_check=False, training_ready=False)
    if cache is None:
        report.update(status="blocked", reason="Requires approved labels and actual frozen-VLM pre-compression cache; no synthetic substitution")
        write_json(output, report)
        return report
    data = torch.load(cache, map_location="cpu", weights_only=True)
    for key in ("vlm_checkpoint_sha256", "record_sha256", "source_binding", "feature_source"):
        if not data.get("provenance", {}).get(key):
            raise ValueError("Missing real-cache provenance: " + key)
    if data["provenance"]["feature_source"] != "actual_frozen_vlm":
        raise ValueError("Only actual frozen VLM caches qualify")
    if file_sha256(data["record_path"]) != data["provenance"]["record_sha256"]:
        raise ValueError("Stale label cache")
    if file_sha256(data["checkpoint_path"]) != data["provenance"]["vlm_checkpoint_sha256"]:
        raise ValueError("Stale VLM cache")
    record = read_json(data["record_path"])
    if record["provenance"]["source_binding"] != data["provenance"]["source_binding"]:
        raise ValueError("Cache and source differ")
    cfg = WarehouseConfig(**data["cfg"])
    torch.manual_seed(42)
    model = FlowTrainingAdapter(data["batch"]["visual"].shape[-1], data["kv"].shape[-1], cfg)
    labels = candidate_tensors(record, cfg)
    loss = model.loss(data["batch"], data["kv"], data["kv_valid"], *labels)
    loss.backward()
    finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    compressor_gradient = any(p.grad is not None and p.grad.abs().sum()>0 for p in model.compressor.parameters())
    report.update(status="action_interface_checked", actual_data_forward_pass=True,
                  loss=float(loss.detach()), nine_mode_gradient_check=bool(finite and compressor_gradient),
                  reason="Reference/text real-data loss and dataset coverage still require separate verification")
    write_json(output, report)
    return report
