"""InternVL adapter with cached decoding and an actually processed query token."""

import copy
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .geometry import delay_features
from .models import ReferenceDecoder, pack_last_kv

QUERY = "<REFLINE>"
SYSTEM = (
    "You navigate a wheeled robot using only the supplied observations. "
    "Body coordinates: x forward, y left, metres; yaw counterclockwise in radians. "
    "Keep the original task and hard constraints when updating the remaining Route. "
    "Output exactly <landmark>[]</landmark><local>unknown</local>"
    "<Route>remaining local instruction</Route><waypoint>(x,y);(x,y);(x,y);(x,y);(x,y)</waypoint>"
    "<REFLINE>. Waypoints are spatial anchors on the same local route, not timed actions. "
    "For no valid route use <waypoint>[]</waypoint>. Do not invent unseen free space."
)


class LoRALinear(nn.Module):
    """Minimal explicit LoRA; frozen base weights remain checkpoint-compatible."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(
            torch.empty(
                rank, base.in_features, device=base.weight.device, dtype=torch.float32
            )
        )
        self.b = nn.Parameter(
            torch.zeros(
                base.out_features, rank, device=base.weight.device, dtype=torch.float32
            )
        )
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        delta = F.linear(F.linear(x.float(), self.a), self.b) * self.scale
        return self.base(x) + delta.to(x.dtype)


def install_lora(module, rank, alpha):
    count = 0
    for name, child in list(module.named_children()):
        if name in ("q_proj", "k_proj", "v_proj", "o_proj") and isinstance(
            child, nn.Linear
        ):
            setattr(module, name, LoRALinear(child, rank, alpha))
            count += 1
        else:
            count += install_lora(child, rank, alpha)
    return count


def load_legacy_vlm(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint.get("state_dict", checkpoint)
    expected = model.state_dict()
    matching = {}
    prefixes = (
        "model.vlm_model.vlm.",
        "model.vlm.",
        "vlm_model.vlm.",
        "vlm.",
        "model.",
        "",
    )
    for key, value in source.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                target = key[len(prefix) :]
                if target in expected and value.shape == expected[target].shape:
                    matching[target] = value
                    break
    if not matching:
        raise ValueError("No matching VLM weights in legacy checkpoint")
    missing, unexpected = model.load_state_dict(matching, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Incomplete legacy VLM import: {len(missing)} missing, {len(unexpected)} unexpected"
        )
    return len(matching)


def image_tensor(path, size):
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(
            image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        ).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    return (
        tensor - tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    ) / tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]


def parse_response(text):
    pattern = (
        r"\s*<landmark>(.*?)</landmark>\s*<local>(.*?)</local>\s*"
        r"<Route>(.*?)</Route>\s*<waypoint>(.*?)</waypoint>\s*<REFLINE>\s*"
    )
    match = re.fullmatch(pattern, text, re.DOTALL)
    if not match or not match[3].strip():
        return None
    try:
        landmarks = json.loads(match[1])
        if not isinstance(landmarks, list):
            return None
        raw = match[4].strip()
        if raw == "[]":
            anchors = []
        else:
            number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
            pair = rf"\(\s*({number})\s*,\s*({number})\s*\)"
            if not re.fullmatch(rf"{pair}(?:\s*;\s*{pair}){{4}}", raw):
                return None
            anchors = [list(map(float, p)) for p in re.findall(pair, raw)]
            if not np.isfinite(anchors).all():
                return None
        return {
            "landmark": landmarks,
            "local": match[2].strip(),
            "route": match[3].strip(),
            "anchors": anchors,
        }
    except (ValueError, TypeError):
        return None


def target_text(supervision):
    # No annotations/latent memory are invented for the deferred semantic mapper.
    route = supervision["route"].replace("<", "").replace(">", "")
    anchors = supervision.get("anchors", [])
    coords = ";".join(f"({x:.3f},{y:.3f})" for x, y in anchors) if anchors else "[]"
    return f"<landmark>[]</landmark><local>unknown</local><Route>{route}</Route><waypoint>{coords}</waypoint>{QUERY}"


class InternVLReference(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.cfg = cfg
        self.compute_dtype = (
            torch.bfloat16 if cfg.device.startswith("cuda") else torch.float32
        )
        self.backbone = AutoModel.from_pretrained(
            cfg.model_path,
            torch_dtype=self.compute_dtype,
            trust_remote_code=True,
            use_flash_attn=False,
        ).to(cfg.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path,
            trust_remote_code=True,
            use_fast=True,
            fix_mistral_regex=True,
        )
        self.backbone.requires_grad_(False)
        if cfg.vlm_checkpoint:
            load_legacy_vlm(self.backbone, cfg.vlm_checkpoint)
        if not install_lora(
            self.backbone.language_model, cfg.lora_rank, cfg.lora_alpha
        ):
            raise ValueError("No language attention projections found for LoRA")
        llm = self.backbone.config.llm_config
        self.visual_dim = llm.hidden_size
        self.kv_dim = (
            2
            * llm.num_key_value_heads
            * getattr(llm, "head_dim", llm.hidden_size // llm.num_attention_heads)
        )
        self.decoder = ReferenceDecoder(llm.hidden_size, cfg).to(cfg.device)
        self.img_token = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.query_ids = self.tokenizer.encode(QUERY, add_special_tokens=False)
        self.train(True)

    @property
    def device(self):
        return self.decoder.control_head.weight.device

    def train(self, mode=True):
        super().train(mode)
        self.backbone.vision_model.eval()
        self.backbone.mlp1.eval()
        return self

    def build_prompt(self, observation, map_mode="correct", replacement_map=None):
        """Whitelisted fields only; never receives the supervision namespace."""
        obs = observation
        images, parts = [], []
        for frame in obs["rgb_history"]:
            if frame["t_obs"] > obs["t_obs"]:
                raise ValueError("Future RGB frame")
            if frame.get("path") and Path(frame["path"]).is_file():
                transform = delay_features(
                    frame["T_odom_body"],
                    obs["T_odom_body"],
                    obs["t_obs"] - frame["t_obs"],
                )
                frame_kind = (
                    "Current source RGB"
                    if frame["t_obs"] == obs["t_obs"]
                    else "Historical RGB"
                )
                parts.append(
                    f"{frame_kind} {frame.get('image_id', '')}; age/relative pose {transform.tolist()}: <image>"
                )
                images.append(frame["path"])
            else:
                parts.append("Historical RGB unavailable.")
        if map_mode != "removed" and obs.get("map_valid"):
            path = replacement_map if map_mode == "shuffled" else obs["map_image"]
            if not path or not Path(path).is_file():
                raise FileNotFoundError("Requested map image is missing")
            images.append(path)
            parts.append(
                "Local observed geometry with metric legend; gray is unknown: <image>"
            )
        else:
            parts.append("Map unavailable.")
        task = obs["task"]
        payload = {
            "instruction": task["instruction"],
            "hard_constraints": task.get("hard_constraints", []),
            "remaining_route": obs["route"],
            "execution_state": obs.get("execution_state", {}),
            "robot_state_v_omega": obs["robot_state"],
            "state_valid": obs["robot_state_mask"],
        }
        for key in ("previous_reference", "follow_target", "task_mode"):
            if key in obs:
                payload[key] = obs[key]
        if "map_landmarks" in obs:
            payload["map_interface"] = "rendered_mapview_baseline"
            payload["observed_map_landmarks"] = obs["map_landmarks"][:16]
        if "navigation_object_question" in obs:
            payload["navigation_object_question"] = obs["navigation_object_question"]
        if "visual_object_evidence" in obs:
            payload["visual_object_evidence"] = obs["visual_object_evidence"]
        parts.append(json.dumps(payload, ensure_ascii=False))
        if "navigation_object_question" in obs:
            parts.append(obs["navigation_object_question"])
        if "visual_object_evidence" in obs:
            parts.append(
                "Your prior visual object list is supplied above as fallible model evidence. "
                "Select distinct individual object names from that list when they are visible "
                "and relevant to this navigation task, then answer in the required format."
            )
        template = copy.deepcopy(self.backbone.conv_template)
        template.system_message = getattr(self, "system_prompt", SYSTEM)
        template.append_message(template.roles[0], "\n".join(parts))
        template.append_message(template.roles[1], None)
        prompt = template.get_prompt()
        for _ in images:
            prompt = prompt.replace(
                "<image>",
                "<img>" + "<IMG_CONTEXT>" * self.backbone.num_image_token + "</img>",
                1,
            )
        return prompt, images

    def _embeddings(self, ids, images):
        embeddings = self.backbone.language_model.get_input_embeddings()(ids).clone()
        if images:
            pixels = torch.stack(
                [image_tensor(p, self.cfg.image_size) for p in images]
            ).to(self.device, self.compute_dtype)
            with torch.no_grad():
                visual = self.backbone.extract_feature(pixels)
            selected = ids == self.img_token
            if selected.sum().item() != visual.shape[0] * visual.shape[1]:
                raise ValueError("Image/token count mismatch")
            embeddings[selected] = visual.reshape(-1, visual.shape[-1]).to(
                embeddings.dtype
            )
        return embeddings

    def forward_record(self, record):
        prompt, images = self.build_prompt(record["observation"])
        prefix = self.tokenizer.encode(prompt, add_special_tokens=True)
        answer = target_text(record["supervision"])
        # Separate marker tokenization guarantees a stable processed query position.
        suffix = (
            self.tokenizer.encode(answer[: -len(QUERY)], add_special_tokens=False)
            + self.query_ids
        )
        ids = torch.tensor([prefix + suffix], device=self.device)
        labels = torch.full_like(ids, -100)
        if record["supervision"].get("text_valid"):
            labels[:, len(prefix) :] = ids[:, len(prefix) :]
        out = self.backbone.language_model(
            inputs_embeds=self._embeddings(ids, images),
            attention_mask=torch.ones_like(ids),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        output = self.decoder(out.hidden_states[-1][:, -1].float())
        if (labels != -100).any():
            text_loss = F.cross_entropy(
                out.logits[:, :-1].float().reshape(-1, out.logits.shape[-1]),
                labels[:, 1:].reshape(-1),
            )
        else:
            text_loss = output["reference_xy"].sum() * 0.0
        return output, text_loss

    @torch.no_grad()
    def generate_record(self, observation, map_mode="correct", replacement_map=None):
        prompt, images = self.build_prompt(observation, map_mode, replacement_map)
        prefix = self.tokenizer.encode(prompt, add_special_tokens=True)
        ids = torch.tensor([prefix], device=self.device)
        attention = torch.ones_like(ids)
        lm = self.backbone.language_model
        out = lm(
            inputs_embeds=self._embeddings(ids, images),
            attention_mask=attention,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        generated, complete = [], False

        def process(token):
            nonlocal out, attention
            attention = torch.cat((attention, torch.ones_like(attention[:, :1])), 1)
            out = lm(
                input_ids=torch.tensor([[token]], device=self.device),
                attention_mask=attention,
                past_key_values=out.past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )

        for _ in range(self.cfg.max_new_tokens):
            token = int(out.logits[0, -1].argmax())
            if token == self.tokenizer.eos_token_id:
                break
            generated.append(token)
            process(token)  # the predicted marker's own hidden state is now available
            if (
                self.tokenizer.decode(generated, skip_special_tokens=False)
                .rstrip()
                .endswith(QUERY)
            ):
                complete = True
                break
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        parsed = parse_response(text) if complete else None
        if not complete:
            # Fixed query permits diagnostic decoding, but the snapshot is invalid.
            for token in self.query_ids:
                process(token)
        output = self.decoder(out.hidden_states[-1][:, -1].float())
        output["raw_kv"] = pack_last_kv(out.past_key_values).detach()
        output["kv_mask"] = attention.bool()
        output["text"], output["parsed"], output["query_complete"] = (
            text,
            parsed,
            complete,
        )
        return output

    @torch.no_grad()
    def current_visual(self, observation):
        path = observation.get("rgb_current")
        if not path or not Path(path).is_file():
            return torch.zeros(1, 1, self.visual_dim, device=self.device), torch.zeros(
                1, 1, dtype=torch.bool, device=self.device
            )
        pixels = image_tensor(path, self.cfg.image_size)[None].to(
            self.device, self.compute_dtype
        )
        features = self.backbone.extract_feature(pixels).float()
        return features, torch.ones(
            features.shape[:2], device=self.device, dtype=torch.bool
        )

    def adapter_state(self):
        return {
            k: v.detach().cpu()
            for k, v in self.state_dict().items()
            if k.startswith("decoder.") or k.endswith((".a", ".b"))
        }

    def load_adapter(self, state):
        expected = set(self.adapter_state())
        if set(state) != expected:
            raise ValueError("Reference adapter architecture mismatch")
        self.load_state_dict(state, strict=False)
