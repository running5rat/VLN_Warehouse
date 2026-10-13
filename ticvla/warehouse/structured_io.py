"""Inference-only v0.6 protocol. Format validity never implies planning quality."""

import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .mapping import SemanticEvent
from .object_io import OBJECT_SYSTEM, object_question
from .models import ContextCompressor, pack_last_kv
from .schema import SemanticSnapshot, TaskContract, TaskState
from .vlm import QUERY, InternVLReference, image_tensor, parse_response


SYSTEM = """You are a visual perception assistant for a wheeled robot.
First inspect the source RGB and identify tangible visible object categories relevant to the task.
The landmark answer must contain object names, not actions, trajectories, or navigation commands.
Name individual objects that a segmentation model can outline. Never use the overall room,
scene, building, or aisle as an object category. List different categories, without repetitions.
Coordinates are x forward, y left, in metres. Preserve the original task and constraints.
Reply in this exact order and case:
<landmark>[{"image_id":"source RGB identifier","category":"visible object class","navigation_role":"localization","reason":"short navigation relevance"}]</landmark>
<local>short location description or unknown</local>
<Route>short remaining instruction</Route>
<waypoint>(x,y);(x,y);(x,y);(x,y);(x,y)</waypoint><REFLINE>
The landmark field is a JSON array, at most three objects. Use only supplied source RGB identifiers.
Answer the navigation_object_question in landmark. Use short English physical object class names,
one entry per class, using one to three English words, suitable for a text-prompted object segmenter. Do not use instance numbers
or spatial relations as class names. navigation_role is localization, avoidance, or both.
reason explains the relevance in at most 120 characters. Neither role nor reason proves staticness.
Only propose visible physical objects; topology and relations belong in local.
Empty landmarks and unavailable waypoints are represented by []. Do not infer hidden geometry.
Use concise English, local at most 80 characters and Route at most 120 characters.
Map identifiers are provenance, not place identities. End immediately after <REFLINE>.
"""


def response_grammar(image_ids, *, extended=False):
    from lmformatenforcer import (
        JsonSchemaParser,
        RegexParser,
        SequenceParser,
        StringParser,
    )
    from lmformatenforcer import CharacterLevelParserConfig

    if not image_ids:
        raise ValueError("Structured TEXT inference requires at least one source image")
    schema = {
        "type": "array",
        "maxItems": 3,
        "items": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string", "enum": list(image_ids)},
                "category": {"type": "string", "minLength": 1, "maxLength": 40},
            },
            "required": ["image_id", "category"],
            "additionalProperties": False,
        },
    }
    if extended:
        # Explicit bounded ASCII syntax: the JSON parser in the pinned enforcer
        # can accept a string token crossing maxLength. No class vocabulary is fixed.
        source = "(" + "|".join(re.escape(x) for x in image_ids) + ")"
        word = r"[A-Za-z][A-Za-z-]{0,11}"
        category = rf"{word}( {word}){{0,2}}"
        reason = r"[A-Za-z0-9][A-Za-z0-9 ,.';:()_/-]{0,119}"
        item = (
            r'\{"image_id":"'
            + source
            + r'","category":"'
            + category
            + r'","navigation_role":"(localization|avoidance|both)","reason":"'
            + reason
            + r'"\}'
        )
        landmark_parser = RegexParser(r"\[(" + item + r"(," + item + r"){0,2})?\]")
    else:
        landmark_parser = JsonSchemaParser(
            schema,
            config=CharacterLevelParserConfig(
                max_consecutive_whitespaces=1, force_json_field_order=True
            ),
        )
    number = r"-?[0-9]{1,2}(\.[0-9]{1,3})?"
    pair = rf"\({number},{number}\)"
    return SequenceParser(
        [
            StringParser("<landmark>"),
            landmark_parser,
            StringParser("</landmark><local>"),
            RegexParser(r"[A-Za-z0-9][A-Za-z0-9 ,.';:()_/-]{0,79}"),
            StringParser("</local><Route>"),
            RegexParser(r"[A-Za-z0-9][A-Za-z0-9 ,.';:()_/-]{0,119}"),
            StringParser("</Route><waypoint>"),
            RegexParser(rf"(\[\]|{pair}(;{pair}){{4}})"),
            StringParser("</waypoint><REFLINE>"),
        ]
    )


def parse_structured(text, image_ids):
    parsed = parse_response(text)
    if parsed is None:
        raise ValueError("Invalid or truncated v0.6 tagged response")
    if not parsed["local"].strip() or len(parsed["landmark"]) > 3:
        raise ValueError("Missing Local or excessive landmarks")
    for item in parsed["landmark"]:
        if not isinstance(item, dict) or set(item) not in (
            {"image_id", "category"},
            {"image_id", "category", "navigation_role", "reason"},
        ):
            raise ValueError(
                "Invalid landmark fields: expected legacy or navigation schema"
            )
        if item["image_id"] not in image_ids:
            raise ValueError("Landmark references an unavailable source image")
        if (
            not isinstance(item["category"], str)
            or not item["category"].strip()
            or len(item["category"]) > 40
        ):
            raise ValueError("Empty landmark category")
        if "navigation_role" in item:
            if item["navigation_role"] not in ("localization", "avoidance", "both"):
                raise ValueError("Invalid navigation role")
            if (
                not isinstance(item["reason"], str)
                or not item["reason"].strip()
                or len(item["reason"]) > 120
            ):
                raise ValueError("Invalid navigation reason")
    return parsed


class StructuredVLM(InternVLReference):
    system_prompt = SYSTEM

    def __init__(self, cfg):
        super().__init__(cfg)
        self.compressor = ContextCompressor(self.kv_dim, cfg).to(self.device)
        self.eval().requires_grad_(False)
        self._tokenizer_data = None

    @torch.inference_mode()
    def describe_objects(self, observation):
        """Optional same-model visual QA evidence, never a manually supplied answer."""
        frame = observation["rgb_history"][-1]
        question, system = object_question(observation), OBJECT_SYSTEM
        previous_system = self.backbone.system_message
        try:
            self.backbone.system_message = system
            pixels = image_tensor(frame["path"], self.cfg.image_size)[None].to(
                self.device, self.compute_dtype
            )
            decoding = dict(max_new_tokens=160, do_sample=False)
            if observation.get("core_vocabulary"):
                from lmformatenforcer import JsonSchemaParser
                from lmformatenforcer.integrations.transformers import (
                    build_token_enforcer_tokenizer_data,
                    build_transformers_prefix_allowed_tokens_fn,
                )

                schema = dict(
                    type="object",
                    properties={
                        "selected_categories": dict(
                            type="array",
                            maxItems=7,
                            items=dict(
                                type="string", enum=observation["core_vocabulary"]
                            ),
                        ),
                        "proposed_categories": dict(
                            type="array",
                            maxItems=5,
                            items=dict(type="string", minLength=1, maxLength=80),
                        ),
                    },
                    required=["selected_categories", "proposed_categories"],
                    additionalProperties=False,
                )
                if self._tokenizer_data is None:
                    self._tokenizer_data = build_token_enforcer_tokenizer_data(
                        self.tokenizer
                    )
                decoding["prefix_allowed_tokens_fn"] = (
                    build_transformers_prefix_allowed_tokens_fn(
                        self._tokenizer_data, JsonSchemaParser(schema)
                    )
                )
            answer = self.backbone.chat(
                self.tokenizer,
                pixels,
                question,
                decoding,
                num_patches_list=[1],
            )
        finally:
            self.backbone.system_message = previous_system
        return dict(
            image_id=frame["image_id"],
            system=system,
            question=question,
            answer=answer,
            max_new_tokens=160,
            token_count=len(self.tokenizer.encode(answer, add_special_tokens=False)),
            origin=(
                "same-model constrained visual QA; not ground truth"
                if observation.get("core_vocabulary")
                else "same-model unconstrained visual QA; not ground truth"
            ),
            format_constraint="JSON schema, one generation"
            if observation.get("core_vocabulary")
            else None,
        )

    @torch.no_grad()
    def infer(self, observation, mode="TEXT", progress_callback=None):
        if mode not in ("TEXT", "PREFILL_FOLLOW"):
            raise ValueError("Unsupported VLM mode")
        prompt, images = self.build_prompt(observation)
        prefix = self.tokenizer.encode(prompt, add_special_tokens=True)
        image_ids = [f["image_id"] for f in observation["rgb_history"] if f.get("path")]
        allowed = None
        if mode == "TEXT":
            from lmformatenforcer.integrations.transformers import (
                build_token_enforcer_tokenizer_data,
                build_transformers_prefix_allowed_tokens_fn,
            )

            if self._tokenizer_data is None:
                self._tokenizer_data = build_token_enforcer_tokenizer_data(
                    self.tokenizer
                )
            allowed = build_transformers_prefix_allowed_tokens_fn(
                self._tokenizer_data, response_grammar(image_ids, extended=observation.get("landmark_schema", "extended") != "legacy")
            )
        else:
            prefix += self.query_ids
        ids = torch.tensor([prefix], device=self.device)
        attention = torch.ones_like(ids)
        lm = self.backbone.language_model
        def clock():
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
            return time.perf_counter()
        started = clock()
        embeddings = self._embeddings(ids, images)
        encoded = clock()
        out = lm(
            inputs_embeds=embeddings,
            attention_mask=attention,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        prefilled = clock()
        generated, complete, constrained_steps = [], mode == "PREFILL_FOLLOW", 0
        raw_argmax_changes = 0
        if mode == "TEXT":
            for _ in range(self.cfg.max_new_tokens):
                candidates = allowed(0, torch.tensor(prefix + generated))
                candidates = torch.tensor(
                    candidates, dtype=torch.long, device=self.device
                )
                logits = out.logits[0, -1]
                token = int(candidates[logits[candidates].argmax()])
                raw_argmax_changes += int(token != int(logits.argmax()))
                constrained_steps += 1
                if token == self.tokenizer.eos_token_id:
                    break
                generated.append(token)
                if progress_callback is not None and len(generated) % 32 == 0:
                    progress_callback(
                        {
                            "generated_tokens": len(generated),
                            "partial_text": self.tokenizer.decode(
                                generated, skip_special_tokens=False
                            ),
                            "complete": False,
                        }
                    )
                attention = torch.cat((attention, torch.ones_like(attention[:, :1])), 1)
                # Process the selected marker itself, using the existing cache.
                out = lm(
                    input_ids=torch.tensor([[token]], device=self.device),
                    attention_mask=attention,
                    past_key_values=out.past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                if self.tokenizer.decode(generated, skip_special_tokens=False).endswith(
                    QUERY
                ):
                    complete = True
                    break
        decoded = clock()
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        parsed, error = None, None
        if mode == "TEXT":
            try:
                if not complete:
                    raise ValueError("Generation ended before processed REFLINE")
                parsed = parse_structured(text, image_ids)
            except ValueError as exc:
                error = str(exc)
        if error:
            return {
                "valid": False,
                "error": error,
                "text": text,
                "parsed": None,
                "inference_mode": mode,
                "prompt": prompt,
                "generated_tokens": len(generated),
                "format_constraint": "lm-format-enforcer/0.11.3 XML+JSON+regex",
                "constrained_steps": constrained_steps,
                "changed_argmax_steps": raw_argmax_changes,
            }
        reference = self.decoder(out.hidden_states[-1][:, -1].float())
        kv = pack_last_kv(out.past_key_values).float()
        z_sem = self.compressor(kv, attention.bool())
        if z_sem.shape != (1, self.cfg.context_queries, self.cfg.hidden_dim):
            raise ValueError("Context shape mismatch")
        for value in (z_sem, reference["reference_xy"], reference["confidence_logits"]):
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite inference tensor")
        if reference["reference_xy"].shape != (1, self.cfg.num_reference_points, 2):
            raise ValueError("Reference coordinate shape mismatch")
        mask = reference["reference_mask"]
        if mask.dtype != torch.bool or mask.shape != (1, self.cfg.num_reference_points):
            raise ValueError("Reference mask shape/type mismatch")
        if not torch.equal(
            mask,
            torch.arange(mask.shape[1], device=mask.device)[None]
            < mask.sum(-1)[:, None],
        ):
            raise ValueError("Reference mask is not a prefix")
        return {
            "valid": True,
            "error": None,
            "inference_mode": mode,
            "text": text,
            "parsed": parsed,
            "prompt": prompt,
            "z_sem": z_sem[0].cpu().half(),
            "reference_xy": reference["reference_xy"][0].cpu(),
            "reference_mask": reference["reference_mask"][0].cpu(),
            "confidence": float(reference["confidence_logits"].sigmoid()[0]),
            "raw_kv_shape": list(kv.shape),
            "generated_tokens": len(generated),
            "processed_query": complete,
            "format_constraint": "lm-format-enforcer/0.11.3 XML+JSON+regex"
            if allowed
            else None,
            "constrained_steps": constrained_steps,
            "changed_argmax_steps": raw_argmax_changes,
            "stage_seconds": {"image_encoding": encoded-started,
                              "prefill": prefilled-encoded, "decode": decoded-prefilled,
                              "reference_and_compressor": clock()-decoded},
            "head_quality": "untrained; interface validation only",
            "quality_validated": False,
        }


def make_observation(store, image_id, mapview, task):
    if store.oracle or mapview.get("oracle"):
        raise ValueError("Oracle maps are forbidden in LLM input")
    if mapview.get("image_id") != image_id or mapview.get("map_version") != str(
        store.version
    ):
        raise ValueError("MapView source/version mismatch; export a fresh view")
    obs, _ = store.source_arrays(image_id)
    available = sorted(
        (
            o
            for o in store.records("observations")
            if o["submap_id"] == obs["submap_id"] and o["t_obs"] <= obs["t_obs"]
        ),
        key=lambda o: o["t_obs"],
    )
    history, seen = [], set()
    for offset in (3, 2, 1, 0):
        candidates = [
            o for o in available if o["t_obs"] <= obs["t_obs"] - offset + 1e-6
        ]
        if candidates:
            frame = candidates[-1]
            if frame["image_id"] not in seen:
                history.append(
                    {
                        "image_id": frame["image_id"],
                        "path": frame["rgb"],
                        "t_obs": frame["t_obs"],
                        "T_odom_body": frame["T_odom_body"],
                    }
                )
                seen.add(frame["image_id"])
    state, valid = [0.0, 0.0], [False, False]
    if len(available) > 1:
        previous = available[-2]
        tf = np.linalg.inv(np.array(previous["T_odom_body"])) @ np.array(
            obs["T_odom_body"]
        )
        dt = obs["t_obs"] - previous["t_obs"]
        state = [float(tf[0, 3] / dt), float(np.arctan2(tf[1, 0], tf[0, 0]) / dt)]
        valid = [True, True]
    return {
        "task": task,
        "t_obs": obs["t_obs"],
        "odom_epoch": obs["odom_epoch"],
        "T_odom_body": obs["T_odom_body"],
        "rgb_history": history,
        "rgb_current": obs["rgb"],
        "route": task["instruction"],
        "route_revision": 0,
        "execution_state": {},
        "robot_state": state,
        "robot_state_mask": valid,
        "map_valid": True,
        "map_image": mapview["image"],
        "map_landmarks": [
            {k: lm[k] for k in ("category", "name", "position", "orientation_kind")}
            for lm in mapview["landmarks"][:16]
        ],
        "map_version": mapview["map_version"],
    }


def packet_and_events(result, observation, request_id, generation_id, elapsed):
    if not result["valid"]:
        raise ValueError("Cannot publish invalid model result")
    task = observation["task"]
    state = TaskState(
        TaskContract(
            task["task_id"],
            task["instruction"],
            tuple(task.get("hard_constraints", [])),
        ),
        observation["route"],
        observation["route_revision"],
    )
    events = []
    if result["inference_mode"] == "TEXT":
        state.commit_route(
            result["parsed"]["route"], task["task_id"], observation["route_revision"]
        )
        if task.get("task_type") != "PERSON_FOLLOW":
            events = [
                SemanticEvent(
                    f"{request_id}:{i}",
                    item["image_id"],
                    item["category"],
                    source="llm",
                    request_id=request_id,
                    navigation_role=item.get("navigation_role"),
                    reason=item.get("reason"),
                )
                for i, item in enumerate(result["parsed"]["landmark"])
            ]
            if result["parsed"]["local"].strip().lower() != "unknown":
                events.append(
                    SemanticEvent(
                        f"{request_id}:local",
                        observation["rgb_history"][-1]["image_id"],
                        result["parsed"]["local"],
                        source="llm",
                        kind="relation",
                        request_id=request_id,
                    )
                )
    # Offline replay clock: service duration is mapped onto this request's source
    # clock. This is not a cross-host latency or online age measurement.
    stamp = observation["t_obs"]
    packet = SemanticSnapshot(
        result["z_sem"],
        result["reference_xy"],
        result["reference_mask"],
        result["confidence"],
        stamp,
        stamp,
        stamp + elapsed,
        observation["T_odom_body"],
        observation["odom_epoch"],
        generation_id,
        task["task_id"],
        observation["route_revision"],
        state.route_revision,
        observation["map_version"],
        result["inference_mode"],
    )
    return packet, events, state


def save_result(output, request, result, packet=None, events=(), state=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    serial = {k: v for k, v in result.items() if not isinstance(v, torch.Tensor)}
    serial["request"] = request
    serial["semantic_events"] = [asdict(e) for e in events]
    if packet:
        values = asdict(packet)
        arrays = {
            k: values.pop(k).numpy() for k in ("z_sem", "refline_xy", "refline_mask")
        }
        np.savez_compressed(output / "snapshot.npz", **arrays)
        serial["snapshot"] = values
        serial["task_state"] = asdict(state)
        serial["clock_scope"] = (
            "offline replay source time + measured local service duration"
        )
    # Final JSON is the commit marker for this complete packet.
    tmp = output / "response.json.tmp"
    tmp.write_text(json.dumps(serial, indent=2, allow_nan=False))
    tmp.replace(output / "response.json")
