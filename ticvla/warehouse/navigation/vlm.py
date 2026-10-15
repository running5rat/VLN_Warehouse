"""Navigation adapter using native cached generation; final query is processed."""

import time
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, LogitsProcessorList
from .grammar_decode import GreedyGrammarProcessor
from ..structured_io import StructuredVLM, response_grammar, parse_structured
from ..models import pack_last_kv
from ..vlm import QUERY


class NavigationVLM(StructuredVLM):
    def build_prompt(self, observation, *args, **kwargs):
        from .route_protocol import NAVIGATION_SYSTEM

        old = self.system_prompt
        if observation.get("route_schema") == "structured-v1":
            self.system_prompt = getattr(self, "navigation_system_override", NAVIGATION_SYSTEM)
        try:
            return super().build_prompt(observation, *args, **kwargs)
        finally:
            self.system_prompt = old

    @torch.inference_mode()
    def infer(self, observation, mode="TEXT", progress_callback=None):
        if mode != "TEXT":
            return super().infer(observation, mode, progress_callback)
        prompt, images = self.build_prompt(observation)
        ids = torch.tensor(
            [self.tokenizer.encode(prompt, add_special_tokens=True)], device=self.device
        )
        image_ids = [x["image_id"] for x in observation["rgb_history"] if x.get("path")]
        grammar = GreedyGrammarProcessor(
            self.tokenizer,
            response_grammar(
                image_ids,
                extended=False,
                structured_route=observation.get("route_schema") == "structured-v1",
            ),
            ids.shape[1],
        )
        tokenizer = self.tokenizer
        prefix_length = ids.shape[1]
        first_token_at = []

        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = tokenizer.decode(
                    input_ids[0, prefix_length:], skip_special_tokens=False
                )
                if not first_token_at:
                    first_token_at.append(time.perf_counter())
                return text.endswith(QUERY)

        torch.cuda.synchronize()
        start = time.perf_counter()
        embeddings = self._embeddings(ids, images)
        torch.cuda.synchronize()
        encoded = time.perf_counter()
        input_audit = dict(
            input_tokens=int(ids.numel()),
            visual_tokens=int((ids == self.img_token).sum().item()),
            image_paths=images,
            prompt_characters=len(prompt),
        )
        lm = self.backbone.language_model
        spans = []

        def before(module, args, kwargs):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            spans.append((a, b))

        def after(module, args, kwargs, output):
            spans[-1][1].record()

        pre = lm.register_forward_pre_hook(before, with_kwargs=True)
        post = lm.register_forward_hook(after, with_kwargs=True)
        generation_start = time.perf_counter()
        try:
            generated = lm.generate(
                input_ids=ids,
                inputs_embeds=embeddings,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                logits_processor=LogitsProcessorList([grammar]),
                stopping_criteria=StoppingCriteriaList([Stop()]),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        except (ValueError, RuntimeError) as exc:
            return dict(
                inference_mode="TEXT",
                prompt=prompt,
                text=grammar.prefix,
                valid=False,
                parsed=None,
                error=str(exc),
                constraint_statistics=grammar.record(),
                format_constraint="lm-format-enforcer character grammar checked greedy argmax",
                quality_validated=False,
                input_audit=input_audit,
            )
        finally:
            pre.remove()
            post.remove()
        torch.cuda.synchronize()
        decoded = time.perf_counter()
        suffix = generated.sequences[0, prefix_length:]
        text = self.tokenizer.decode(suffix, skip_special_tokens=False)
        base = dict(
            inference_mode="TEXT",
            text=text,
            prompt=prompt,
            generated_tokens=len(suffix),
            format_constraint="lm-format-enforcer/0.11.3 character grammar; checked greedy argmax; native cached generation",
            constraint_statistics=grammar.record(),
            quality_validated=False,
            head_quality="untrained; interface validation only",
            input_audit=input_audit,
        )
        durations = [a.elapsed_time(b) / 1000 for a, b in spans]
        base["stage_seconds"] = dict(
            image_encoding=encoded - start,
            native_generation=decoded - generation_start,
            first_token_wall=(first_token_at[0] - generation_start) if first_token_at else None,
            decode_wall=(decoded - first_token_at[0]) if first_token_at else None,
            grammar_wall=grammar.wall_seconds,
            prefill_gpu=durations[0] if durations else None,
            decode_gpu=sum(durations[1:]),
        )
        base["timing_scope"] = (
            "first_token_wall + decode_wall = native_generation; first token includes prefill "
            "and token selection. grammar_wall and GPU times are nested diagnostics, not additive. "
            "Excludes queue, loading, prompt construction and artifact writing."
        )
        try:
            parsed = parse_structured(
                text,
                image_ids,
                structured_route=observation.get("route_schema") == "structured-v1",
            )
        except ValueError as exc:
            return dict(base, valid=False, error=str(exc), parsed=None)
        cache = generated.past_key_values
        length = (
            cache.get_seq_length()
            if hasattr(cache, "get_seq_length")
            else cache[-1][0].shape[2]
        )
        complete = generated.sequences
        missing = complete[:, length:]
        if missing.numel() == 0:
            raise ValueError(
                "Unexpected cache already includes final query; hidden state unavailable"
            )
        torch.cuda.synchronize()
        query_start = time.perf_counter()
        out = lm(
            input_ids=missing,
            past_key_values=cache,
            attention_mask=torch.ones_like(complete),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        torch.cuda.synchronize()
        query_end = time.perf_counter()
        reference = self.decoder(out.hidden_states[-1][:, -1].float())
        kv = pack_last_kv(out.past_key_values).float()
        z = self.compressor(
            kv, torch.ones(kv.shape[:2], dtype=torch.bool, device=kv.device)
        )
        for tensor in (z, reference["reference_xy"], reference["confidence_logits"]):
            if not torch.isfinite(tensor).all():
                raise ValueError("Nonfinite model output")
        torch.cuda.synchronize()
        ended = time.perf_counter()
        base["stage_seconds"].update(
            final_query=query_end - query_start,
            heads=ended - query_end,
            reference_and_compressor=ended - decoded,
        )
        return dict(
            base,
            valid=True,
            error=None,
            parsed=parsed,
            z_sem=z[0].cpu().half(),
            reference_xy=reference["reference_xy"][0].cpu(),
            reference_mask=reference["reference_mask"][0].cpu(),
            confidence=float(reference["confidence_logits"].sigmoid()[0]),
            raw_kv_shape=list(kv.shape),
            processed_query=True,
        )
