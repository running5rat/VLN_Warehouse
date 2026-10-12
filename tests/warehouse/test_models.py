import unittest
from types import SimpleNamespace

import torch
from torch import nn

from ticvla.warehouse.config import WarehouseConfig
from ticvla.warehouse.models import (
    ContextCompressor,
    FlowActionExpert,
    QueryActionExpert,
    ReferenceDecoder,
    pack_last_kv,
    spline_basis,
)
from ticvla.warehouse.vlm import QUERY, InternVLReference, LoRALinear, parse_response


def tiny_config(**kwargs):
    values = {
        "device": "cpu",
        "hidden_dim": 32,
        "num_heads": 4,
        "num_layers": 1,
        "context_queries": 4,
        "num_control_points": 4,
        "num_reference_points": 8,
        "reference_hidden_dim": 16,
        "horizon": 6,
    }
    return WarehouseConfig(**{**values, **kwargs})


def action_batch(cfg):
    return {
        "visual": torch.randn(2, 4, 12),
        "visual_mask": torch.ones(2, 4, dtype=torch.bool),
        "kv": torch.randn(2, 7, 8),
        "kv_mask": torch.ones(2, 7, dtype=torch.bool),
        "bev": torch.randn(2, 4, 16, 16),
        "map_valid": torch.ones(2, dtype=torch.bool),
        "state": torch.zeros(2, 2),
        "state_mask": torch.ones(2, 2, dtype=torch.bool),
        "reference_xy": torch.randn(2, cfg.num_reference_points, 2),
        "reference_mask": torch.ones(2, cfg.num_reference_points, dtype=torch.bool),
        "reference_confidence": torch.ones(2),
        "semantic_valid": torch.ones(2, dtype=torch.bool),
        "delay": torch.zeros(2, 5),
        "previous_plan": torch.zeros(2, cfg.horizon, 3),
        "previous_plan_mask": torch.zeros(2, cfg.horizon, dtype=torch.bool),
        "actions": torch.randn(2, cfg.horizon, 2),
        "action_mask": torch.ones(2, cfg.horizon, dtype=torch.bool),
    }


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_spline_partition_and_gradient(self):
        basis = spline_basis(8, 32)
        torch.testing.assert_close(basis.sum(-1), torch.ones(32))
        self.assertEqual(float(basis[0, 0]), 1.0)
        self.assertEqual(float(basis[-1, -1]), 1.0)
        cfg = tiny_config()
        model = ReferenceDecoder(12, cfg)
        output = model(torch.randn(2, 12))
        target = {
            "reference_xy": torch.ones(2, 8, 2),
            "reference_mask": torch.tensor(
                [[1, 1, 1, 0, 0, 0, 0, 0], [0] * 8], dtype=torch.bool
            ),
            "reference_known": torch.tensor([True, True]),
            "reference_valid": torch.tensor([True, False]),
            "text_valid": torch.tensor([True, False]),
        }
        sum(model.loss(output, target).values()).backward()
        self.assertGreater(float(model.control_head.weight.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(model.length_head.weight.grad).all())

    def test_kv_padding_and_empty_attention(self):
        cfg = tiny_config()
        model = ContextCompressor(8, cfg).eval()
        kv = torch.randn(2, 5, 8)
        mask = torch.tensor([[1, 1, 0, 0, 0], [0] * 5], dtype=torch.bool)
        before = model(kv, mask)
        kv[~mask] = 10000.0
        after = model(kv, mask)
        torch.testing.assert_close(before, after)
        self.assertTrue(torch.isfinite(after).all())
        empty = model(torch.empty(2, 0, 8), torch.empty(2, 0, dtype=torch.bool))
        self.assertTrue(torch.isfinite(empty).all())
        k = torch.randn(2, 2, 5, 2)
        v = k + 1
        self.assertEqual(pack_last_kv(((k, v),)).shape, (2, 5, 8))

    def test_flow_masks_gradients_and_sample_shape(self):
        cfg = tiny_config()
        model = FlowActionExpert(12, 8, cfg)
        batch = action_batch(cfg)
        batch["action_mask"][:, 3:] = False
        generator = torch.Generator().manual_seed(7)
        loss = model.loss(batch, generator)
        loss.backward()
        self.assertGreater(
            float(model.context_encoder.compressor.project[1].weight.grad.abs().sum()),
            0.0,
        )
        before = float(loss.detach())
        batch["actions"][:, 3:] = 10000.0
        after = float(model.loss(batch, torch.Generator().manual_seed(7)).detach())
        self.assertAlmostEqual(before, after, places=5)
        first = model.sample(batch, torch.Generator().manual_seed(8))["trajectories"]
        second = model.sample(batch, torch.Generator().manual_seed(8))["trajectories"]
        self.assertEqual(first.shape, (2, 1, 1, 6, 2))
        torch.testing.assert_close(first, second)
        query = QueryActionExpert(12, 8, cfg)
        self.assertEqual(query.sample(batch)["trajectories"].shape, first.shape)

    def test_lora_freezes_base_and_round_trip(self):
        model = LoRALinear(nn.Linear(4, 3), 2, 4)
        model(torch.randn(5, 4)).sum().backward()
        self.assertIsNone(model.base.weight.grad)
        self.assertGreater(float(model.b.grad.abs().sum()), 0.0)
        clone = LoRALinear(nn.Linear(4, 3), 2, 4)
        clone.load_state_dict(model.state_dict())
        x = torch.randn(2, 4)
        torch.testing.assert_close(model(x), clone(x))

    def test_default_architecture_interfaces(self):
        cfg = WarehouseConfig(device="cpu")
        batch = {k: v[:1] for k, v in action_batch(cfg).items()}
        batch["visual"] = torch.randn(1, 256, 896)
        batch["visual_mask"] = torch.ones(1, 256, dtype=torch.bool)
        batch["kv"] = torch.randn(1, 1280, 256)
        batch["kv_mask"] = torch.ones(1, 1280, dtype=torch.bool)
        batch["bev"] = torch.zeros(1, 4, 160, 160)
        decoder = ReferenceDecoder(896, cfg)
        reference = decoder(torch.randn(1, 896))
        self.assertEqual(reference["reference_xy"].shape, (1, 32, 2))
        model = FlowActionExpert(896, 256, cfg).eval()
        result = model.sample(batch, torch.Generator().manual_seed(42))
        self.assertEqual(result["trajectories"].shape, (1, 1, 1, 30, 2))
        self.assertTrue(torch.isfinite(result["trajectories"]).all())

    def test_processed_marker_and_no_full_prompt_replay(self):
        response = "<landmark>[]</landmark><local>unknown</local><Route>forward</Route><waypoint>[]</waypoint><REFLINE>"

        class Tokenizer:
            eos_token_id = 0

            def encode(self, text, **kwargs):
                return [ord(c) for c in text]

            def decode(self, ids, **kwargs):
                return "".join(chr(i) for i in ids)

        class LM(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []
                self.embedding = nn.Embedding(128, 4)

            def get_input_embeddings(self):
                return self.embedding

            def forward(
                self, input_ids=None, inputs_embeds=None, past_key_values=None, **kwargs
            ):
                n = (
                    inputs_embeds.shape[1]
                    if inputs_embeds is not None
                    else input_ids.shape[1]
                )
                previous = past_key_values[-1][0].shape[2] if past_key_values else 0
                self.calls.append(n)
                total = previous + n
                logits = torch.zeros(1, n, 128)
                next_token = ord(response[total - 1]) if total <= len(response) else 0
                logits[:, -1, next_token] = 10
                value = float(input_ids[0, -1]) if input_ids is not None else 0.0
                hidden = torch.full((1, n, 4), value)
                k = torch.ones(1, 1, total, 2)
                return SimpleNamespace(
                    logits=logits, hidden_states=[hidden], past_key_values=((k, k),)
                )

        cfg = tiny_config(max_new_tokens=200)
        adapter = InternVLReference.__new__(InternVLReference)
        nn.Module.__init__(adapter)
        adapter.cfg = cfg
        adapter.tokenizer = Tokenizer()
        adapter.query_ids = adapter.tokenizer.encode(QUERY)
        adapter.backbone = nn.Module()
        adapter.backbone.language_model = LM()
        adapter.decoder = ReferenceDecoder(4, cfg)
        adapter.build_prompt = lambda *args: ("P", [])
        adapter._embeddings = lambda ids, images: (
            adapter.backbone.language_model.embedding(ids)
        )
        seen = []
        hook = adapter.decoder.register_forward_pre_hook(
            lambda m, args: seen.append(args[0].clone())
        )
        result = adapter.generate_record({})
        hook.remove()
        self.assertTrue(result["query_complete"])
        self.assertIsNotNone(result["parsed"])
        self.assertEqual(float(seen[0][0, 0]), ord(">"))
        self.assertEqual(
            adapter.backbone.language_model.calls, [1] * (len(response) + 1)
        )
        self.assertEqual(result["raw_kv"].shape[1], len(response) + 1)
        self.assertIsNone(parse_response(response.replace("<Route>", "<route>")))


if __name__ == "__main__":
    unittest.main()
