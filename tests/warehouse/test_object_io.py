import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from ticvla.warehouse.object_io import (
    add_mode_argument,
    parse_object_list,
    run_object_qa,
    validate_mode_args,
    category_policy,
)
from ticvla.warehouse.structured_io import save_result


class ObjectQATests(unittest.TestCase):
    def test_navigation_filters_background_but_keeps_specific_cues(self):
        classes = [
            "floor",
            "gray floor",
            "warehouse floor",
            "wall",
            "black ceiling",
            "warehouse",
            "floor markings",
            "ceiling lights",
            "wall signs",
            "wall panel",
            "warehouse shelves",
            "fire extinguisher",
        ]
        kept, filtered = category_policy(classes, "navigation")
        self.assertEqual(kept, classes[6:])
        self.assertEqual([r["category"] for r in filtered], classes[:6])
        self.assertEqual(category_policy(classes, "all"), (classes, []))
        self.assertEqual(
            category_policy(
                [
                    "black floor with yellow lines",
                    "white ceiling with lights",
                    "floors",
                ],
                "navigation",
            )[0],
            [],
        )

    def test_lists_normalize_without_three_category_limit(self):
        self.assertEqual(
            parse_object_list(
                "- Shelves\n* BOXES\n• fire   extinguisher\n1. Wall\n2) Ceiling lights\n  boxes  "
            ),
            ["shelves", "boxes", "fire extinguisher", "wall", "ceiling lights"],
        )
        self.assertEqual(parse_object_list("Boxes\nShelves"), ["boxes", "shelves"])
        self.assertEqual(parse_object_list("[]"), [])
        for text in (
            "",
            "Here are the objects:\n- boxes",
            "<answer>(1,2)</answer>",
            "- boxes\nI see: shelves",
            '["boxes"]',
        ):
            with self.assertRaises(ValueError):
                parse_object_list(text)

    def test_mode_default_and_conflict(self):
        parser = argparse.ArgumentParser()
        add_mode_argument(parser)
        parser.add_argument("--object-evidence", action="store_true")
        self.assertEqual(parser.parse_args([]).llm_mode, "objects-only")
        with self.assertRaises(SystemExit):
            validate_mode_args(parser, parser.parse_args(["--object-evidence"]))
        validate_mode_args(
            parser, parser.parse_args(["--llm-mode", "structured", "--object-evidence"])
        )

    def test_one_call_source_binding_and_no_navigation_outputs(self):
        model = Mock()
        model.describe_objects.return_value = dict(
            image_id="a:0",
            answer="- Boxes\n- shelves",
            token_count=6,
            max_new_tokens=160,
        )
        model.infer.side_effect = AssertionError("Second inference forbidden")
        obs = dict(rgb_history=[dict(image_id="a:0", path="rgb.png")])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result, events, timings = run_object_qa(model, obs, "req", root)
            self.assertTrue(result["valid"])
            self.assertEqual(result["llm_calls"], 1)
            self.assertFalse(result["structured_inference_executed"])
            self.assertEqual(
                [
                    (e.image_id, e.category, e.source, e.static_confirmed)
                    for e in events
                ],
                [("a:0", "boxes", "llm", False), ("a:0", "shelves", "llm", False)],
            )
            self.assertTrue(
                all(e.navigation_role is None and e.reason is None for e in events)
            )
            self.assertGreater(timings["object_qa_s"], 0)
            save_result(root / "llm", {}, result, events=events)
            response = json.loads((root / "llm/response.json").read_text())
            for field in ("snapshot", "task_state", "reference_xy", "z_sem"):
                self.assertNotIn(field, response)
            self.assertFalse((root / "llm/snapshot.npz").exists())
            model.describe_objects.assert_called_once_with(obs)
            model.infer.assert_not_called()
            model.build_prompt.assert_not_called()
            model.decoder.assert_not_called()
            model.compressor.assert_not_called()
            # Deterministic event identity permits existing consumer replay.
            again, repeated, _ = run_object_qa(model, obs, "req", root)
            self.assertEqual([asdict(e) for e in events], [asdict(e) for e in repeated])

    def test_failed_and_empty_answers_do_not_publish_events(self):
        base = dict(image_id="a:0", answer="- boxes", token_count=3, max_new_tokens=160)
        obs = dict(rgb_history=[dict(image_id="a:0", path="rgb.png")])
        for change in (
            dict(image_id="a:1"),
            dict(token_count=160),
            dict(answer=""),
            dict(answer="Here are objects:\n- boxes"),
            dict(answer="[]"),
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                model = Mock()
                model.describe_objects.return_value = dict(base, **change)
                result, events, _ = run_object_qa(model, obs, "req", temp)
                self.assertEqual(events, [])
                self.assertEqual(result["valid"], change.get("answer") == "[]")
                self.assertTrue((Path(temp) / "object-answer.json").exists())
                model.describe_objects.assert_called_once()
                model.infer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
