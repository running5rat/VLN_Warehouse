"""CLI for human-first data preparation; use --help for non-training commands."""

import argparse
import json
from pathlib import Path

from .dataset_tools.core import extract, export_project, read_json, validate_v2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--episode", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--count", type=int, default=10)
    for command in ("annotate", "export"):
        c = sub.add_parser(command)
        c.add_argument("--project", required=True)
        if command == "export": c.add_argument("--output", required=True)
        else: c.add_argument("--port", type=int, default=8767)
    c = sub.add_parser("validate")
    c.add_argument("--record", required=True)
    c = sub.add_parser("audit")
    c.add_argument("--root", default="/data/ticvla/dataset")
    c.add_argument("--output", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--catalog", required=True)
    c.add_argument("--output", required=True)
    c = sub.add_parser("teacher")
    c.add_argument("--project", required=True)
    c.add_argument("--config", required=True)
    c.add_argument("--output", required=True)
    c = sub.add_parser("semantic-prepare", help="Prepare <=6 causal text-teacher samples; no API call")
    src = c.add_mutually_exclusive_group(required=True)
    src.add_argument("--project")
    src.add_argument("--dynanav-root")
    c.add_argument("--episode-id", default="warehouse_14")
    c.add_argument("--indices", type=int, nargs="+", default=[0, 3, 4, 6, 8, 9])
    c.add_argument("--context-mode", choices=["recursive", "gold_context"], default="recursive")
    c.add_argument("--output", required=True)
    c = sub.add_parser("semantic-teacher", help="One semantic request per source; archive unreviewed proposals")
    c.add_argument("--bundle", required=True)
    c.add_argument("--config", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--prepare-only", action="store_true")
    c = sub.add_parser("semantic-session", help="Prepare/import conversation-assisted answers; no API calls")
    c.add_argument("--output", required=True)
    mode = c.add_mutually_exclusive_group()
    mode.add_argument("--bundle", help="Start a new recursive session")
    mode.add_argument("--response", help="Import one source-bound answer JSON")
    c = sub.add_parser("oracle")
    c.add_argument("--request", required=True)
    c.add_argument("--solver", required=True, help="JSON configuration containing official ALTRO worker argv")
    c.add_argument("--output", required=True)
    c = sub.add_parser("readiness")
    c.add_argument("--output", required=True)
    c.add_argument("--cache", help="Versioned actual VLM KV + deployment Flow batch .pt")
    a = p.parse_args()
    if a.command == "extract": result = extract(a.episode, a.output, count=a.count)
    elif a.command == "export": result = export_project(a.project, a.output)
    elif a.command == "validate":
        validate_v2(read_json(a.record))
        result = {"valid": True}
    elif a.command == "audit":
        from .dataset_tools.catalog import audit
        result = audit(a.root, a.output)
        result = {k:{q:v[q] for q in ("episode_count", "eligible_episodes", "status")} for k,v in result["sources"].items()}
    elif a.command == "convert":
        from .dataset_tools.catalog import convert_audited
        result = convert_audited(a.catalog, a.output)
    elif a.command == "teacher":
        from .dataset_tools.teacher import run_teacher
        result = run_teacher(a.project, a.config, a.output)
    elif a.command == "semantic-prepare":
        if a.project:
            from .dataset_tools.semantic_annotation import prepare_manual
            result = prepare_manual(a.project, a.output, a.indices, a.context_mode)
        else:
            if a.context_mode != "recursive":
                p.error("DynaNav has no human structured labels for gold_context")
            from .dataset_tools.dynanav_semantic import prepare_dynanav
            result = prepare_dynanav(a.dynanav_root, a.episode_id, a.output)
    elif a.command == "semantic-teacher":
        from .dataset_tools.semantic_annotation import run_pilot
        result = run_pilot(a.bundle, a.config, a.output, a.prepare_only)
    elif a.command == "semantic-session":
        from .dataset_tools.session_teacher import start_session, prepare_next, submit_answer
        result = (start_session(a.bundle, a.output) if a.bundle else
                  submit_answer(a.output, a.response) if a.response else prepare_next(a.output))
    elif a.command == "oracle":
        from .dataset_tools.oracle import run_oracle
        result = run_oracle(a.request, a.solver, a.output)
    elif a.command == "readiness":
        from .dataset_tools.flow_training import readiness
        result = readiness(a.output, a.cache)
    else:
        import sys
        from DynaNav.warehouse_loop.annotation import main as serve
        project = Path(a.project).resolve()
        if project.name != "annotation":
            p.error("Project should be <session>/annotation")
        sys.argv = [sys.argv[0], "--mode", "replay", "--output", str(project.parent), "--port", str(a.port)]
        serve()
        return
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
