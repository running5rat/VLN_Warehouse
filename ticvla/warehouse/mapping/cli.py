"""Capture replay, persistence, registration and MapView export.

python -m ticvla.warehouse.mapping.cli --help
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import ImageDraw

from ..geometry import render_map_image
from .store import MapStore, SemanticEvent, SourceObservation


def import_episode(store, path, submap_id=None):
    path = Path(path).resolve()
    metadata = json.loads(path.read_text())
    frames = [
        json.loads(x)
        for x in (path.parent / metadata["frames_file"]).read_text().splitlines()
    ]
    ids = []
    for i, frame in enumerate(frames):
        image_id = f"{metadata['episode_id']}:{i:08d}"
        source = SourceObservation(
            image_id,
            str(path.parent / frame["rgb"]),
            str(path.parent / frame["depth"]),
            frame["K"],
            frame["T_body_camera"],
            frame["T_odom_body"],
            frame["t_obs"],
            frame["odom_epoch"],
            submap_id or frame["odom_epoch"],
            metadata["task"]["task_id"],
        )
        store.add_observation(source)
        ids.append(image_id)
    return ids


def export_view(store, image_id, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    view = store.view(image_id)
    np.savez_compressed(output / "map.npz", grid=view["grid"], age=view["age"])
    canvas = render_map_image(view["grid"], 0.1)
    draw = ImageDraw.Draw(canvas)
    for lm in view["landmarks"]:
        x, y = lm["position"][:2]
        px = canvas.width / 2 - y / 16 * canvas.width
        py = (canvas.height - 40) / 2 - x / 16 * (canvas.height - 40)
        if 0 <= px < canvas.width and 0 <= py < canvas.height - 40:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill="orange")
            draw.text((px + 4, py), lm["name"] or lm["category"], fill="blue")
    canvas.save(output / "map.png")
    metadata = {k: v for k, v in view.items() if k not in ("grid", "age")}
    metadata.update(
        image_id=image_id,
        image=str((output / "map.png").resolve()),
        arrays=str((output / "map.npz").resolve()),
    )
    (output / "mapview.json").write_text(json.dumps(metadata, indent=2))
    (output / "map-summary.json").write_text(json.dumps(store.summary(), indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument(
        "--oracle", action="store_true", help="Explicitly isolated test-only map"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("import")
    capture.add_argument("--episode", required=True)
    capture.add_argument("--submap")
    segment = commands.add_parser("segment")
    segment.add_argument("--events", required=True)
    segment.add_argument("--weights", required=True)
    segment.add_argument("--device", default="cuda:0")
    export = commands.add_parser("export")
    export.add_argument("--image-id", required=True)
    export.add_argument("--output", required=True)
    register = commands.add_parser("register")
    register.add_argument("--source", required=True)
    register.add_argument("--target", required=True)
    revoke = commands.add_parser("revoke")
    revoke.add_argument("--constraint-id", required=True)
    revoke_landmark = commands.add_parser("revoke-landmark")
    revoke_landmark.add_argument("--landmark-id", required=True)
    revoke_landmark.add_argument("--reason", required=True)
    args = parser.parse_args()
    store = MapStore(args.store, oracle=args.oracle)
    try:
        if args.command == "import":
            result = import_episode(store, args.episode, args.submap)
        elif args.command == "segment":
            if args.oracle:
                raise ValueError("Real segmenter only writes deployment maps")
            from .segmentation import YOLOESegmenter, process_event

            backend = YOLOESegmenter(args.weights, args.device)
            result = [
                process_event(store, SemanticEvent(**e), backend)
                for e in json.loads(Path(args.events).read_text())
            ]
        elif args.command == "export":
            result = export_view(store, args.image_id, args.output)
        elif args.command == "register":
            result = store.register(args.source, args.target)
        elif args.command == "revoke":
            store.revoke(args.constraint_id)
            result = store.summary()
        else:
            store.revoke_landmark(args.landmark_id, args.reason)
            result = store.summary()
        print(json.dumps(result, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
