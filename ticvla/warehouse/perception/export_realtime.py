"""Export fixed-class YOLO-E engines in an isolated TensorRT environment."""

import argparse
from pathlib import Path
import shutil
from .realtime import CORE, VOCAB_VERSION
from ..mapping.segmentation import YOLOESegmenter
from ..mapping.semantic_frame import write_json
from ..schema import file_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    import torch
    import tensorrt

    torch.set_num_threads(2)
    for name in ("11s", "v8s", "v8l"):
        out = a.root / "engines" / name
        out.mkdir(parents=True, exist_ok=True)
        weights = Path("/data/ticvla/models") / f"yoloe-{name}-seg.pt"
        model = YOLOESegmenter(weights)
        model._set_text_classes(list(CORE))
        target = out / "model.pt"
        model.model.save(target)
        # Ultralytics derives the output path from pt_path.
        model.model.model.pt_path = str(target)
        model.model.ckpt_path = str(target)
        exported = Path(
            model.model.export(
                format="engine",
                imgsz=640,
                batch=1,
                half=True,
                device=0,
                dynamic=False,
                simplify=False,
                workspace=2,
                verbose=False,
            )
        )
        engine = out / "model.engine"
        if exported.resolve() != engine.resolve():
            shutil.copy2(exported, engine)
        write_json(
            Path(str(engine) + ".json"),
            dict(
                categories=list(CORE),
                vocabulary_version=VOCAB_VERSION,
                sha256=file_sha256(engine),
                source_weights=str(weights),
                weights_sha256=file_sha256(weights),
                tensorrt=tensorrt.__version__,
                torch=torch.__version__,
                dtype="fp16",
                imgsz=640,
                parameters=sum(x.numel() for x in model.model.model.parameters()),
            ),
        )
        print("EXPORTED", engine, flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
