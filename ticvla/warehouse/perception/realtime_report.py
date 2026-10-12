"""Render the frozen experiment, with failed and unselected instances visible."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .report import render_persistent_maps
from ..mapping.semantic_frame import write_json


def overlay(folder):
    data = json.loads((folder / "frame.json").read_text())
    image = np.array(Image.open(data["source"]["rgb"]).convert("RGB"))
    colors = [(50, 180, 255), (255, 160, 20), (170, 100, 240), (60, 205, 100)]
    for i, row in enumerate(data["rows"]):
        mask = np.load(folder / f"mask-{i:03}.npz")["mask"]
        color = colors[i % len(colors)] if row["selected"] else (170, 170, 170)
        image[mask] = (image[mask] * 0.65 + np.array(color) * 0.35).astype("uint8")
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for i, row in enumerate(data["rows"]):
        color = colors[i % len(colors)] if row["selected"] else (170, 170, 170)
        draw.rectangle(row["bbox"], outline=color, width=2)
        draw.text(
            row["bbox"][:2],
            f"{i}: {row['category']} [{row['result']['status']}]",
            fill=color,
            stroke_width=1,
            stroke_fill="black",
        )
    for t in (data.get("binding") or {}).get("texts", []):
        xy = [tuple(p) for p in t["polygon"]]
        draw.line(xy + xy[:1], fill="yellow", width=1)
        draw.text(
            xy[0],
            f"{t['text']} [{t['status']}]",
            fill="yellow",
            stroke_width=1,
            stroke_fill="black",
        )
    image.save(folder / "segmentation.png")
    return image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--reuse-images", action="store_true")
    a = p.parse_args()
    r = a.root
    candidates = ["11s-engine", "v8s-engine", "v8l-engine"]
    reports = {}
    for name in candidates:
        score = json.loads((r / name / "evaluation-score.json").read_text())
        variants = {}
        for row in score["rows"]:
            v = variants.setdefault(
                row["variant"],
                dict(frames=0, targets=0, matched=0, readable=0, correct_3d=0),
            )
            v["frames"] += 1
            for key in ("targets", "matched", "readable", "correct_3d"):
                v[key] += row[key]
        score["variants"] = variants
        score["quality_gate"] = bool(
            all(variants[v]["correct_3d"] > 0 for v in ("original", "color", "surface"))
            and score["wrong_binding_observations"] == 0
            and score["false_merges"] == 0
        )
        reports[name] = score
        if not a.reuse_images:
            render_persistent_maps(r, name)
    write_json(r / "quality-report.json", reports)
    sections = [
        "<h1>固定词表实时感知：留出集结果</h1>",
        "<p>灰色 mask：未被 VLM 选择；彩色：被选择的候选。黄色为文字读数及绑定状态。所有候选保持 tentative。真实采集、模型原文与失败结果均保留。</p>",
        '<p><a href="quality-report.json">质量报告</a> · <a href="freeze.json">冻结配置</a> · <a href="acceptance.json">总体验收</a></p>',
        "<table><tr><th>模型</th><th>目标检出</th><th>正确三维编号观测</th><th>错绑</th><th>错合并</th><th>质量门槛</th></tr>",
    ]
    for name, s in reports.items():
        values = [
            name,
            f"{s['matched']}/{s['targets']}",
            f"{s['correct_3d']}/{s['readable_unique']}",
            s["wrong_binding_observations"],
            s["false_merges"],
            s["quality_gate"],
        ]
        sections.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(x))}</td>" for x in values)
            + "</tr>"
        )
    sections.append(
        "</table><p>检出统计限于独立标注的牌子与容器；其余设施无完整真值，不能由这些数字推断总体物体准确率。离线推理时间不代表 Isaac 同时运行时的发布延迟。</p>"
    )
    if (r / "acceptance.json").exists():
        acceptance = json.loads((r / "acceptance.json").read_text())
        sections.insert(
            1,
            "<p style='color:#ffd080;font-size:20px'>"
            + (
                "总体通过"
                if acceptance["overall_pass"]
                else "实现可运行；质量与实时延迟验收未通过"
            )
            + "</p>",
        )
        sections.append(
            "<h2>真实 Isaac 绕环</h2><p>门槛：完成频率 ≥4.75 Hz；mask 发布 p95 ≤0.5 s；OCR 绑定 p95 ≤2 s。迟到结果不能补算通过。</p><table><tr><th>场景</th><th>RGB-D Hz</th><th>几何 Hz</th><th>分割 Hz</th><th>mask p95 秒</th><th>文字 p95 秒</th><th>OCR 完成率</th></tr>"
        )
        for variant, result in acceptance["realtime"].items():
            values = [variant] + [
                f"{result[k]:.3f}"
                for k in (
                    "rgbd_hz",
                    "geometry_hz",
                    "mask_hz",
                    "mask_p95_s",
                    "ocr_p95_s",
                    "ocr_completion_fraction",
                )
            ]
            sections.append("<tr>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>")
        sections.append("</table><div class='panes'>")
        for variant in acceptance["realtime"]:
            base = f"live-{variant}-optimized"
            sections.append(
                f"<div>{variant}<img loading='lazy' src='{base}/mapping/map/semantic_bev.png'><a href='{base}/mapping/map/semantic_3d.png'>3D</a> · <a href='{base}/mapping/realtime-report.json'>实时报告</a></div>"
            )
        sections.append(
            "</div><p><a href='manual-review.json'>目视检查范围</a> · <a href='recovery-report.json'>重启与去重</a> · <a href='implementation-notes.md'>实现与复现命令</a></p>"
        )
        sections.append(
            "<p>固定第 200 帧："
            + str(acceptance["frame_200"]["accepted_instances"])
            + " 个实例入图；没有用人工提示替换失败回答。</p>"
        )
    for variant in reports[candidates[0]]["variants"]:
        sections.append(f"<h2>{variant}</h2><div class='panes'>")
        for name in candidates:
            sections.append(
                f"<div>{name}<img loading='lazy' src='{name}/maps/{variant}/semantic_bev.png'><a href='{name}/maps/{variant}/semantic_3d.png'>3D</a></div>"
            )
        sections.append("</div>")
    manifest = json.loads((r / "manifest.json").read_text())
    overview = None
    for row in manifest["rows"]:
        if row["variant"] == "development":
            continue
        key = row["key"]
        qa = json.loads((r / "qa" / key / "result.json").read_text())
        sections.append(
            f"<details><summary>{key} · {html.escape(str(qa['parsed']))}</summary><div class='panes'>"
        )
        images = []
        for name in candidates:
            folder = r / name / key
            im = (
                Image.open(folder / "segmentation.png").convert("RGB")
                if a.reuse_images and (folder / "segmentation.png").exists()
                else overlay(folder)
            )
            im.thumbnail((640, 480))
            images.append(im)
            sections.append(
                f"<div>{name}<img loading='lazy' src='{name}/{key}/segmentation.png'><a href='{name}/{key}/frame.json'>源帧、事件与文字</a></div>"
            )
        sections.append(f"</div><a href='qa/{key}/result.json'>VLM 原文</a></details>")
        if overview is None:
            overview = Image.new("RGB", (1920, 560), "#182531")
            draw = ImageDraw.Draw(overview)
            for i, (name, im) in enumerate(zip(candidates, images)):
                draw.text((i * 640 + 12, 10), f"{key}: {name}", fill="white")
                overview.paste(im, (i * 640, 45))
            draw.text(
                (12, 535),
                "Grey: detected but not selected. Coloured: tentative. Quality/realtime failures are reported, not hidden.",
                fill="white",
            )
            overview.save(r / "overview.png")
    (r / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><style>body{font:15px sans-serif;background:#182531;color:white;margin:20px}a{color:#8cd8ff}.panes{display:grid;grid-template-columns:repeat(3,1fr)}img{width:100%}td,th{padding:8px;border:1px solid #888}details{margin:12px}</style>"
        + "\n".join(sections)
    )


if __name__ == "__main__":
    main()
