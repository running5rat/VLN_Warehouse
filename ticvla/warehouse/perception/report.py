"""Portable comparison artifacts; uses measured results, never model-made truth."""

import html
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ..mapping.semantic_frame import write_json


def render_persistent_maps(root, backend):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from ..mapping import MapStore

    for path in sorted((root / backend / "maps").iterdir()):
        if path.name == "development":
            continue
        store = MapStore(path)
        try:
            observations = sorted(
                store.records("observations"), key=lambda r: r["t_obs"]
            )
            if len({r["submap_id"] for r in observations}) != 1:
                continue  # Never overlay independent, unregistered odom epochs.
            landmarks = store.observed_landmarks()
            background = []
            for obs in observations:
                points = store.points(obs)
                points = points[(points[:, 2] > 0.15) & (points[:, 2] < 3.5)]
                background.append(points[:: max(1, len(points) // 2000)])
            geometry = np.concatenate(background) if background else np.empty((0, 3))
            if len(geometry) > 200000:
                geometry = geometry[:: int(np.ceil(len(geometry) / 200000))]
            trail = np.array(
                [np.asarray(o["T_odom_body"])[:3, 3] for o in observations]
            )
            for dimensions in (2, 3):
                fig = plt.figure(figsize=(12, 9))
                ax = fig.add_subplot(111, projection="3d" if dimensions == 3 else None)
                if len(geometry):
                    ax.scatter(
                        *geometry[:, :dimensions].T, s=0.3, c="#aeb8c1", alpha=0.4
                    )
                if len(trail):
                    ax.plot(
                        *trail[:, :dimensions].T,
                        color="#2574bd",
                        label="Source trajectory",
                    )
                for i, lm in enumerate(landmarks):
                    points = store.points(lm)
                    points = points[:: max(1, len(points) // 1500)]
                    color = plt.get_cmap("tab20")(i % 20)
                    ax.scatter(*points[:, :dimensions].T, s=2, color=color)
                    if i < 24:
                        label = f"{i}: {lm['category']} {lm.get('identity_text') or ''}\n{lm.get('identity_status', 'unknown')}"
                        ax.text(*lm["position"][:dimensions], label, fontsize=6)
                ax.set_xlabel("Odom X (m)")
                ax.set_ylabel("Odom Y (m)")
                if dimensions == 3:
                    ax.set_zlabel("Odom Z (m)")
                else:
                    ax.set_aspect("equal", adjustable="datalim")
                ax.set_title(
                    f"{backend} / {path.name}: accumulated observations\nTentative objects; not verified static anchors"
                )
                fig.tight_layout()
                fig.savefig(
                    path
                    / ("semantic_bev.png" if dimensions == 2 else "semantic_3d.png"),
                    dpi=130,
                )
                plt.close(fig)
            write_json(path / "observed-landmarks.json", landmarks)
            write_json(
                path / "needs-reobservation.json",
                [
                    {
                        k: lm.get(k)
                        for k in (
                            "landmark_uid",
                            "category",
                            "submap_id",
                            "position",
                            "identity_status",
                            "identity_candidates",
                            "reobservation_reasons",
                        )
                    }
                    for lm in landmarks
                    if lm.get("needs_reobservation")
                ],
            )
        finally:
            store.close()


def render_report(root):
    root = Path(root)
    data = json.loads((root / "comparison.json").read_text())
    font = ImageFont.truetype("DejaVuSans.ttf", 24)
    small = ImageFont.truetype("DejaVuSans.ttf", 18)
    # The exact same frozen first RGB is used for this representative comparison.
    key = "color-00000000"
    canvas = Image.new("RGB", (1800, 895), "#182531")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (25, 18),
        "VLM-selected landmarks: masks + text + source-depth projection",
        font=font,
        fill="white",
    )
    for i, name in enumerate(("yoloe-ppocr", "florence2")):
        render_persistent_maps(root, name)
        stats = data["backends"][name]
        image = root / name / key / "segmentation.png"
        if image.exists():
            im = Image.open(image).convert("RGB")
            im.thumbnail((850, 650))
            canvas.paste(im, (25 + i * 900, 85))
        draw.text((25 + i * 900, 55), name, font=font, fill="white")
        draw.text(
            (25 + i * 900, 748),
            f"Readable unique targets: {stats['readable_unique_targets']} | correct 3D observations: {stats['correct_3d']}",
            font=small,
            fill="white",
        )
        draw.text(
            (25 + i * 900, 778),
            f"Selection gate: {'PASS' if stats['gate_pass'] else 'FAIL'} | evaluated frames: {stats['frames']}/{stats['planned_frames']}",
            font=small,
            fill="white",
        )
    draw.text(
        (25, 815),
        "Confirmed unique identities: "
        + " | ".join(
            f"{name}: {stats['confirmed_correct_unique_landmarks']}"
            for name, stats in data["backends"].items()
        ),
        font=small,
        fill="#ffd780",
    )
    draw.text(
        (25, 850),
        "Coloured masks are tentative observations, not verified static anchors. See comparison.json for denominators.",
        font=small,
        fill="#ffd780",
    )
    canvas.save(root / "comparison-overview.png")
    rows = []
    for name, stats in data["backends"].items():
        for variant, v in stats["variants"].items():
            rows.append(
                "<tr>"
                + "".join(
                    "<td>" + html.escape(str(x)) + "</td>"
                    for x in (
                        name,
                        variant,
                        v["matched_instances"],
                        v["target_observations"],
                        v["readable_targets"],
                        v["correct_3d"],
                        v["perception_latency"]["p95_s"],
                    )
                )
                + "</tr>"
            )
    links = []
    for row in data["backends"]["yoloe-ppocr"]["rows"]:
        key = row["key"]
        cells = [html.escape(key)]
        for name in ("yoloe-ppocr", "florence2"):
            cells.append(
                f'<a href="{name}/{key}/overview.png">overview</a> · <a href="{name}/{key}/report.json">JSON</a>'
            )
        links.append("<tr>" + "".join("<td>" + x + "</td>" for x in cells) + "</tr>")
    text = """<!doctype html><meta charset="utf-8"><title>Landmark perception comparison</title>
<style>body{font:16px sans-serif;max-width:1400px;margin:24px auto;padding:12px;background:#edf2f6;color:#172638}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:8px;border:1px solid #a6b4c2}img{max-width:100%}pre{white-space:pre-wrap}</style>
<h1>导航地标感知对照</h1><p>两种后端共用同一份真实 InternVL 类别回答。文字身份一致性与永久静态锚点资格分开。</p>"""
    text += f'<p>通过选优门槛的后端：{html.escape(str(data["winner"]))}。</p><img src="comparison-overview.png">'
    text += (
        "<p>正确的单帧编号观测不等于跨帧确认身份；确认唯一身份数量："
        + html.escape(
            str(
                {
                    n: s["confirmed_correct_unique_landmarks"]
                    for n, s in data["backends"].items()
                }
            )
        )
        + "。</p>"
    )
    text += '<p><a href="comparison.json">完整统计</a> · <a href="manifest.json">冻结帧清单</a> · <a href="evaluation/annotations.json">独立标注</a> · <a href="evaluation/visual-review-decisions.json">源图可读性检查</a></p>'
    text += (
        "<table><tr><th>后端</th><th>场景</th><th>匹配实例</th><th>标注实例</th><th>可读唯一编号观测</th><th>正确三维身份观测</th><th>感知 p95 秒</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    for backend in data["backends"]:
        for path in sorted((root / backend / "maps").glob("*/semantic_bev.png")):
            relative = path.parent.relative_to(root)
            text += (
                f"<p>{backend} / {path.parent.name} 候选地图："
                f'<a href="{relative}/semantic_bev.png">BEV</a> · '
                f'<a href="{relative}/semantic_3d.png">三维图</a> · '
                f'<a href="{relative}/observed-landmarks.json">地标记录</a> · '
                f'<a href="{relative}/needs-reobservation.json">待补看对象</a></p>'
            )
    for name in ("frame200-a", "frame200-b"):
        if (root / name / "overview.png").exists():
            text += f'<p><a href="{name}/overview.png">第 200 帧 {name} 总览</a> · <a href="{name}/report.json">JSON</a></p>'
    if (root / "diagnostic-summary.json").exists():
        text += '<p><a href="diagnostic-summary.json">人工类别诊断（不计入端到端成绩）</a></p>'
        for backend in data["backends"]:
            for path in sorted(
                (root / ("diagnostic-" + backend)).glob("*/overview.png")
            ):
                text += f'<p><a href="{path.relative_to(root)}">{backend} / {path.parent.name} 人工提示总览</a></p>'
    live_reports = []
    for path in sorted(
        root.glob("live-validation*.json"), key=lambda p: p.stat().st_mtime
    ):
        live = json.loads(path.read_text())
        live_reports.append((path, live))
        text += f'<p><a href="{path.name}">绕环：渲染 {live.get("render_hz", 30)} Hz，功能检查 {live.get("functional_pass")}</a></p>'
    if live_reports:
        latest = live_reports[-1][1]
        text += "<p><strong>最终绕环结果：两种场景运动期间均入图的功能验收"
        text += "通过" if latest.get("functional_pass") else "未通过"
        text += "。语义采样目标 1 Hz 不代表模型完成频率。</strong></p>"
        text += "<table><tr><th>场景</th><th>运动中几何 Hz</th><th>运动中语义 Hz</th><th>成功语义批次</th><th>几何全部消费</th></tr>"
        for run in latest["runs"]:
            rate = run["actual_motion_hz"]
            text += (
                f"<tr><td>{html.escape(run['variant'])}</td>"
                f"<td>{rate['geometry']:.3f}</td><td>{rate['semantic']:.4f}</td>"
                f"<td>{run['accepted_semantic_commits_during_motion']}</td>"
                f"<td>{run['all_geometry_consumed']}</td></tr>"
            )
        text += "</table>"
    text += "<p>耗时来自同一 GPU 上各 3 个场景分组 worker；不代表实时单 worker 帧率。未标注普通物品不默认算作正确或误检。</p>"
    text += (
        "<table><tr><th>帧</th><th>YOLO-E＋PP-OCRv5</th><th>Florence-2</th></tr>"
        + "".join(links)
        + "</table>"
    )
    (root / "index.html").write_text(text)
    summary = {
        name: {
            k: stats[k]
            for k in (
                "complete",
                "gate_pass",
                "qa_valid",
                "targets",
                "readable_unique_targets",
                "correct_3d",
                "confirmed",
                "confirmed_correct_unique_landmarks",
                "confirmed_unique_identity_coverage",
                "wrong_confirmed",
                "unscored_confirmed",
                "false_merge",
                "latency",
                "variants",
            )
        }
        for name, stats in data["backends"].items()
    }
    write_json(root / "summary.json", dict(winner=data["winner"], backends=summary))
    validation = {}
    for backend in data["backends"]:
        replay = root / ("replay-validation-" + backend) / "report.json"
        validation[backend] = (
            json.loads(replay.read_text()) if replay.exists() else None
        )
    test_log = root / "unit-tests.log"
    write_json(
        root / "acceptance.json",
        dict(
            model_comparison_complete=all(
                r["complete"] for r in data["backends"].values()
            ),
            unit_test_log=str(test_log),
            unit_tests_passed=test_log.exists()
            and test_log.read_text().rstrip().endswith("OK"),
            restart_replay_validation={
                name: None
                if value is None
                else {k: value[k] for k in ("passed", "frames", "latency")}
                for name, value in validation.items()
            },
            selected_backend=data["winner"],
            live_rollout_status=live_reports[-1][1]["status"]
            if live_reports
            else ("pending" if data["winner"] else "not_eligible"),
            live_functional_pass=live_reports[-1][1].get("functional_pass")
            if live_reports
            else None,
            live_validation_report=str(live_reports[-1][0]) if live_reports else None,
            live_5hz_geometry_1hz_semantics_verified=bool(
                live_reports and live_reports[-1][1]["runs"]
            )
            and all(r["all_requested_rates_met"] for r in live_reports[-1][1]["runs"]),
            quality_scope="Measured on installed warehouse targets only; unreadable text excluded from readable-text accuracy. No long-term staticness or navigation success claim.",
            identity_scope="Correct source observations and cross-frame identity confirmations are reported separately.",
        ),
    )


if __name__ == "__main__":
    import sys

    render_report(Path(sys.argv[1]))
