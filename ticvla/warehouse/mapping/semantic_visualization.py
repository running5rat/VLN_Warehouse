"""Single-observation overlays; tentative evidence is never a static MapView."""

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..geometry import transform_points


def render_semantic_frame(
    store, image_id, report, llm_result, root, *, weight_mode="unspecified"
):
    root = Path(root)
    obs, _ = store.source_arrays(image_id)
    rgb = np.array(Image.open(obs["rgb"]).convert("RGB"))
    inverse = np.linalg.inv(np.asarray(obs["T_odom_body"]))
    geometry = transform_points(store.points(obs), inverse)
    # Ground is shown by the grid; only observed obstacle geometry is plotted.
    geometry = geometry[(geometry[:, 2] > 0.15) & (geometry[:, 2] < 3.5)]
    geometry = geometry[np.linalg.norm(geometry[:, :2], axis=1) < 16]
    geometry = geometry[:: max(1, len(geometry) // 30000)]
    colors = plt.get_cmap("tab20")
    rows = []
    for row in report["instances"]:
        with np.load(root / "segmentation" / row["mask_file"]) as arr:
            mask = arr["mask"]
        with np.load(root / "segmentation" / row["points_file"]) as arr:
            points = transform_points(arr["points"], inverse)
        rows.append((row, mask, points, colors(row["instance_index"] % 20)))

    def draw_masks(ax):
        blended = rgb.astype(float).copy()
        for row, mask, _, color in rows:
            if mask.shape == rgb.shape[:2] and row["result"]["status"] == "accepted":
                blended[mask] = 0.55 * blended[mask] + 0.45 * np.array(color[:3]) * 255
        ax.imshow(blended.astype(np.uint8))
        for row, _, _, color in rows:
            x1, y1, x2, y2 = row["bbox"]
            accepted = row["result"]["status"] == "accepted"
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor=color if accepted else "red",
                    linewidth=1.5,
                    linestyle="-" if accepted else "--",
                )
            )
            label = (
                f"#{row['instance_index']} {row['category']} {row.get('identity_text') or ''} "
                + (
                    f"{row['confidence']:.2f}"
                    if row["confidence"] is not None
                    else "confidence unavailable"
                )
            )
            if not accepted:
                label += "\nREJECTED: " + str(row["result"]["reason"])
            ax.text(
                x1,
                max(y1 - 4, 12),
                label,
                fontsize=7,
                color="white",
                bbox=dict(facecolor="black", alpha=0.65, pad=2),
            )
        for text in report.get("texts", []):
            poly = np.array(text["polygon"])
            ax.plot(*np.vstack([poly, poly[:1]]).T, color="yellow", linewidth=1)
            ax.text(
                poly[0, 0],
                poly[0, 1],
                text["text"],
                fontsize=6,
                color="yellow",
                bbox=dict(facecolor="black", alpha=0.6, pad=1),
            )
        ax.set_title("Perception masks | accepted = tentative, red = rejected")
        ax.axis("off")

    def draw_bev(ax):
        if len(geometry):
            ax.scatter(
                -geometry[:, 1],
                geometry[:, 0],
                s=1,
                c="#b7bdc5",
                label="All depth obstacles",
            )
        for row, _, pts, color in rows:
            if row["result"]["status"] != "accepted" or not len(pts):
                continue
            ax.scatter(
                -pts[:, 1],
                pts[:, 0],
                s=3,
                color=color,
                label=f"#{row['instance_index']} {row['category']} {row.get('identity_text') or ''}",
            )
            center = pts.mean(0)
            ax.text(
                -center[1],
                center[0],
                f"#{row['instance_index']} {row.get('identity_text') or ''}",
                fontsize=9,
            )
        ax.scatter([0], [0], marker="^", color="black", s=60, label="Source robot")
        ax.set(
            xlabel="Right (-body Y), metres",
            ylabel="Forward (body X), metres",
            title="Single-frame semantic BEV | tentative evidence",
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc="upper left")

    fig, ax = plt.subplots(figsize=(12, 9))
    draw_masks(ax)
    fig.tight_layout()
    fig.savefig(root / "segmentation.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 9))
    draw_bev(ax)
    fig.tight_layout()
    fig.savefig(root / "semantic_bev.png", dpi=160)
    plt.close(fig)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    if len(geometry):
        ax.scatter(*geometry.T, s=1, c="#88929e", alpha=0.45)
    for row, _, pts, color in rows:
        if row["result"]["status"] == "accepted" and len(pts):
            ax.scatter(
                *pts.T,
                s=3,
                color=color,
                label=f"#{row['instance_index']} {row['category']} {row.get('identity_text') or ''}",
            )
    ax.set(
        xlabel="Forward X (m)",
        ylabel="Left Y (m)",
        zlabel="Up Z (m)",
        title="Observed surfaces in source robot frame | NOT verified static anchors",
    )
    ax.view_init(elev=24, azim=145)
    if rows:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "semantic_3d.png", dpi=160)
    plt.close(fig)
    render_overview(
        obs["rgb"], image_id, report, llm_result, root, weight_mode=weight_mode
    )


def render_overview(
    source_rgb, image_id, report, llm_result, root, *, weight_mode="unspecified"
):
    """Compose existing measured panels with explicit selector/model provenance."""
    root = Path(root)
    manual = bool(report["categories"]) and all(
        row["request"]["source"] == "manual" for row in report["categories"]
    )
    backend = report.get("provenance", {}).get("backend", "YOLO-E")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        "MANUAL CATEGORY DIAGNOSTIC - excluded from end-to-end scores"
        if manual
        else (
            "InternVL3-1B BASE control (TIC-VLA weights NOT loaded)"
            if weight_mode in ("internvl_base_control", "shared_internvl_base_answer")
            else (
                "TIC-VLA initialized InternVL3-1B"
                if weight_mode == "ticvla_initialized"
                else "Weight mode: " + weight_mode
            )
        ),
        fontsize=15,
    )
    axes[0, 0].imshow(Image.open(source_rgb).convert("RGB"))
    axes[0, 0].set_title(f"Source RGB: {image_id}")
    axes[0, 0].axis("off")
    for axis, filename in (
        (axes[0, 1], "segmentation.png"),
        (axes[1, 0], "semantic_bev.png"),
    ):
        axis.imshow(Image.open(root / filename).convert("RGB"))
        axis.axis("off")
    text_ax = axes[1, 1]
    text_ax.axis("off")
    lines = [
        ("MANUAL category prompts" if manual else "REAL InternVL answer")
        + " -> "
        + backend,
        "",
    ]
    parsed = llm_result.get("parsed") or {}
    items = parsed.get(
        "landmark", [{"category": c} for c in parsed.get("categories", [])]
    )
    if manual:
        items = [row["request"] for row in report["categories"]]
    for item in items:
        if manual or llm_result.get("inference_mode") == "OBJECTS_ONLY":
            lines.append(item["category"])
        else:
            lines += [
                f"{item['category']} [{item.get('navigation_role', 'unspecified')}]",
                textwrap.fill(item.get("reason", ""), 65),
                "",
            ]
    if manual:
        lines += ["", "VLM selection was NOT used in this diagnostic.", ""]
    elif llm_result.get("inference_mode") == "OBJECTS_ONLY":
        lines += ["", "One visual QA; structured inference skipped.", ""]
    for cat in report["categories"]:
        lines.append(
            f"{cat['request']['category']}: {cat['instances']} masks, {cat['accepted']} accepted"
        )
    lines += [
        "",
        "Single-frame observations remain TENTATIVE.",
        "Geometry also includes obstacles outside the selected classes.",
        "Ground-truth metrics, when available, are reported separately.",
        "Navigation or model accuracy is not certified by this example.",
    ]
    if not manual and llm_result.get("error"):
        lines += ["LLM ERROR: " + llm_result["error"]]
    text_ax.text(0, 1, "\n".join(lines), va="top", fontsize=11, linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(root / "overview.png", dpi=160)
    plt.close(fig)
