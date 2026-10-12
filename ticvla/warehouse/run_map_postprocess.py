"""Read-only map audit and reversible focused landmark view (no model calls)."""
import argparse
import copy
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from .mapping.postprocess import LandmarkViewCache


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_view(bev, records, path, ids, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_axes([.06, .12, .48, .8])
    legend = fig.add_axes([.57, .04, .42, .9]); legend.axis("off")
    points = np.asarray(list(bev.cells)).reshape(-1, 2)*.1
    trail = np.asarray(bev.trail).reshape(-1, 2)
    if len(points): ax.scatter(points[:, 0], points[:, 1], s=.6, c="#7b8793")
    if len(trail): ax.plot(trail[:, 0], trail[:, 1], c="#2186db", linewidth=1)
    visible = [r for r in records if r.get("landmark_view", {}).get("visible", True)]
    for line, r in enumerate(visible):
        index = ids[r["landmark_uid"]]
        color = "#bd6500" if r.get("landmark_view", {}).get("decision") != "review" else "#675b98"
        x, y = r["position"][:2]
        ax.scatter([x], [y], s=15, c=color)
        ax.annotate(str(index), (x,y), xytext=(3,3), textcoords="offset points", fontsize=7, color=color)
        text = r.get("identity_text") or "?"
        state = r.get("identity_status", "unknown")
        legend.text(0, .99-line*.029, f"{index:02d}  {r['category']}  {text}  [{state}]", fontsize=8.5, color=color, va="top")
    ax.set(xlabel="Odom x (m)", ylabel="Odom y (m)", title=title)
    ax.set_aspect("equal"); ax.grid(alpha=.2)
    fig.text(.06, .03, "Geometry unchanged; blue: vehicle trajectory. All objects remain tentative. Purple: requires review.", fontsize=9)
    fig.savefig(path, dpi=150); plt.close(fig)


def run(map_root, output):
    from PIL import Image, ImageDraw
    from DynaNav.warehouse_loop.live_mapping import LiveBEV
    map_root, output = Path(map_root).resolve(), Path(output).resolve()
    if output == map_root or map_root in output.parents:
        raise ValueError("Output must be outside the source map")
    output.mkdir(parents=True, exist_ok=False)
    database = map_root / "map.sqlite"
    before_hash = digest(database)
    with sqlite3.connect(database.as_uri()+"?mode=ro", uri=True) as db:
        def records(table):
            return [json.loads(s) for (s,) in db.execute("SELECT payload FROM " + table + " ORDER BY id")]
        raw = records("landmarks")
        observations = {o["image_id"]: o for o in records("observations")}
        events = records("events")
        text_events = records("text_events")
    cache = LandmarkViewCache()
    started = time.perf_counter()
    view = cache.evaluate(raw, observations.get)
    elapsed = time.perf_counter()-started
    # A fresh cache simulates reconstruction after process restart.
    restarted = LandmarkViewCache().evaluate(raw, observations.get)
    # Replay saved evidence into the derived view, including geometry placeholders
    # followed by delayed text. Geometry/association are already established here;
    # this is not a replay of model inference or vehicle scheduling.
    incremental = copy.deepcopy(raw)
    evidence_stream = []
    for r in incremental:
        evidence_stream.extend((r, e) for e in r.get("text_evidence", []))
        r["text_evidence"] = []
    replay_cache = LandmarkViewCache(); replay_times = []
    for record, evidence in evidence_stream:
        record["text_evidence"].append(evidence)
        tick = time.perf_counter()
        replayed = replay_cache.evaluate(incremental, observations.get)
        replay_times.append(time.perf_counter()-tick)
    replayed = replay_cache.evaluate(incremental, observations.get)
    changes = [dict(landmark_uid=v["landmark_uid"], category=v["category"],
                    before_text=v["raw_identity_text"], before_status=v["raw_identity_status"],
                    after_text=v["identity_text"], after_status=v["identity_status"],
                    decision=v["landmark_view"], evidence=v["resolved_identity"])
               for v in view["landmarks"]]
    for name, data in [("landmark-view.json", view), ("changes.json", changes)]:
        (output/name).write_text(json.dumps(data, ensure_ascii=False, indent=2))
    images = []
    # Never overlay independent, unregistered odom epochs.
    for index, submap in enumerate(sorted({r["submap_id"] for r in raw})):
        bev = LiveBEV()
        for obs in sorted(observations.values(), key=lambda o: o["t_obs"]):
            if obs["submap_id"] != submap:
                continue
            with np.load(map_root/obs["geometry"], allow_pickle=False) as a:
                bev.add(a["points"], np.array(obs["T_odom_body"]))
        old = [r for r in raw if r["submap_id"] == submap and r["status"] != "revoked"]
        new = [r for r in view["landmarks"] if r["submap_id"] == submap]
        prefix = "" if index == 0 else f"submap-{index}-"
        ids = {r["landmark_uid"]: i+1 for i,r in enumerate(old)}
        plot_view(bev, old, output/(prefix+"before.png"), ids, "Raw observations")
        plot_view(bev, new, output/(prefix+"after.png"), ids, "Focused landmark view")
        before = Image.open(output/(prefix+"before.png"))
        after = Image.open(output/(prefix+"after.png"))
        canvas = Image.new("RGB", (before.width*2, before.height+35), "white")
        canvas.paste(before, (0, 35)); canvas.paste(after, (before.width, 35))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), "RAW OBSERVATIONS", fill="black")
        draw.text((before.width+10, 10), "FOCUSED VIEW (keep + review; geometry unchanged)", fill="black")
        name = prefix+"comparison.png"; canvas.save(output/name); images.append(name)
    report = dict(source_map=str(map_root), source_database_sha256=before_hash,
                  source_database_unchanged=before_hash == digest(database),
                  restart_identical=view == restarted, model_calls=0,
                  incremental_evidence_replay_identical=view == replayed,
                  incremental_updates=len(replay_times),
                  incremental_processing_p50_ms=float(np.percentile(replay_times,50)*1000) if replay_times else None,
                  incremental_processing_p95_ms=float(np.percentile(replay_times,95)*1000) if replay_times else None,
                  processing_seconds=elapsed, counts=view["counts"],
                  geometry_observations=len(observations), events=len(events), text_events=len(text_events),
                  recovered_conflicts=sum(c["before_status"] == "conflict" and c["after_text"] is not None for c in changes),
                  unresolved_conflicts=sum(c["after_status"] == "conflict" for c in changes),
                  static_flags_unchanged=all(v["static_confirmed"] == r["static_confirmed"] for v,r in zip(view["landmarks"], [r for r in raw if r["status"] != "revoked"])),
                  hidden_categories=dict(Counter(v["category"] for v in view["landmarks"] if not v["landmark_view"]["visible"])),
                  scope="Derived view validation; no new perception, accuracy or live-rate claim.")
    (output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    page = ['<!doctype html><meta charset="utf-8"><title>Landmark postprocessing audit</title>',
            '<style>body{font:16px sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:7px}img{max-width:100%}pre{white-space:pre-wrap}</style>',
            '<h1>Landmark postprocessing audit</h1><p>Raw evidence is unchanged. Hidden objects remain in the database and geometric map. Review objects are not verified identities.</p>',
            '<p><a href="report.json">Report</a> · <a href="landmark-view.json">Derived view</a> · <a href="changes.json">Evidence decisions</a></p>',
            '<pre>'+html.escape(json.dumps(report, ensure_ascii=False, indent=2))+'</pre>']
    page.extend('<img src="'+name+'">' for name in images)
    page.append('<table><tr><th>ID/category</th><th>Before</th><th>After</th><th>Decision</th><th>Evidence/reason</th></tr>')
    for c in changes:
        cells = [c["landmark_uid"][:8]+" / "+c["category"], str(c["before_text"])+" / "+c["before_status"],
                 str(c["after_text"])+" / "+c["after_status"],c["decision"]["decision"]+": "+c["decision"]["reason"],
                 json.dumps(c["evidence"], ensure_ascii=False)]
        page.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in cells)+'</tr>')
    (output/"index.html").write_text(''.join(page)+'</table>')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    print(json.dumps(run(args.map, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
