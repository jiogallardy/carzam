"""Compare multiple trained contrastive runs side-by-side.

For each run dir (must have classes.json, checkpoint.pt, prototypes.pt):
  1. Test-split classification accuracy via the classifier head.
  2. Test-split nearest-prototype accuracy via the embedding head.
  3. Per-class F1 from both heads.
  4. Open-set test: embed an out-of-distribution clip (LFA by default — not in
     the trained CARS set), measure max prototype sim. The gap between this
     and the mean known-clip sim is the "open-set headroom".

Renders an HTML report + serves it locally.

Usage:
  .venv/bin/python scripts/compare_runs.py \
      --runs runs/20260514_231244,runs/20260515_xxxxxx,runs/...
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import json
import random
import socketserver
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from sklearn.metrics import classification_report

from carzam.audio import load_wav
from carzam.data.dataset import CarAudioDataset
from carzam.data.manifest import read_manifest
from carzam.data.splits import split_by_video
from carzam.embedding import (
    embed_audio,
    load_embedding_model,
    load_prototypes,
    match,
)

console = Console()


@torch.no_grad()
def eval_run(run_dir: Path, device: torch.device, oos_clip: Path | None) -> dict:
    """Run all evaluations on a single contrastive run."""
    classes_path = run_dir / "classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(classes_path)
    classes = json.loads(classes_path.read_text())
    cars = tuple(classes["cars"])

    model = load_embedding_model(run_dir, device)
    prototypes = load_prototypes(run_dir)

    # Split the manifest the same way training did (seed=42).
    rows = read_manifest("data/manifest.csv")
    splits = split_by_video(rows, ratios=(0.7, 0.15, 0.15), seed=42)
    test_rows = [r for r in splits.test if r.label is not None and r.car in cars]
    console.print(f"  evaluating {len(test_rows)} test-split windows")

    # Loop the test set, get both classifier prediction and nearest-prototype prediction.
    cls_pred: list[int] = []
    emb_pred: list[int] = []
    truth: list[int] = []
    emb_max_sim: list[float] = []

    from carzam.data.dataset import compute_logmel
    from carzam.audio import resample_to
    SR = 32000

    car_to_idx = {c: i for i, c in enumerate(cars)}
    proto_emb = prototypes.embeddings.to(device)

    for r in test_rows:
        audio, sr = load_wav(r.path)
        if sr != SR:
            audio = resample_to(audio, src_sr=sr, dst_sr=SR)
        logmel = compute_logmel(torch.from_numpy(audio)).unsqueeze(0).to(device)
        # classifier
        car_logits, _, _, z = model.forward_with_embedding(logmel)
        cls_top = int(car_logits.argmax(-1).item())
        # embedding (only valid prototypes)
        valid = prototypes.n_samples > 0
        sims = (proto_emb @ z.squeeze(0))
        sims_masked = sims.clone()
        sims_masked[~valid.to(device)] = -1.0
        emb_top = int(sims_masked.argmax().item())
        cls_pred.append(cls_top)
        emb_pred.append(emb_top)
        truth.append(car_to_idx[r.car])
        emb_max_sim.append(float(sims_masked.max().item()))

    cls_acc = float(np.mean(np.array(cls_pred) == np.array(truth)))
    emb_acc = float(np.mean(np.array(emb_pred) == np.array(truth)))

    # Per-class F1
    target_names = list(cars)
    cls_report = classification_report(
        truth, cls_pred, labels=list(range(len(cars))),
        target_names=target_names, output_dict=True, zero_division=0,
    )
    emb_report = classification_report(
        truth, emb_pred, labels=list(range(len(cars))),
        target_names=target_names, output_dict=True, zero_division=0,
    )

    # Mean known-clip sim (for open-set headroom)
    mean_known_sim = float(np.mean(emb_max_sim))

    # Open-set: LFA out-of-distribution
    oos = {"checked": False}
    if oos_clip and oos_clip.exists():
        audio, sr = load_wav(oos_clip)
        z = embed_audio(audio, sr, model, device)
        result = match(z, prototypes, top_k=5, unknown_threshold=0.55)
        oos = {
            "checked": True,
            "clip": str(oos_clip),
            "nearest": result.nearest_car,
            "nearest_sim": result.nearest_sim,
            "headroom_below_known_mean": mean_known_sim - result.nearest_sim,
            "top_k": result.top_k,
        }

    return {
        "run_dir": str(run_dir),
        "n_classes": len(cars),
        "cls_acc": cls_acc,
        "emb_acc": emb_acc,
        "macro_f1_cls": cls_report["macro avg"]["f1-score"],
        "macro_f1_emb": emb_report["macro avg"]["f1-score"],
        "mean_known_sim": mean_known_sim,
        "oos": oos,
        "cls_report": cls_report,
        "emb_report": emb_report,
    }


REPORT_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>contrastive runs comparison</title>
<style>
  body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,sans-serif;
          background:#0e0e10; color:#f4f4f5; }}
  .wrap {{ max-width:1280px; margin:0 auto; padding:24px; }}
  h1 {{ font-size:26px; margin:0 0 4px; }}
  h2 {{ font-size:18px; margin:28px 0 10px; color:#c9c9d1;
        border-bottom:1px solid #2c2c30; padding-bottom:6px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; margin:10px 0; }}
  th, td {{ padding:6px 10px; text-align:left; border-bottom:1px solid #1f1f23; }}
  th {{ background:#15151a; color:#c9c9d1; }}
  .n {{ font-family:ui-monospace,"SF Mono",Menlo,monospace; }}
  .ok {{ color:#4ade80; }}
  .bad {{ color:#fb7185; }}
  .dim {{ color:#71717a; }}
</style></head>
<body><div class="wrap">

<h1>contrastive runs comparison</h1>
<div class="dim">{ts}</div>

<h2>overall metrics</h2>
<table><thead><tr>
<th>run</th><th>classes</th>
<th>cls acc</th><th>emb acc</th>
<th>cls F1</th><th>emb F1</th>
<th>mean known sim</th>
<th>LFA sim</th>
<th>headroom</th>
</tr></thead><tbody>
{rows}
</tbody></table>

<h2>open-set details (Lexus LFA — not in training set)</h2>
<table><thead><tr>
<th>run</th><th>nearest</th><th>sim</th><th>top-3</th>
</tr></thead><tbody>
{oos_rows}
</tbody></table>

</div></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="Comma-separated list of run dirs.")
    ap.add_argument("--oos-clip", type=Path,
                    default=Path("data/raw/lexus_lfa/F0m6dvpK4m8.wav"),
                    help="Out-of-distribution clip (Lexus LFA by default).")
    ap.add_argument("--port", type=int, default=8013)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    runs = [Path(r.strip()) for r in args.runs.split(",")]
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    summaries: list[dict] = []
    for r in runs:
        console.print(f"[bold]evaluating[/bold] {r}")
        try:
            s = eval_run(r, device, args.oos_clip)
            summaries.append(s)
        except Exception as e:
            console.print(f"  [red]failed[/red]: {e}")

    rows: list[str] = []
    oos_rows: list[str] = []
    for s in summaries:
        name = Path(s["run_dir"]).name
        oos = s["oos"]
        oos_sim = f"{oos['nearest_sim']:.3f}" if oos.get("checked") else "—"
        headroom = (
            f"{oos['headroom_below_known_mean']:+.3f}"
            if oos.get("checked") else "—"
        )
        rows.append(
            f"<tr><td class='n'>{name}</td>"
            f"<td class='n'>{s['n_classes']}</td>"
            f"<td class='n'>{s['cls_acc']:.3f}</td>"
            f"<td class='n'>{s['emb_acc']:.3f}</td>"
            f"<td class='n'>{s['macro_f1_cls']:.3f}</td>"
            f"<td class='n'>{s['macro_f1_emb']:.3f}</td>"
            f"<td class='n'>{s['mean_known_sim']:.3f}</td>"
            f"<td class='n'>{oos_sim}</td>"
            f"<td class='n'>{headroom}</td></tr>"
        )
        if oos.get("checked"):
            top3 = " · ".join(f"{n}: {sim:.3f}" for n, sim in oos["top_k"][:3])
            oos_rows.append(
                f"<tr><td class='n'>{name}</td>"
                f"<td class='n'>{oos['nearest']}</td>"
                f"<td class='n'>{oos['nearest_sim']:.3f}</td>"
                f"<td class='dim'>{top3}</td></tr>"
            )

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = REPORT_HTML.format(
        ts=ts,
        rows="\n".join(rows),
        oos_rows="\n".join(oos_rows) if oos_rows else "<tr><td colspan='4' class='dim'>no LFA clip available</td></tr>",
    )
    out_dir = Path("runs/compare") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.html").write_text(html)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    console.print(f"\n[green]wrote[/green] {out_dir / 'report.html'}")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(out_dir))
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        url = f"http://127.0.0.1:{args.port}/report.html"
        console.print(f"\n[bold cyan]serving on {url}[/bold cyan]  (ctrl-c to stop)")
        if not args.no_open:
            threading.Thread(target=lambda: (time.sleep(0.4), webbrowser.open(url)),
                             daemon=True).start()
        with contextlib.suppress(KeyboardInterrupt):
            httpd.serve_forever()


if __name__ == "__main__":
    main()
