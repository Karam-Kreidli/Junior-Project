"""
BioReef.ai — dashboard worker
=============================
Runs on the GPU VM. Polls Supabase for videos the dashboard uploaded, runs the
pipeline on each (the same process_video the demo uses) and writes the results
back. All traffic is outbound from the VM, so it needs no open port.

    dashboard upload -> datasets row -> trigger queues a `runs` row (pending)
    worker: claim_run -> download -> process_video -> upload annotated mp4 -> finish_run

Env:
    SUPABASE_URL           https://<ref>.supabase.co
    SUPABASE_SERVICE_KEY   secret key (sb_secret_...) or legacy service_role JWT
                           (Project Settings -> API Keys). Never ship it
                           to the frontend: it bypasses row-level security.

Usage (in tmux, conda env bioreef_deploy, from the repo root):
    python scripts/pipeline/worker.py
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.parse
import urllib.request
from dataclasses import asdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viz")))
from bioreef._9_pipeline.config import InferenceConfig
from bioreef._9_pipeline.models import load_models
from demo_video import process_video

URL = os.environ["SUPABASE_URL"].rstrip("/")
KEY = os.environ["SUPABASE_SERVICE_KEY"]
BUCKET = "datasets"
POLL_SECONDS = 5


def call(method, path, body=None, headers=None, raw=False):
    """One Supabase REST call. `body` is JSON unless bytes."""
    data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
    h = {"apikey": KEY, "Content-Type": "application/json"}
    if KEY.startswith("eyJ"):       # legacy JWT service_role key; new sb_secret_ keys go in apikey only
        h["Authorization"] = f"Bearer {KEY}"
    h.update(headers or {})
    req = urllib.request.Request(URL + path, data=data, method=method, headers=h)
    with urllib.request.urlopen(req, timeout=600) as r:
        if raw:
            return r
        text = r.read().decode()
        return json.loads(text) if text else None


def set_run(run_id, **fields):
    call("PATCH", f"/rest/v1/runs?id=eq.{run_id}", fields, {"Prefer": "return=minimal"})


def download(storage_path, dest):
    r = call("GET", f"/storage/v1/object/{BUCKET}/{urllib.parse.quote(storage_path)}", raw=True)
    with r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def upload(src, storage_path):
    with open(src, "rb") as f:
        call("POST", f"/storage/v1/object/{BUCKET}/{urllib.parse.quote(storage_path)}", f.read(),
             {"Content-Type": "video/mp4", "x-upsert": "true"})


def to_h264(src, dst):
    """OpenCV writes mp4v, which browsers won't play; re-encode to H.264."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found: conda install -c conda-forge ffmpeg")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", dst], check=True)


def git_version():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       cwd=os.path.dirname(__file__)).strip()
    except Exception:
        return "unknown"


def result_payload(summary, verdicts, tracks, annotated_path, cfg):
    spans = {t["track_id"]: (t["frames"][0][0], t["frames"][-1][0]) for t in tracks if t["frames"]}
    fish = [{**{k: v[k] for k in ("track_id", "level", "name", "confidence", "n_frames")},
             "first_frame": spans.get(v["track_id"], (None, None))[0],
             "last_frame": spans.get(v["track_id"], (None, None))[1]} for v in verdicts]
    counts = [{"species": s, "maxn": n, "maxn_frame": summary["maxn_frame"].get(s)}
              for s, n in summary["maxn"].items()]
    keep = ("conf_threshold", "classify_every", "stage1_ckpt", "stage1_logit_tau", "rfdetr_size",
            "track_thresh", "iou_threshold", "max_age", "min_hits", "fast_inference")
    return {"fish": fish, "counts": counts, "annotated_path": annotated_path,
            "model_version": git_version(),
            "config": {k: v for k, v in asdict(cfg).items() if k in keep},
            "frames": summary["frames"], "fish_tracked": summary["fish_tracked"],
            "richness": summary["richness"], "shannon": summary["shannon"],
            "simpson": summary["simpson"]}


def handle(job, models, cfg):
    run_id = job["run_id"]
    print(f"[{time.strftime('%H:%M:%S')}] run {run_id}: {job['storage_path']}", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        video = os.path.join(tmp, os.path.basename(job["storage_path"]))
        download(job["storage_path"], video)
        last = [0.0]

        def progress(frac):          # at most one DB write every 5 s
            if time.time() - last[0] > 5:
                last[0] = time.time()
                set_run(run_id, progress=round(frac, 3))

        summary, verdicts, tracks, rendered = process_video(
            video, os.path.join(tmp, "out"), models, cfg, progress=progress)
        set_run(run_id, progress=0.95)
        web = os.path.join(tmp, "annotated.mp4")
        to_h264(rendered, web)
        annotated_path = f"{job['user_id']}/{job['dataset_id']}_{run_id}.mp4"
        upload(web, annotated_path)
        call("POST", "/rest/v1/rpc/finish_run",
             {"p_run": run_id, "p_result": result_payload(summary, verdicts, tracks, annotated_path, cfg)})
    print(f"  done: {summary['fish_tracked']} fish, richness {summary['richness']}", flush=True)


def main():
    cfg = InferenceConfig(stage1_ckpt="models/d6a_stage1.pt", stage1_prior="models/d6a_prior.json",
                          csv_path="frame_metadata.csv")
    print("Loading models...", flush=True)
    models = load_models(cfg)
    name = socket.gethostname()
    print(f"Worker {name} polling {URL}", flush=True)
    while True:
        try:
            job = call("POST", "/rest/v1/rpc/claim_run", {"p_worker": name})
        except Exception as e:       # network blip / paused project: keep polling
            detail = e.read().decode(errors="replace")[:300] if hasattr(e, "read") else ""
            print(f"  poll failed: {e} {detail}", flush=True)
            time.sleep(30)
            continue
        if not job:
            time.sleep(POLL_SECONDS)
            continue
        try:
            handle(job, models, cfg)
        except Exception as e:
            traceback.print_exc()
            try:
                set_run(job["run_id"], status="failed", error=f"{type(e).__name__}: {e}"[:2000])
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    main()
