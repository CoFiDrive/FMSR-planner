"""Run one clean/BEV-noise pair for FMSR, FM, and FM+SR on route 23658."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from visual_case_campaign import ROOT, prepare, run, write
from fmsrpdm.evaluation_watchdog import inspect_checkpoint


ROUTE_ID = "23658"
MODELS = ["FMSR", "FM", "FM_CR"]
NOISE = {
    "id": "feature_bev_0.03",
    "family": "feature",
    "component": "bev",
    "amplitude": 0.03,
    "mode": "continuous",
    "start_s": 2.0,
}
CLEAN = {"id": "clean", "family": "none", "amplitude": 0.0, "mode": "continuous"}


def prepare_campaign(output):
    if output.exists():
        raise ValueError(f"Use a new output directory: {output}")
    output.mkdir(parents=True)
    for condition, spec in [("clean", CLEAN), ("perturbed", NOISE)]:
        base = output / condition
        prepare(base, route_ids=[ROUTE_ID], model_names=MODELS)
        write(base / "perturbation_job.json", {"seed": 0, "route_conditions": {ROUTE_ID: spec}})
        for model in MODELS:
            path = base / model / "source.json"
            source = json.loads(path.read_text())
            source["analysis_recording"].update(probe_every=100, probe_max=40)
            source.update(
                visual_sensor_every=5,
                save_actual_injection_arrays=True,
                route_perturbations={ROUTE_ID: spec},
            )
            write(path, source)
    write(output / "manifest.json", {
        "route_ids": [ROUTE_ID],
        "scenario": "Town12 / HighwayExit",
        "models": MODELS,
        "display_names": {"FMSR": "FMSR", "FM": "FM", "FM_CR": "FM+SR"},
        "conditions": ["clean", "perturbed"],
        "seed": 0,
        "total_route_runs": 6,
        "route_conditions": {ROUTE_ID: NOISE},
        "protocol": "Fresh clean and perturbed runs for each model. BEV Gaussian feature noise has RMS ratio 0.03 relative to the BEV tensor and is applied continuously from simulation time 2 s. Original route XML, weather, checkpoint, sampler and controller are retained. All six outcomes are reported.",
        "recording": "Camera every 5 ticks; diagnostic ego, actor, event and policy traces; planning probes every 100 steps, maximum 40; sampled actual before/after/delta injection arrays.",
    })


def summarize(output):
    rows = []
    for condition in ["clean", "perturbed"]:
        for model in MODELS:
            root = output / condition / model
            checkpoint = root / "checkpoint_endpoint.json"
            state, payload = inspect_checkpoint(checkpoint, ["RouteScenario_23658_rep0"])
            if not state["complete"]:
                raise ValueError(f"Incomplete evaluation: {root}")
            record = next(
                row for row in json.loads(payload)["_checkpoint"]["records"]
                if row["route_id"] == "RouteScenario_23658_rep0"
            )
            diag = root / "diagnostics" / record["save_name"]
            metric = root / "eval_metrics" / record["save_name"]
            for path in [diag / "frames.jsonl", diag / "visual/frames.jsonl", metric / "perturbations.jsonl"]:
                if not path.is_file() or (path.name != "perturbations.jsonl" and not path.stat().st_size):
                    raise ValueError(f"Missing recording: {path}")
            events = [
                event
                for line in (metric / "perturbations.jsonl").read_text().splitlines()
                for event in json.loads(line).get("events", [])
            ]
            arrays = sorted((metric / "injection_arrays").glob("*.npz"))
            if condition == "clean":
                if events or arrays:
                    raise ValueError(f"Clean run contains injections: {root}")
            elif not events or not arrays:
                raise ValueError(f"Perturbation was not recorded: {root}")
            if any(event.get("component") != "bev" for event in events):
                raise ValueError(f"Non-BEV injection recorded: {root}")
            ratios = []
            for path in arrays:
                with np.load(path) as data:
                    before = data["before"]
                    after = data["after"]
                    delta = data["delta"]
                    if before.shape != after.shape or before.shape != delta.shape or not np.allclose(after - before, delta):
                        raise ValueError(f"Injection arrays differ from recorded delta: {path}")
                    baseline = float(np.sqrt(np.mean(before.astype(np.float64) ** 2)))
                    change = float(np.sqrt(np.mean(delta.astype(np.float64) ** 2)))
                    if baseline > 0:
                        ratios.append(change / baseline)
            rows.append({
                "route_id": ROUTE_ID,
                "model": model,
                "display_model": "FM+SR" if model == "FM_CR" else model,
                "condition": condition,
                "DS": record["scores"]["score_composed"],
                "RC": record["scores"]["score_route"],
                "status": record["status"],
                "vehicle_collisions": len(record["infractions"].get("collisions_vehicle", [])),
                "injection_events": len(events),
                "injection_array_samples": len(arrays),
                "sampled_delta_to_BEV_RMS": sum(ratios) / len(ratios) if ratios else None,
                "source": str(checkpoint),
            })
    with (output / "all_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(output / "recording_audit.json", {"complete": True, "route_runs": len(rows), "results": rows})


def main():
    output = Path(sys.argv[1]).resolve()
    if "--foreground" not in sys.argv:
        prepare_campaign(output)
        with (output / "monitor.log").open("a") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", __file__, str(output), "--foreground"],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (output / "monitor.pid").write_text(str(process.pid) + "\n")
        print("Monitor PID:", process.pid, "Output:", output)
        return

    os.environ.update(VISUAL_SAVE_INJECTION_ARRAYS="1", VISUAL_PROBE_EVERY="100", VISUAL_PROBE_MAX="40")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "fmsr_pdm_lite"), str(ROOT), os.environ.get("PYTHONPATH", "")]
    )
    try:
        for condition in ["clean", "perturbed"]:
            write(output / "status.json", {"state": "evaluating", "condition": condition, "total_route_runs": 6})
            os.environ["ROBUSTNESS_SPEC"] = str(output / condition / "perturbation_job.json")
            run(output / condition)
        write(output / "status.json", {"state": "auditing", "total_route_runs": 6})
        summarize(output)
        write(output / "status.json", {"state": "complete", "total_route_runs": 6})
    except Exception as error:
        write(output / "status.json", {"state": "error", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
