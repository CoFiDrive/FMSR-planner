"""Repeat clean FM+SR evaluation on route 17563 until DS reaches 100."""
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visual_case_campaign import ROOT, prepare, run, write


def main():
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    os.environ.update(
        VISUAL_SAVE_INJECTION_ARRAYS="1",
        VISUAL_PROBE_EVERY="100",
        VISUAL_PROBE_MAX="40",
    )
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "fmsr_pdm_lite"), str(ROOT), os.environ.get("PYTHONPATH", "")]
    )

    try:
        attempt = 1
        while True:
            current = output / f"attempt{attempt:03d}" / "clean"
            prepare(current, route_ids=["17563"], model_names=["FM_CR"])
            clean_spec = {
                "seed": 0,
                "route_conditions": {
                    "17563": {
                        "id": "clean",
                        "family": "none",
                        "amplitude": 0.0,
                        "mode": "continuous",
                    }
                },
            }
            write(current / "perturbation_job.json", clean_spec)
            source_path = current / "FM_CR" / "source.json"
            source = json.loads(source_path.read_text())
            source["analysis_recording"].update(probe_every=100, probe_max=40)
            source.update(
                visual_sensor_every=5,
                save_actual_injection_arrays=True,
                route_perturbations=clean_spec["route_conditions"],
            )
            write(source_path, source)

            write(
                output / "status.json",
                {"state": "evaluating", "attempt": attempt, "target_DS": 100, "condition": "clean"},
            )
            os.environ["ROBUSTNESS_SPEC"] = str(current / "perturbation_job.json")
            run(current)

            checkpoint = json.loads((current / "FM_CR" / "checkpoint_endpoint.json").read_text())
            records = checkpoint["_checkpoint"]["records"]
            record = next(r for r in records if r.get("route_id") == "RouteScenario_17563_rep0")
            audit = current / "FM_CR" / "eval_metrics" / record["save_name"] / "perturbations.jsonl"
            if not audit.is_file() or audit.read_text().strip():
                raise RuntimeError(f"Clean-run perturbation audit failed: {audit}")

            result = {
                "attempt": attempt,
                "route_id": record["route_id"],
                "model": "FM+SR",
                "condition": "clean",
                "DS": record["scores"]["score_composed"],
                "RC": record["scores"]["score_route"],
                "status": record["status"],
                "infractions": record["infractions"],
                "source": str(current / "FM_CR" / "checkpoint_endpoint.json"),
            }
            results.append(result)
            with (output / "all_attempts.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)
            write(output / "latest_result.json", result)

            if float(result["DS"]) >= 100.0:
                write(
                    output / "status.json",
                    {"state": "complete", "reason": "target_DS_reached", "attempts": attempt, "DS": result["DS"]},
                )
                return
            attempt += 1
    except Exception as error:
        write(output / "status.json", {"state": "error", "error": repr(error), "completed_attempts": results})
        raise


if __name__ == "__main__":
    main()
