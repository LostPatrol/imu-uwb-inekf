"""Summarize the separately frozen author plus early-XY overlay after replay."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# All output remains inside this independent attribution experiment.
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results"
WINDOWS = ("early", "future", "post")
TITLES = ("20–22 s", "26–31 s", "26–45 s")


def main():
    """Retain all cases, including nontriggers and numerical failures."""
    records = json.loads((OUTPUT / "records.json").read_text(encoding="utf8"))
    if len(records) != 16:
        raise RuntimeError("All sixteen frozen conditions must finish before analysis")
    frame = pd.read_csv(OUTPUT / "paired.csv")
    summaries = []
    for cohort in ("primary", "extension"):
        for method in ("native", "early_reserved", "early_switch"):
            selected = frame[(frame.cohort == cohort) & (frame.method == method)]
            item = dict(cohort=cohort, method=method, cases=len(selected), triggers=int(selected.triggered.sum()),
                        failures=int(selected.failed.sum()))
            for window in WINDOWS:
                for metric in ("xy_rmse", "yaw_rmse_deg"):
                    item[window+"_"+metric+"_median"] = float(selected[window+"_"+metric].median())
                    delta = selected[window+"_"+metric+"_minus_native"]
                    item[window+"_"+metric+"_paired_improvements"] = int((delta < -1e-10).sum())
                item[window+"_minimum_coverage"] = float(selected[window+"_coverage"].min())
            summaries.append(item)
    (OUTPUT / "overview.json").write_text(json.dumps(summaries, indent=2), encoding="utf8")
    checks = []
    reasons = []
    for record in records:
        trigger = record["trigger_s"]
        first = record["first_post_packet_s"]
        checks.append(dict(name=record["name"], yaw_offset_deg=record["yaw_offset_deg"], cohort=record["cohort"],
            triggered=trigger is not None, trigger_s=trigger,
            strict_subsequent_packets=first is None or (trigger is not None and first > trigger),
            preservation=record["preservation"], failures=record["failures"], runtime_seconds=record["runtime_seconds"]))
        for attempt in record["attempts"]:
            reasons.append(dict(name=record["name"], yaw_offset_deg=record["yaw_offset_deg"], cohort=record["cohort"], **attempt))
    (OUTPUT / "integrity.json").write_text(json.dumps(checks, indent=2), encoding="utf8")
    pd.DataFrame(reasons).to_csv(OUTPUT / "availability_attempts.csv", index=False)
    # A fresh Python interpreter avoids module-name collisions with author adapters.
    reruns = []
    python = HERE.parents[4] / ".venv" / "Scripts" / "python.exe"
    for record in sorted(records, key=lambda item: (item["cohort"], item["name"], item["yaw_offset_deg"])):
        program = "import runpy; runpy.run_path(" + json.dumps(str(HERE/"run.py")) + ", run_name=\"overlay_api\")[\"run_case\"]((" + json.dumps(record["name"]) + "," + str(record["yaw_offset_deg"]) + "))"
        reruns.append(dict(name=record["name"], yaw_offset_deg=record["yaw_offset_deg"], cohort=record["cohort"],
            powershell="& '"+str(python)+"' -c '"+program+"'"))
    pd.DataFrame(reruns).to_csv(OUTPUT/"recording_rerun_commands.csv", index=False)
    figure, axes = plt.subplots(2, 3, figsize=(13, 6))
    colors = {"primary": "#23775d", "extension": "#aa6835"}
    for column, (window, title) in enumerate(zip(WINDOWS, TITLES)):
        for row, (metric, unit) in enumerate((("xy_rmse", "XY RMSE difference (m)"), ("yaw_rmse_deg", "Yaw RMSE difference (deg)"))):
            axis = axes[row, column]
            all_values = []
            for offset, cohort in ((-.12, "primary"), (.12, "extension")):
                selected = frame[(frame.cohort == cohort) & (frame.method == "early_switch")].sort_values(["name", "yaw_offset_deg"])
                values = selected[window+"_"+metric+"_minus_native"].to_numpy()
                all_values.extend(values)
                axis.scatter(np.arange(len(values))+offset, values, color=colors[cohort], label=cohort, s=24)
            axis.axhline(0, color="black", linewidth=.8)
            axis.set_yscale("symlog", linthresh=.05 if row == 0 else 1.)
            floor = .1 if row == 0 else 1.
            lower = -10.**np.ceil(np.log10(max(floor, -min(all_values))))
            upper = 10.**np.ceil(np.log10(max(floor, max(all_values))))
            axis.set_ylim(lower*1.1, upper*1.1)
            ticks = (-100.,-10.,-1.,-.1,0.,.1,1.,10.,100.) if row == 0 else (-100.,-10.,-1.,0.,1.,10.,100.)
            ticks = [tick for tick in ticks if lower <= tick <= upper]
            axis.set_yticks(ticks, [f"{tick:g}" for tick in ticks])
            axis.grid(alpha=.2)
            axis.set_title(title)
            axis.set_xlabel("Frozen recording/yaw pair")
            axis.set_ylabel(unit+"; symlog")
    axes[0, 0].legend()
    figure.suptitle("Author + frozen early XY output switch versus untouched author\nNegative values favor the overlay; primary and extension remain separate")
    figure.tight_layout()
    figure.savefig(OUTPUT / "paired_overlay.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
