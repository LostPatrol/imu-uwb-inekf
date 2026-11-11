"""Replay two preset cold-rotation conditions against current shared solver source."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cases import make_case
from run_checks import evaluate

# Cover large unknown tilt and weak geometry/excitation without repeating the panel.
PANEL = ((2026111100, "rich", 179., "normal"), (2026111101, "weak", 60., "weak_height"))


def main():
    """Compare deterministic saved metrics while excluding runtime and pooled ranking."""
    source = Path(__file__).resolve().parent / "results"
    saved = pd.read_csv(source / "metrics.csv")
    rerun = []
    for seed, motion, tilt, geometry in PANEL:
        case = make_case(seed, motion, tilt, geometry)
        rerun.extend(evaluate(case))
        rerun.extend(evaluate(case, include_multi=False, only_local_gravity=True))
    replay = pd.DataFrame(rerun).sort_values(["seed", "method"]).reset_index(drop=True)
    masks = [(saved.seed == seed) & (saved.motion == motion) & (saved.tilt_deg == tilt)
             & (saved.geometry == geometry) for seed, motion, tilt, geometry in PANEL]
    reference = saved[np.logical_or.reduce(masks)].sort_values(["seed", "method"]).reset_index(drop=True)
    columns = [name for name in replay.select_dtypes(include=np.number)
               if name not in ("shared_seed_seconds", "refine_seconds")]
    np.testing.assert_allclose(replay[columns], reference[columns], atol=1e-8, rtol=1e-8)
    result = dict(panel=PANEL, replayed_method_estimates=len(replay),
        max_numeric_difference=float(np.max(np.abs(replay[columns].to_numpy()-reference[columns].to_numpy()))),
        total_method_estimates=len(saved), all_saved_numeric_finite=bool(np.isfinite(saved.select_dtypes(include=np.number)).all().all()))
    (source / "replay_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
