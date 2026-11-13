"""Score causal Pfeiffer publications only where sealed reference is available."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# Offline truth stays outside the estimator replay module.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SCORING = ROOT / "agent/codex/acquisition_challenge_20261113/scoring"
GRID = np.arange(0., 15.001, .5)


def score_file(path):
    """Hold the last published estimate at each deadline without backdating."""
    estimates = pd.read_csv(path)
    if not {"arrival_s", "state_s", "x", "y", "z"}.issubset(estimates.columns):
        return None
    episode = estimates.episode.iloc[0]
    truth = pd.read_csv(SCORING / f"{episode}.csv")
    valid_grid = GRID[(GRID >= truth.t.min()) & (GRID <= truth.t.max()) &
                      (GRID <= estimates.arrival_s.max()+.025)]
    indices = np.searchsorted(estimates.arrival_s, valid_grid, side="right") - 1
    present = indices >= 0
    valid_grid, indices = valid_grid[present], indices[present]
    predicted = estimates.iloc[indices][["x", "y", "z"]].to_numpy()
    reference = np.column_stack([np.interp(valid_grid, truth.t, truth[x]) for x in ("x", "y", "z")])
    error = predicted - reference
    xy = np.linalg.norm(error[:, :2], axis=1)
    result = dict(episode=episode, method=estimates.method.iloc[0],
        case_id=estimates.case_id.iloc[0], reference_start_s=float(truth.t.min()),
        reference_end_s=float(truth.t.max()), scored_samples=len(valid_grid),
        core_failed=bool((estimates.status != "ok").any()))
    for threshold in (.25, .5, 1.):
        fixed = np.flatnonzero(np.convolve((xy < threshold).astype(int), np.ones(3, int), "valid") == 3) if len(xy)>=3 else []
        result[f"first_confirmation_{threshold}m_s"] = float(valid_grid[fixed[0]+2]) if len(fixed) else np.nan
    for deadline in (2., 10., 15.):
        if ((estimates.arrival_s <= deadline) & (estimates.status != "ok")).any():
            continue
        usable = (estimates.arrival_s <= deadline) & (estimates.status == "ok")
        if not usable.any() or not truth.t.min() <= deadline <= truth.t.max():
            continue
        estimate = estimates.loc[usable].iloc[-1]
        target = np.array([np.interp(deadline, truth.t, truth[x]) for x in ("x", "y", "z")])
        difference = estimate[["x", "y", "z"]].to_numpy(float)-target
        result[f"xy_{deadline:g}s_m"] = float(np.linalg.norm(difference[:2]))
        result[f"xyz_{deadline:g}s_m"] = float(np.linalg.norm(difference))
    return result


def main():
    """Write flight/case metrics for one completed replay directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = [score_file(p) for p in sorted(args.directory.glob("delft_*.csv"))]
    rows = [r for r in rows if r is not None]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.directory / "metrics.csv", index=False)
    print(metrics.groupby("method").agg(n=("episode", "count"),
        fixed=("first_confirmation_0.5m_s", "count"),
        median_confirmation=("first_confirmation_0.5m_s", "median"),
        median_xy10=("xy_10s_m", "median")).to_json())


if __name__ == "__main__":
    main()
