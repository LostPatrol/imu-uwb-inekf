"""Verify the numerical cells of the six printed tables against their trial-level evidence."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Locate the separate manuscript revision and the experiment outputs under version control.
ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "imu-uwb-inekf/experiments"
TEX = ROOT / "tex/recovery_window_revision.tex"
ACQ = EXP / "radio_inertial_models/evidence/paired_acquisition"


def rms(values):
    """Return root mean squared error without rounding individual observations."""
    return np.sqrt(np.mean(np.square(values)))


def read_table(key, first_data):
    """Extract numerical cells following the first named data row of a LaTeX table."""
    body = TEX.read_text(encoding="utf8").split(r"\label{tab:" + key + "}", 1)[1]
    segment = body[body.index(first_data):body.index(r"\end{tabular}")]
    rows = []
    for line in segment.splitlines():
        if "&" not in line:
            continue
        cells = line.split(r"\\", 1)[0].split("&")
        rows.append([cell.strip().replace(",", "") for cell in cells
                     if re.fullmatch(r"[\d.,]+", cell.strip())])
    return rows


def check_table(key, first, expected):
    """Require each cell to agree with the reference at its displayed precision."""
    rows = read_table(key, first)
    assert len(rows) == len(expected), (key, len(rows), len(expected))
    count = 0
    for i, (found, target) in enumerate(zip(rows, expected), 1):
        assert len(found) == len(target), (key, i, found, target)
        for cell, value in zip(found, target):
            digits = len(cell.split(".")[1]) if "." in cell else 0
            tol = .500001 * 10**(-digits) if "." in cell else 1.e-9
            if abs(float(cell) - value) > tol:
                raise ValueError(f"{key}, data row {i}: printed {cell}; reference {value:.10f}")
            count += 1
    return count


def endpoint_tables():
    """Build exact references for paired endpoint, motion-control and seed-policy tables."""
    d = pd.read_csv(ACQ / "delft_same_profile.csv")
    q = pd.read_csv(ACQ / "qdrone_window_controls.csv")
    end = []
    for layout in ("all8", "lower4", None):
        g = d[d.layout == layout] if layout else d
        end.append([rms(g[f"{axis}_error_m_{state}"]) for state in ("old", "new")
                    for axis in ("xy", "z", "3d")] + [len(g)])
    end.append([rms(q[q.method == name][axis]) for name in ("Profile1_affine", "profile_1")
                for axis in ("xy_error_m", "abs_z_error_m", "position_error_m")] + [30])
    ctrl = pd.read_csv(ACQ / "delft_frame0_controls.csv").set_index(["method", "layout", "axis"])
    model = [[ctrl.loc[(method, lay, axis), "rmse_m"] for lay in ("all8", "lower4") for axis in ("xy", "3d")]
             for method in ("K1_static", "K2_static", "BiasPrior_static", "K2_geometry", "BiasPrior_geometry")]
    seeds = pd.read_csv(ACQ / "delft_new_objective_seeds.csv")
    seed = []
    for method in ("local", "profile_1", "uniform_4"):
        grp = seeds[seeds.method == method]
        seed.append([grp.missed_best_found.sum(), grp.total_nfev.sum(),
                     rms(grp.xy_error_m), rms(grp.position_error_m)])
    return (check_table("end", "Delft, eight anchors", end)
            + check_table("model", "Affine / constant", model)
            + check_table("seed", "One local yaw", seed))


def external_tables():
    """Read the directly compared author methods and the physical Toronto yaw endpoint."""
    pr = pd.read_csv(EXP / "published_range_only/results/overall_summary.csv").set_index(["method", "layout"])
    ph = pd.read_csv(EXP / "published_window_comparison/evidence/profile1_author_overall.csv").set_index(
        ["calibration", "method", "layout"])
    published = [[pr.loc[(m, l), a] for l in ("all8", "lower4")
                  for a in ("xy_error_m_rmse", "abs_z_error_m_rmse", "position_error_m_rmse")]
                 for m in ("relax_recover_K2", "relax_recover_weighted_K2",
                           "relax_recover_weighted_K3_selected_here")]
    for cal in ("static", "raw"):
        for m in ("Pfeiffer_MHE", "Pfeiffer_EKF", "Profile1_BiasPrior"):
            published.append([ph.loc[(cal, m, l), a] for l in ("all8", "lower4")
                              for a in ("xy_rmse_m", "z_rmse_m", "3d_rmse_m")])
    source = EXP / "split_handoff/oursland/early_overlay/results/summary.csv"
    if not source.exists():
        source = Path(__file__).with_name("toronto_author_summary.csv")
    tr = pd.read_csv(source).set_index(["cohort", "method"])
    toronto = [[tr.loc[(c, m), a] for c in ("primary", "extension") for a in
                ("post_xy_rmse", "post_yaw_rmse_deg")] for m in ("native", "early_reserved")]
    return (check_table("published", "Pacholska et al., degree one, unweighted", published)
            + check_table("oursland", "Oursland, author code", toronto))


def closed_table():
    """Use matched-history, unique-interval scoring; never pool the flights or phases."""
    s = pd.read_csv(EXP / "unified_window_recovery/results/supported_summary.csv").set_index(
        ["domain", "gap_s", "method", "window"])
    rows = []
    for domain in ("delft", "qdrone"):
        for gap in (2, 5, 10, 20):
            row = [gap]
            for phase in ("first2", "two_to_five", "six_to_eleven"):
                methods = ["profile_startup_native_recovery", "ieks_profile", "old_lifted"]
                if phase == "first2":
                    assert s.loc[(domain, gap, methods[0], phase), "median_xy_m"] == s.loc[
                        (domain, gap, methods[1], phase), "median_xy_m"]
                    methods = methods[1:]
                row += [s.loc[(domain, gap, m, phase), "median_xy_m"] for m in methods]
            rows.append(row)
    return check_table("loop", "Delft&2", rows)


if __name__ == "__main__":
    print(f"Verified {endpoint_tables()+external_tables()+closed_table()} numerical table cells.")
