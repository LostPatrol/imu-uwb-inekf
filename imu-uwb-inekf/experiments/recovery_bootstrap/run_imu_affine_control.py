"""Control polynomial order using the same frozen two-translation-seed solver.

All inputs and scoring are inherited from the quadratic physical-IMU ablation;
only solve_polynomial's degree argument is fixed to one instead of two.
"""

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import os

# Each process uses one BLAS thread, matching both preceding experiments.
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
import pandas as pd
import polynomial_bias
import run_imu_polynomial_ablation as ablation
from run_imu_mismatch_stress import HERE


def evaluate_case(specification):
    """Reuse identical physical data and metrics with the unchanged affine solver."""
    original = polynomial_bias.solve_polynomial
    def affine(t, d, a, r, sigma, degree=2):
        """Fix only the public model-degree argument of the frozen implementation."""
        return original(t, d, a, r, sigma, degree=1)
    polynomial_bias.solve_polynomial = affine
    try:
        rows = ablation.evaluate_case(specification)
    finally:
        polynomial_bias.solve_polynomial = original
    for row in rows:
        row["method"] = "affine_two_translation_seeds"
    return rows


def main():
    """Freeze the same-policy control and independently rerun twenty fixed cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--input", type=Path, default=HERE.parents[1]/"results/recovery_imu_mismatch_20261107")
    parser.add_argument("--output", type=Path, default=HERE.parents[1]/"results/recovery_imu_affine_control_20261107")
    args = parser.parse_args()
    parent = json.loads((args.input/"protocol.json").read_text(encoding="utf-8"))
    cases = parent["cases"]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.audit:
        cases = [cases[index] for index in parent["audit_case_ids"]]
    else:
        if (args.output/"metrics.csv").exists():
            raise FileExistsError("Completed results will not be overwritten")
        protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), degree=1,
            parent_protocol_sha256=hashlib.sha256((args.input/"protocol.json").read_bytes()).hexdigest(),
            cases=cases, workers=args.workers, blas_threads=1,
            solver="Unmodified solve_polynomial(degree=1); same two translation-seed strategy as degree=2",
            seeds="Lift translation and robust CV translation, each with that model order's lifted yaw",
            scope="Model order and its naturally induced lift change; seed count, fitting loss, limits, and scoring match K=2",
            source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                (Path(__file__), HERE/"run_imu_polynomial_ablation.py", HERE/"run_imu_mismatch_stress.py",
                 HERE/"polynomial_bias.py", HERE/"initializer.py")})
        (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        frame = pd.DataFrame([row for group in pool.map(evaluate_case, cases) for row in group])
    frame = frame.sort_values(["case_id", "level"]).reset_index(drop=True)
    if args.audit:
        saved = pd.read_csv(args.output/"metrics.csv")
        subset = saved[saved.case_id.isin(parent["audit_case_ids"])].sort_values(
            ["case_id", "level"]).reset_index(drop=True)
        columns = [name for name in frame.select_dtypes(include=np.number) if name != "elapsed_s"]
        np.testing.assert_allclose(frame[columns], subset[columns], rtol=1e-12, atol=1e-11)
        for name in ("solver_success", "finite", "diagnostic_accepted", "accepted_but_inaccurate"):
            np.testing.assert_array_equal(frame[name], subset[name])
        result = dict(cases=len(cases), rerun_fits=len(frame),
            maximum_numeric_difference=float(np.max(np.abs(frame[columns].to_numpy()-subset[columns].to_numpy()))),
            all_outputs_finite=bool(saved.finite.all()), completed_utc=datetime.now(timezone.utc).isoformat())
        (args.output/"audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
    else:
        frame.to_csv(args.output/"metrics.csv", index=False)
        summary = ablation.summarize(frame)
        summary.to_csv(args.output/"summary.csv", index=False)
        print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
