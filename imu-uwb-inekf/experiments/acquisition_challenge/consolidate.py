"""Collect compact final evidence, enforcing the actual paired design and clocks."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from score import DEFAULT_DATA, HERE, SCRATCH

# Only scalar evidence, the acquisition figure and input provenance are published.
FILES = ("input_audit.json", "acquisition_scores.csv", "snapshot_scores.csv", "cold_summary.csv",
         "cold_recordings.csv", "cold_paired.csv", "cold_acquisition.png", "recovery_summary.csv",
         "recovery_by_treatment.csv", "recovery_by_recording.csv")


def verify(scores, snapshots):
    """Check expected independent native records, active input semantics and all finite/failed cases."""
    sensor = scores[scores.clock == "sensor"]
    reference = scores[scores.clock == "measured_queue"]
    index = ["episode", "method", "case_id", "protocol"]
    assert not sensor.duplicated(index).any()
    assert set(map(tuple, sensor[index].to_numpy())) == set(map(tuple, reference[index].to_numpy()))
    cold = sensor[(sensor.kind == "start") & (sensor.protocol == "native")]
    expected = {"SFUISE": 8, "Jiang_independent": 8, "profile1": 8, "body_bias_prior": 8,
                "Pfeiffer_EKF": 6, "Pfeiffer_MHE": 6}
    assert cold.groupby("method").size().to_dict() == expected
    stress = sensor[sensor.case_id != "native"]
    registered = {("SFUISE", "seed_trajectory15"): 768,
                  ("body_bias_prior", "recovery_MAP_trajectory"): 768,
                  ("Pfeiffer_EKF", "conditional_seed_or_prior"): 576,
                  ("Pfeiffer_MHE", "conditional_seed_or_prior"): 96}
    for (method, protocol), count in registered.items():
        selected = stress[(stress.method == method) & (stress.protocol == protocol)]
        assert len(selected) == count, (method, protocol, len(selected), count)
    sf = stress[(stress.method == "SFUISE") & (stress.protocol == "seed_trajectory15")]
    joined = sf.pivot(index=["episode", "direction", "magnitude_m", "yaw_delta_deg"],
                      columns="confidence", values="xy_10s_m")
    assert len(joined) == 384 and np.allclose(joined.wide, joined.tight, equal_nan=True)
    fixed = snapshots[(snapshots.case_id != "native") & (snapshots.arrival_s == 10)]
    assert len(snapshots) == 8 * 2 * 97 * 2 + 768  # Episodes, horizons, cases (including native), window models.
    assert fixed.groupby("method").size().to_dict() == {
        "Jiang_independent": 768, "body_bias_prior": 768, "profile1": 768}
    graph = stress[(stress.method == "Jiang_independent") & (stress.protocol == "graph_MAP_trajectory")]
    assert len(graph) == 768 and graph.estimate_count.max() <= 6
    window = stress[(stress.method == "body_bias_prior") & (stress.protocol == "recovery_MAP_trajectory")]
    for name, tracks in (("Jiang_independent", graph), ("body_bias_prior", window)):
        endpoint = fixed.loc[fixed.method == name, ["episode", "case_id", "xy_m"]]
        paired = tracks[["episode", "case_id", "xy_10s_m"]].merge(
            endpoint, on=["episode", "case_id"], validate="one_to_one")
        assert len(paired) == 768 and np.allclose(paired.xy_10s_m, paired.xy_m,
                                                atol=1e-5, rtol=1e-6, equal_nan=True)
    return dict(native_start_by_method=expected, stress_protocol_counts={
        f"{k[0]}/{k[1]}": n for k, n in registered.items()},
        sf_distinct_seed_centers=len(joined), jiang_continued_cases=len(graph),
        xy_snapshots_at_10s_by_method=fixed.groupby("method").size().to_dict())


def main():
    """Validate complete normalized outputs before copying scalar, shareable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    args = parser.parse_args()
    dst = args.out
    dst.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.results / "acquisition_scores.csv")
    snapshots = pd.read_csv(args.results / "snapshot_scores.csv")
    checks = verify(scores, snapshots)
    for filename in FILES:
        shutil.copyfile(args.results/filename, dst/filename)
    extras = {DEFAULT_DATA/"delft_manifest.csv": "delft_inputs.csv",
              DEFAULT_DATA/"scoring/isas_walk1_calibration.json": "isas_training_calibration.json",
              DEFAULT_DATA/"scoring/isas_reference_diagnostics.json": "isas_reference_diagnostics.json",
              DEFAULT_DATA/"scoring/isas_reference_diagnostics.png": "isas_reference_diagnostics.png",
              SCRATCH/"profile_challenge/results/audit.json": "window_coverage_audit.json",
              SCRATCH/"profile_challenge/diagnosis/prefix_statistics.csv": "window_prefix_diagnostics.csv",
              SCRATCH/"profile_challenge/diagnosis/diagnostic_fits.csv": "window_bias_diagnostic_fits.csv",
              SCRATCH/"profile_challenge/diagnosis/diagnostic_summary.csv": "window_bias_diagnostic_summary.csv",
              SCRATCH/"profile_challenge/diagnosis/diagnostic_protocol.json": "window_bias_diagnostic_protocol.json",
              SCRATCH/"profile_challenge/diagnosis/findings.md": "window_bias_findings.md",
              SCRATCH/"profile_challenge/diagnosis/blind_reset_paired_by_record.csv": "blind_reset_paired_by_record.csv",
              SCRATCH/"profile_challenge/diagnosis/oracle_physics_expressivity.csv": "oracle_physics_expressivity.csv",
              SCRATCH/"sfuise_challenge/anchor_warmup.json": "sfuise_anchor_io.json",
              SCRATCH/"sfuise_challenge/validation_native/validation.json": "sfuise_native_validation.json",
              SCRATCH/"sfuise_challenge/validation_stress15/validation.json": "sfuise_stress_validation.json",
              SCRATCH/"sfuise_challenge/native_build_manifest.json": "sfuise_build_manifest.json"}
    for source, name in extras.items():
        shutil.copyfile(source, dst/name)
    (dst/"coverage.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
