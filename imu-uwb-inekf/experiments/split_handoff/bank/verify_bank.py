"""Check causal block separation and affine IMU gauge invariance on physical data.

Synthetic truth only constructs and scores these implementation checks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import bank

# Reuse physical held-IMU synthesis, rather than constructing arbitrary shapes.
CASES_DIR = Path(__file__).resolve().parents[2]/"radio_inertial_models"/"shape_branch"/"cold_rotation_checks"
sys.path.insert(0, str(CASES_DIR))
from cases import make_case


def stable_diagnostics(result):
    """Exclude runtime from deterministic decision and prediction diagnostics."""
    return {key: value for key, value in result.items() if key != "elapsed_s"}


def arguments(case, error_deg):
    """Build a shared candidate interface from raw asynchronous physical ranges."""
    reference = bank.yaw_rotation(np.deg2rad(error_deg))@case.truth_rotation
    return dict(tau=case.tau, force=case.shape, B=np.zeros((len(case.tau), 3, 3)),
                plus=case.anchors, minus=None, y=case.ranges, sigma=case.sigmas,
                path=np.zeros((len(case.tau), 3)), Cref=reference)


def main():
    """Run fixed smoke cases and invariance checks without selecting any rules."""
    out_dir = Path(__file__).with_name("verification")
    out_dir.mkdir(exist_ok=True)
    # This seed and the rules are fixed before receiving any UTIL scores.
    seed = 20261111
    result = {"synthetic_only": True, "oracle_tilt_supplied": True, "seed": seed, "rules_sha256": hashlib.sha256(bank.RULE_PATH.read_bytes()).hexdigest(),
              "bank_sha256": hashlib.sha256(Path(bank.__file__).read_bytes()).hexdigest(), "cases": []}
    rich = make_case(seed, motion="rich", tilt_deg=20.)
    for motion in ("rich", "weak", "constant_acceleration", "constant_velocity"):
        case = rich if motion == "rich" else make_case(seed, motion=motion, tilt_deg=20.)
        for error in (0., 90., 180.):
            fit = bank.decide(**arguments(case, error))
            result["cases"].append(dict(motion=motion, initial_error_deg=error, applied_delta_deg=float(np.rad2deg(fit["delta"])),
                accepted=fit["accepted"], posterior_mass=fit["posterior_mass"], sd_deg=fit["sd_deg"],
                blocks=fit["blocks"], nonconverged_prefix_fits=fit["nonconverged_prefix_fits"], elapsed_s=fit["elapsed_s"]))
    values = arguments(rich, 180.)
    original = bank.decide(**values)
    poisoned_path = dict(values, path=np.full_like(values["path"], np.nan))
    path_result = bank.decide(**poisoned_path)
    result["unused_path_identical"] = stable_diagnostics(original) == stable_diagnostics(path_result)
    # An endpoint change adds arbitrary affine terms to force and bias sensitivity.
    rng = np.random.default_rng(seed)
    affine = np.column_stack((np.ones(len(rich.tau)), rich.tau))
    gauge_values = dict(values, force=values["force"]+affine@rng.normal(size=(2, 3)),
        B=values["B"]+(affine@rng.normal(size=(2, 9))).reshape(-1, 3, 3))
    gauge_result = bank.decide(**gauge_values)
    result["affine_gauge_same_decision"] = original["delta"] == gauge_result["delta"] and original["accepted"] == gauge_result["accepted"]
    result["affine_gauge_max_loss_difference"] = float(np.max(np.abs(
        np.asarray([item["mean_prediction_losses"] for item in original["trace"]])-
        np.asarray([item["mean_prediction_losses"] for item in gauge_result["trace"]]))))
    # Corrupt only the final validation block: every earlier trace must be unchanged.
    corruption_start = original["trace"][-1]["prefix_end_tau"]
    changed_y = values["y"].copy()
    changed_y[values["tau"] >= corruption_start] += 10.
    future_result = bank.decide(**dict(values, y=changed_y))
    result["later_radio_preserves_earlier_traces"] = original["trace"][:-1] == future_result["trace"][:-1]
    short = values["tau"] < values["tau"][0]+2.
    short_values = {key: value[short] if key in ("tau", "force", "B", "plus", "y", "sigma", "path") else value
                    for key, value in values.items()}
    short_result = bank.decide(**short_values)
    result["two_second_defer"] = dict(accepted=short_result["accepted"], blocks=short_result["blocks"],
        delta=short_result["delta"])
    # A persistent coherent radio fault can exactly imitate another yaw trajectory.
    nominal_values = arguments(rich, 0.)
    false_rotation = bank.yaw_rotation(np.pi/2)@rich.truth_rotation
    false_path = (rich.truth_position+rich.tau[:, None]*rich.truth_velocity+
                  .5*rich.tau[:, None]**2*bank.GRAVITY+rich.shape@false_rotation.T)
    original_noise = rich.ranges-np.linalg.norm(rich.truth_tag_positions-rich.anchors, axis=1)
    corrupted_radio = np.linalg.norm(false_path-rich.anchors, axis=1)+original_noise
    coherent_result = bank.decide(**dict(nominal_values, y=corrupted_radio))
    result["coherent_radio_fault"] = dict(synthetic_constructed_fault=True,
        description="IMU and retained heading remain correct; radio imitates a trajectory rotated by +90 degrees",
        accepted=coherent_result["accepted"], applied_false_heading_deg=float(np.rad2deg(coherent_result["delta"])),
        posterior_mass=coherent_result["posterior_mass"], sd_deg=coherent_result["sd_deg"],
        radio_fault_rms_m=float(np.sqrt(np.mean((corrupted_radio-rich.ranges)**2))))
    out_path = out_dir/"implementation_checks.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
