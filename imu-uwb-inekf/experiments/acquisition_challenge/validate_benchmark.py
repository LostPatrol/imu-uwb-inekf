"""Audit registered raw-input treatments and the separation of truth from replays."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Only the benchmarking/audit side imports scoring and ever opens the reference files.
from score import DEFAULT_DATA, HERE, track_score, truth_at
from prepare import PROTOCOL


def clocks_check():
    """Exercise delayed publication, moving truth, missing prefixes and terminal failures."""
    t = np.arange(0, 16, .5)
    ref = pd.DataFrame(dict(t=t, x=t, y=0., z=0.))
    track = pd.DataFrame(dict(arrival_s=[2., 2.5, 3., 3.5], state_s=[2., 2.5, 3., 3.5],
        wall_s=[.8, .1, .1, .1], x=[2., 2.5, 3., 3.5], y=0., z=0.))
    sensor = track_score(track, ref, "sensor", 15., "increment", "clock check")
    queue = track_score(track, ref, "measured_queue", 15., "increment", "clock check")
    assert sensor["confirm_050_s"] == 3. and queue["first_publish_s"] == 2.8
    assert np.isnan(queue["xy_2s_m"]) and queue["first_evaluable_grid_s"] == 3.
    assert not sensor["stayed_after_confirm_050"]
    track.loc[3, ["x", "y", "z"]] = np.nan
    failed = track_score(track, ref, "sensor", 15., "increment", "terminal check")
    assert failed["terminal_failure"] and np.isnan(failed["xy_10s_m"])
    first_unknown = track_score(track, ref[ref.t >= 2.5], "sensor", 15., "increment", "support check")
    assert np.isnan(first_unknown["first_xy_m"])


def audit_episode(folder, truth):
    """Verify dimensions, input hashes and all injected physical centers independently."""
    kind = folder.name.rsplit("_", 1)[-1]
    data_files = ["radio.csv", "imu.csv", "anchors.csv", "meta.json"]
    hashes = {key: hashlib.sha256((folder/key).read_bytes()).hexdigest() for key in data_files}
    imu = pd.read_csv(folder/"imu.csv")
    radio = pd.read_csv(folder/"radio.csv")
    anchors = pd.read_csv(folder/"anchors.csv")
    meta = json.loads((folder/"meta.json").read_text(encoding="utf-8"))
    assert set(imu.columns) == set("t ax ay az gx gy gz".split())
    assert {"t", "anchor_id", "range_m", "sigma_m"} <= set(radio.columns)
    assert not any(k in radio or k in imu for k in ["x", "y", "z", "yaw_rad", "gt_x", "otX"])
    assert imu.t.min() >= -1.5000001 and imu.t.min() <= -1.46 and np.any(np.isclose(imu.t, 0., atol=1e-8))
    assert radio.t.min() >= 0 and radio.t.max() <= 15.0001
    assert set(radio.anchor_id) <= set(anchors.anchor_id) and anchors.anchor_id.is_unique
    r0 = np.array(meta["R0"])
    assert np.allclose(r0.T @ r0, np.eye(3), atol=1e-8)
    delta_position, delta_yaw, num = np.nan, np.nan, 0
    if kind == "recovery":
        p = pd.read_csv(folder/"priors.csv")
        num = len(p)
        assert num == p.case_id.nunique() == 96 and len(
            p[["magnitude_m", "direction", "yaw_delta_deg", "confidence"]].drop_duplicates()) == 96
        assert set(p.magnitude_m) == set(PROTOCOL["prior_magnitude_xy_m"])
        assert set(p.direction) == set(PROTOCOL["prior_direction"])
        assert set(p.yaw_delta_deg) == set(PROTOCOL["prior_yaw_deg"])
        base = truth_at(truth, [0])[0]
        delta_position = float(np.abs(p[["px", "py"]].to_numpy() - base[:2] - p[["dx", "dy"]].to_numpy()).max())
        delta_yaw_raw = p.yaw_rad - (base[3] if np.isfinite(base[3]) else 0.) - np.deg2rad(p.yaw_delta_deg)
        delta_yaw = float(np.max(abs(np.arctan2(np.sin(delta_yaw_raw), np.cos(delta_yaw_raw)))))
        assert delta_position < 1e-7 and delta_yaw < 1e-7
        assert np.allclose(p.pz, anchors.z.mean())
        for confidence, (sp, sy) in PROTOCOL["prior_1sigma"].items():
            selected = p[p.confidence == confidence]
            assert len(selected) == 48 and np.allclose(selected.pos_std_m, sp)
            assert np.allclose(np.rad2deg(selected.yaw_std_rad), sy)
        hashes["priors.csv"] = hashlib.sha256((folder/"priors.csv").read_bytes()).hexdigest()
    else:
        assert not (folder/"priors.csv").exists()
    return dict(episode=folder.name, dataset=meta["dataset"], prefix_and_release_valid=True,
                independent_yaw_reference="yaw_rad" in truth, initial_xy_identity_max_m=delta_position,
                initial_yaw_identity_max_rad=delta_yaw, sealed_priors=num, imu=len(imu), radio=len(radio),
                ref_first_s=truth.t.min(), ref_last_s=truth.t.max(), hashes=hashes)


def main():
    """Save the reproducible shared-input audit and reject missing test episodes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=HERE/"results/input_audit.json")
    args = parser.parse_args()
    clocks_check()
    entries = []
    for folder in sorted((args.data/"streams").iterdir()):
        if folder.name.endswith(("_start", "_recovery")):
            ref = pd.read_csv(args.data/"scoring"/(folder.name+".csv"))
            entries.append(audit_episode(folder, ref))
    assert len(entries) == 16 and sum(x["sealed_priors"] for x in entries) == 768
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(tracks=len(entries), priors=768, causal_clock_checks="passed",
                                        episodes=entries), indent=2), encoding="utf-8")
    print(f"{len(entries)} episodes, 768 perturbed priors, clock/geometry checks passed")


if __name__ == "__main__":
    main()
