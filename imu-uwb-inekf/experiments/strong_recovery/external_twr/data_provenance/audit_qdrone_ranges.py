"""Check raw TWR ranges against published timestamp-bound prism ground truth."""

from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np

# These are published aggregate dataset MAEs, not tuning targets.
PUBLISHED_MAE = {"indoor": .37, "field": .17, "building": 2.02, "bridge": .34, "tunnel": 1.61}


def normalized_name(name):
    """Match GT copies whose author filenames differ only by suffix or case."""
    return name.lower().replace("_groundtruth_addedtest", "").replace("_groundtruth_added", "")


def read_radio(path, ground_truth=False):
    """Read 0-events and bind only their next supplied XYZ row, never height GT."""
    events, pairs = [], []
    pending, previous_radio = None, None
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if not row or not any(x.strip() for x in row):
                continue
            try:
                values = [float(x) if x.strip() else np.nan for x in row]
            except ValueError:
                continue
            if pending is not None:
                if pending == 0:
                    previous_radio = values[:4]
                    events.append(previous_radio)
                else:
                    previous_radio = None
                pending = None
                continue
            finite_count = np.sum(np.isfinite(values))
            unlabelled_xyz = ground_truth and finite_count == 3
            labelled_xyz = ground_truth and values[0] == -1
            if unlabelled_xyz or labelled_xyz:
                if previous_radio is not None:
                    xyz = values[:3] if unlabelled_xyz else values[1:4]
                    pairs.append([*previous_radio, *xyz])
                previous_radio = None
                continue
            if values[0] in [0, 2, 3] and all(not np.isfinite(x) or x == 0 for x in values[1:]):
                pending = int(values[0])
                previous_radio = None
    return events, np.asarray(pairs)


def summary(residual):
    """Report signed median and absolute error without fitting any parameter."""
    absolute = np.abs(residual)
    return {"n": len(residual), "signed_bias_median_m": float(np.median(residual)),
            "abs_p50_m": float(np.median(absolute)), "abs_p95_m": float(np.quantile(absolute, .95)),
            "mae_m": float(np.mean(absolute)), "abs_max_m": float(np.max(absolute)),
            "abs_gt10m_count": int(np.sum(absolute > 10)),
            "abs_gt100m_count": int(np.sum(absolute > 100))}


def main():
    """Audit all thirteen GT recordings with a fixed publisher-coordinate mapping."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("E:/datasets/qdrone-benchmark/extracted"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    anchor_file = Path(__file__).resolve().parent / "qdrone_anchors_final.json"
    anchors = json.loads(anchor_file.read_text(encoding="utf-8"))
    originals = {normalized_name(path.name): path for path in (args.root / "All Datasets").rglob("*.csv")}
    records, site_errors, extreme_packets = [], {}, []
    for path in sorted((args.root / "Dataset with Ground Truth").rglob("*.csv")):
        site = path.relative_to(args.root / "Dataset with Ground Truth").parts[0].removesuffix("-gt")
        original = originals[normalized_name(path.name)]
        raw_events, _ = read_radio(original)
        _, pairs = read_radio(path, ground_truth=True)
        raw_keys = {tuple(event[:3]) for event in raw_events}
        exact = [tuple(pair[:3]) in raw_keys for pair in pairs]
        assert all(exact), str(path)
        # Hardware self-range-error metadata is not part of the physical residual.
        finite = np.all(np.isfinite(pairs[:, [0, 1, 2, 4, 5, 6]]), axis=1)
        accepted = pairs[finite]
        anchor = np.array([anchors[site][str(int(pair[1]))] for pair in accepted])
        residual = accepted[:, 2] - np.linalg.norm(accepted[:, 4:7] - anchor, axis=1)
        # This diagnostic threshold reports source corruption without removing packets.
        for index in np.flatnonzero(np.abs(residual) > 100):
            extreme_packets.append({"site": site, "record": original.stem,
                                    "timestamp_s": float(accepted[index, 0]),
                                    "anchor_id": int(accepted[index, 1]),
                                    "raw_range_m": float(accepted[index, 2]),
                                    "self_range_error_metadata": float(accepted[index, 3]),
                                    "prism_xyz_m": accepted[index, 4:7].tolist(),
                                    "residual_m": float(residual[index])})
        site_errors.setdefault(site, []).append(residual)
        row = {"site": site, "record": original.stem, "raw_path": str(original), "gt_path": str(path),
               "paired_radio_count": len(pairs), "nonfinite_pair_count": int(np.sum(~finite)),
               "first_valid_time": float(accepted[0, 0]), "last_valid_time": float(accepted[-1, 0]),
               **summary(residual)}
        records.append(row)
    sites = []
    for site, errors in site_errors.items():
        pooled = np.concatenate(errors)
        sites.append({"site": site, "record_count": len(errors), "published_full_dataset_mae_m": PUBLISHED_MAE[site],
                      "equal_record_mean_mae_m": float(np.mean([np.mean(np.abs(x)) for x in errors])),
                      **summary(pooled)})
    result = {"formula": "e = raw_range_m - norm(published_prism_xyz_m - published_anchor_xyz_m)",
              "calibration_or_alignment_fit": "none", "id_mapping": "raw100+i -> publisherUWB_i",
              "anchor_json_sha256": hashlib.sha256(anchor_file.read_bytes()).hexdigest(),
              "comparison_limit": "Published MAEs aggregate the original23-record collection; only13 GT copies are public here. Prism/tag/IMU levers and independent original RTS timestamp interpolation are not supplied. Supplied per-event XYZ used only for this post-hoc audit.",
              "records": records, "sites": sites, "extreme_packets_abs_gt100m": extreme_packets}
    (args.output / "qdrone_range_truth_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, rows in [("qdrone_range_truth_records.csv", records), ("qdrone_range_truth_sites.csv", sites)]:
        with (args.output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(sites, indent=2))


if __name__ == "__main__":
    main()
