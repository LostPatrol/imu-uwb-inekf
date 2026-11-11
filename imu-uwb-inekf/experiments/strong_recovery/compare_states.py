"""Publish paired outcomes for actual state changes, distinct from temporary output changes."""

import argparse
from pathlib import Path

import pandas as pd

from compare import EVIDENCE, HERE, NUM_TIE, PRIVATE, WINDOWS

# The three native data adapters name the same score windows differently.
LABELS = {"first2":"0-2", "two_to_five":"2-5", "six_to_eleven":"6-11"}


def long_pairs(data, domain, variant, fields):
    """Pair two recursions on the same observation schedule and ground-truth support."""
    data = data.rename(columns=fields).copy()
    data["window"] = data["window"].replace(LABELS)
    keys = ["sequence","gap_s","window"]
    base = data[data.method == "K"].set_index(keys)
    other = data[data.method != "K"].set_index(keys)
    if not base.index.is_unique or not other.index.is_unique or set(base.index) != set(other.index):
        raise ValueError("Paired state runs must have identical recording, outage and window keys.")
    pairs = base[["xy","coverage"]].rename(columns={"xy":"K_m"}).join(other[["xy"]].rename(columns={"xy":"candidate_m"}))
    return pairs.reset_index().assign(domain=domain, method=variant)


def collect():
    """Retain each actual feedback comparison, including unsupported QDrone windows."""
    parts = []
    data = pd.read_csv(PRIVATE/"strong_twr/feedback_results/paired_metrics.csv")
    for window, native in zip(WINDOWS, ("early0_2","early2_5","future6_11")):
        data["window"] = window
        parts.append(long_pairs(data, "Delft TWR", "post_update_write",
                                {native+"_xy_rmse_m":"xy",native+"_coverage":"coverage"}))
    data = pd.read_csv(HERE/"delft_tdoa/posthoc_state_projection/results/paired_xy.csv").rename(
        columns={"name":"sequence","outage_s":"gap_s","interval":"window","K":"K_m","single_state_projection":"candidate_m"})
    parts.append(data.assign(domain="Delft TDoA", method="post_update_write", coverage=1.0))
    data = pd.read_csv(HERE/"oursland/feedback_extension/feedback_scores/paired.csv")
    for window, native in zip(WINDOWS, ("early","middle","later")):
        data["window"] = window
        parts.append(long_pairs(data, "Toronto supplementary TDoA", "post_update_write",
            {"name":"sequence",native+"_xy_rmse":"xy",native+"_coverage":"coverage"}))
    data = pd.read_csv(PRIVATE/"strong_external/qdrone_feedback_verified/paired_metrics.csv")
    parts.append(long_pairs(data, "QDrone TWR", "post_update_write",
                            {"length_s":"gap_s","xy_rmse_m":"xy","truth_coverage":"coverage"}))
    for folder, label in [("delft","Delft TWR"), ("qdrone","QDrone TWR")]:
        data = pd.read_csv(PRIVATE/"strong_prior"/folder/"metrics.csv")
        parts.append(long_pairs(data, label, "bound_before_capture", {"length_s":"gap_s","xy_rmse_m":"xy"}))
    cols = ["domain","sequence","gap_s","window","method","K_m","candidate_m","coverage"]
    result = pd.concat(parts, ignore_index=True)[cols].sort_values(cols[:5])
    result["sequence"] = result.sequence.str.removesuffix(".csv").str.split("__").str[-1]
    # The reference already published for each native case must be identical in both studies.
    keys = ["domain","sequence","gap_s","window"]
    reference = pd.read_csv(EVIDENCE/"paired_xy.csv")[keys+["K","reference_coverage"]]
    check = result.merge(reference, on=keys, how="left", validate="many_to_one", indicator=True)
    if (check._merge != "both").any() or not check.K.isna().equals(check.K_m.isna()):
        raise ValueError("Missing a reference case or the two replays disagree about score support.")
    if (check.K_m-check.K).abs().max() > 1e-8 or (check.coverage-check.reference_coverage).abs().max() > 1e-8:
        raise ValueError("Native reference or its score support differs between paired experiments.")
    return result


def main():
    """Recompute summary from published state pairs or collect the native scoring files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    path = EVIDENCE/"state_pairs.csv"
    if args.collect:
        collect().to_csv(path,index=False)
    cases = pd.read_csv(path)
    keys = ["domain","gap_s","window","method"]
    rows = []
    for labels, group in cases.groupby(keys):
        eligible = group[group.coverage >= .95]
        pairs = eligible[["K_m","candidate_m"]].dropna()
        delta = pairs.candidate_m - pairs.K_m
        rows.append({**dict(zip(keys,labels)),"slots":len(group),"eligible":len(eligible),"pairs":len(pairs),
            "K_m":pairs.K_m.median(),"candidate_m":pairs.candidate_m.median(),"paired_delta_m":delta.median(),
            "wins":int((delta < -NUM_TIE).sum()),"ties":int((delta.abs() <= NUM_TIE).sum()),
            "losses":int((delta > NUM_TIE).sum()),"best_delta_m":delta.min(),"worst_delta_m":delta.max()})
    pd.DataFrame(rows).to_csv(EVIDENCE/"state_by_duration.csv",index=False)


if __name__ == "__main__":
    main()
