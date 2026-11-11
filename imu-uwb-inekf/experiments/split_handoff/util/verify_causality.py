"""Poison unavailable real radio packets and verify conditional handoff invariance."""

import importlib
import json
from pathlib import Path

import numpy as np

import run


def stable_diagnostics(value):
    """Exclude wall-clock profiling only; keep every statistical and fit diagnostic."""
    if isinstance(value, dict):
        return {key: stable_diagnostics(item) for key, item in value.items() if key != "elapsed_s"}
    if isinstance(value, list):
        return [stable_diagnostics(item) for item in value]
    return value


def condition(data, scenario):
    """Reproduce the sensor-only shared state and candidate inputs through handoff."""
    visible = run.visible_data(data, scenario)
    path = run.radio_path(visible)
    origin = visible["times"][0]
    rule = run.PROTOCOL["scenarios"][scenario]
    start = visible["times"][np.searchsorted(visible["times"], origin + rule["capture_start"])]
    end = visible["times"][np.searchsorted(visible["times"], origin + rule["capture_end"])]
    initial, state, bias = run.initial_state(visible)
    change = run.kernel.mapping(state)
    covariance = change @ run.PHYSICAL @ change.T
    outage = None if rule["outage_start"] is None else (rule["outage_start"], rule["outage_end"])
    if start > initial:
        state, covariance, bias, *_ = run.bounded_replay(visible, state, covariance, bias, initial, start, outage)
    window_bias = bias.copy()
    state, covariance, bias, *_ = run.bounded_replay(visible, state, covariance, bias, max(initial, start), end, outage)
    inputs = run.window_inputs(visible, start, end, window_bias, path)
    Cref = state[:3, :3] @ inputs["Qend"].T
    recent = visible["radio"][(visible["radio"][:, 0] > inputs["endpoint"] - .6)
                              & (visible["radio"][:, 0] <= inputs["endpoint"])]
    pv = run.fit_radio_pv(recent, visible["anchors"], inputs["endpoint"])
    full, diagnostic = run.fit_full_attitude(inputs, Cref, pv)
    decisions = {name: run.invoke_decision(importlib.import_module(module), inputs, Cref)
                 for name, module in [("evidence", "yaw_evidence"), ("bank", "bank.bank")]}
    return dict(state=state, covariance=covariance, bias=bias, inputs=inputs, pv=pv,
                full=full, full_diagnostic=diagnostic, decisions=decisions,
                path=path[path[:, 0] <= inputs["endpoint"]])


def main():
    """Use one fixed development recording; truth is removed before the sentinel runs."""
    data = run.load("const3-trial3-tdoa2", duration=45)
    data.pop("truth")
    results = {}
    for scenario in ["cold_weak", "dynamic", "outage"]:
        rule = run.PROTOCOL["scenarios"][scenario]
        origin = data["times"][0]
        endpoint = data["times"][np.searchsorted(data["times"], origin + rule["capture_end"])]
        poisoned = dict(data)
        poisoned["radio"] = data["radio"].copy()
        mask = poisoned["radio"][:, 0] > endpoint
        if rule["outage_start"] is not None:
            mask |= ((poisoned["radio"][:, 0] >= origin + rule["outage_start"])
                     & (poisoned["radio"][:, 0] < origin + rule["outage_end"]))
        poisoned["radio"][mask, 3] += 1e6
        base, changed = condition(data, scenario), condition(poisoned, scenario)
        differences = {}
        for key in ["state", "covariance", "bias", "pv", "full", "path"]:
            differences[key] = float(np.max(np.abs(base[key] - changed[key])))
        for key in ["tau", "force", "B", "plus", "minus", "y", "sigma", "path"]:
            differences["input_" + key] = float(np.max(np.abs(base["inputs"][key] - changed["inputs"][key])))
        assert max(differences.values()) == 0., differences
        assert json.dumps(stable_diagnostics(run.safe_json(base["decisions"])), sort_keys=True) == json.dumps(stable_diagnostics(run.safe_json(changed["decisions"])), sort_keys=True)
        if scenario == "outage":
            path = base["path"]
            assert not np.any((path[:, 0] > origin + 10) & (path[:, 0] < origin + 20))
            recovery = path[path[:, 0] > origin + 20]
            first_recovery = float(recovery[0, 0] - origin)
            assert first_recovery >= 20.19
        else:
            first_recovery = None
        results[scenario] = dict(poisoned_packets=int(mask.sum()), maximum_differences=differences,
                                 decisions_identical=True, excluded_diagnostic_fields=["elapsed_s"],
                                 first_position_seconds=float(base["path"][0, 0] - origin),
                                 first_recovery_seconds=first_recovery)
    folder = Path(__file__).parent / "verification"
    folder.mkdir(exist_ok=True)
    (folder / "causality.json").write_text(json.dumps(results, indent=2), encoding="utf8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
