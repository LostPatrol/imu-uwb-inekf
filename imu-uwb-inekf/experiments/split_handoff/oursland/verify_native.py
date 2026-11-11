"""Verify compiled event orchestration against the literal Python call sequence."""

import json
import sys

# Imported source trees remain read-only.
sys.dont_write_bytecode = True
import native_schedule as native
import numpy as np


def main():
    """Check matching states, trajectories, packet counts, units and thread limits."""
    raw = native.load_raw("const1-trial4-tdoa2", False)
    raw.pop("truth")
    data = native.ours.util.load(raw["name"], duration=45.)
    data.pop("truth")
    start, state, bias = native.ours.util.initial_state(data)
    change = native.ours.util.kernel.mapping(state)
    covariance = change @ native.ours.util.PHYSICAL @ change.T
    python = native.replay(raw, state, covariance, bias, start, "current_kernel", None, False,
                           end=start + 1., return_final=True, use_compiled=False)
    compiled = native.replay(raw, state, covariance, bias, start, "current_kernel", None, False,
                             end=start + 1., return_final=True)
    differences = [float(np.max(np.abs(python[k] - compiled[k]))) for k in range(4)]
    for index in range(4):
        np.testing.assert_allclose(python[index], compiled[index], rtol=1e-12, atol=1e-12)
    if python[4:] != compiled[4:]:
        raise AssertionError("Compiled replay changed native acceptance/failure counts")
    result = dict(compiled_vs_literal_python_max_abs=dict(state=differences[0], covariance=differences[1], bias=differences[2], history=differences[3]),
                  acceptance_counts_equal=True, raw_acc_samples=len(raw["times"]), raw_radio_packets=len(raw["radio"]),
                  imu_acc_norm_median_g=float(np.median(np.linalg.norm(raw["imu"][:, :3], axis=1))),
                  native_imu_median_dt=float(np.median(np.diff(raw["times"]))), thread_pools=native.THREAD_POOLS,
                  original_source_sha256=native.ours.digest(native.ours.AUTHOR))
    (native.ours.HERE / "native_schedule_checks.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
