# Published range-only endpoint comparison

This experiment independently evaluates **Pacholska, Dümbgen and Scholefield,
“Relax and Recover: Guaranteed Range-Only Continuous Localization,” IEEE RA-L
5(2), 2248–2255 (2020)**, DOI [10.1109/LRA.2020.2970952](https://doi.org/10.1109/LRA.2020.2970952).
It uses the same 66 physical Delft TWR windows and two anchor layouts as the
existing experiment: **132 configurations per setting**, without replicated yaw
frames. All configurations are retained in the endpoint summaries.

## Source and frozen settings

- [Author repository](https://github.com/LCAV/continuous-localization), MIT license,
  pinned revision `b2a2b58205a3c33d79c10615e22a3252a55ba090`.
  Verbatim numerical sources and license are in `sources/`.
- The published article is from the [EPFL author deposit](https://infoscience.epfl.ch/server/api/core/bitstreams/77c56102-6fa2-47ce-9b13-3078ef2580be/content?download=1).
  The article is linked rather than redistributed. The numerical source snapshots and
  MIT license below are sufficient for the literal-function check.
- `relax_recover_K2` and `relax_recover_weighted_K2` implement the author's
  unweighted and weighted linear reconstruction. The author's
  `scripts/generate_results_polynomial.py` explicitly sets
  `list_complexities = [2]`; the published real-data polynomial evaluation uses
  linear trajectory parts (Section VI-B and Table I). Here **K counts basis
  functions**, so K=2 means `[1,t]`, degree one. This is a specific published
  polynomial setting, not a universal default for every trajectory.
- `relax_recover_weighted_K3_selected_here` uses the same supported polynomial
  solver with `[1,t,t²]`. It was prespecified here as a quadratic motion-model
  control before its results were computed. It is **not an author default or a
  claim that the original experiments used this Delft setting**. No other
  degrees, model families, regularizers or initializations were searched.
- Weighted rows use the **exact author-code rule**
  `1/sqrt(calibrated_range**2 + 0.1)`. The paper describes an approximate inverse
  range weight plus regularization; the precise `0.1 m²` value comes from
  `source/constraints.py::get_constraints`. There is no fitted variance weight,
  robust refinement, squared-noise correction, position prior or height bound.

## Formula and scope

For event n, let `f_n=[1,s_n,...,s_n^(K-1)]`,
`s_n=(t_n-t_endpoint)/scale`. The time scale is the observed event-time span
(at least one second), an invertible change of polynomial coordinates.

\[
 r(t_n)=C f_n,\qquad
 b_n=\tfrac12(\|a_n\|^2-\tilde d_n^2),\qquad
 b_n=a_n^T C f_n-\tfrac12 f_n^T L f_n.
\]

The original nonconvex relation `L=CᵀC` is discarded. Define rows
`T_A[n]=vec(a_n f_nᵀ)ᵀ`, `T_B[n]=vec(f_n f_nᵀ)ᵀ`. After optional row
weighting and `T_B=UΣVᵀ`, retain its `2K-1` polynomial columns and solve

\[
 \min_{c,q}\|[T_A,-\tfrac12U_r\Sigma_r][c^T,q^T]^T-b\|_2^2.
\]

`run.py` directly implements paper equations (5)–(6) and the author solver's
SVD reduction. It replaces the author's inverse of the normal equations with
SVD least squares for numerical stability, with the same full-rank linear
objective. The endpoint position is the constant coefficient `C[:,0]`.

The method is **range-only**: it consumes measurement times, calibrated TWR
ranges and surveyed anchors. It does not consume IMU acceleration, angular rate,
orientation, inertial displacement, or an endpoint position seed. Test truth is
used only after the fit to score position; the shared dataset loader still uses
OptiTrack availability to crop its common clock. The 0.20 s common sensor delay
and discarded 1.5 s static prefix reproduce the comparison's clock convention,
not an IMU input to this estimator.

The original guarantee requires a matching trajectory model, noiseless data and
nondegenerate measurement/anchor geometry. The Delft windows need not be exact
linear/quadratic trajectories; lower4 has weak vertical geometry. Full numeric
rank does **not** certify global optimality of the original raw-range objective,
accuracy under noise, or satisfaction of `L=CᵀC`. This baseline cannot be ranked
by objective misses for the IMU-assisted Huber problem: its sensor inputs,
trajectory model and squared-range objective differ. Endpoint errors are a
descriptive comparison on the same observations and clock.

## Common data and scoring

- Survey: dataset `anchor_positions.yaml`; layouts are all IDs 0–7 and IDs 0–3
  (the same lower four anchors used by `joint_branch/study.py`).
- Calibration: `../radio_inertial_models/radio_branch/results/calibration.json`,
  only its `static` per-anchor constant corrections are applied. Its frozen
  `sigma` values enter only the shared Huber-2 **diagnostic**, never this solver.
- Split: the six evaluation files in `radio_branch/results/protocol.json`.
  Earlier separate logs supplied training calibration. This is not a blind test.
- Windows: endpoints 5/10/20/30 s; durations 2/4/8 s no greater than the endpoint.
  Include only changed positive raw transactions that arrived by the actual
  endpoint, retaining their individual timestamps. No held-value resampling.
- The script asserts all 132 window keys and event counts match frame zero of
  `evidence/paired_acquisition/delft_same_profile.csv`. Each new event array also
  has a SHA-256 in `windows.csv`.
- No initialization or truth-dependent model selection. Position, horizontal and
  absolute vertical endpoint errors use the original world coordinates. All
  failures would remain rows with infinite scored errors and explicit failure
  counts; none occurred here. The overlapping windows and reused layouts are
  descriptive observations, not 132 independent trials.

## Results (metres)

| Setting | Layout | 3D median | 3D p95 | 3D RMSE | Horizontal median | Vertical median |
|---|---|---:|---:|---:|---:|---:|
| Original K2 unweighted | all8 | 0.606 | 1.551 | 0.793 | 0.315 | 0.292 |
| Original K2 weighted | all8 | 0.569 | 1.550 | 0.788 | 0.316 | 0.274 |
| K3 weighted, selected here | all8 | 0.532 | 1.023 | 0.606 | 0.205 | 0.411 |
| Original K2 unweighted | lower4 | 4.111 | 10.434 | 5.844 | 0.389 | 4.096 |
| Original K2 weighted | lower4 | 3.999 | 11.468 | 6.305 | 0.380 | 3.946 |
| K3 weighted, selected here | lower4 | 5.480 | 15.409 | 7.996 | 0.281 | 5.451 |

All 396 fits were finite and full numeric column rank: 9 columns for K2 and 14
for K3. The lower4 errors are predominantly vertical and remain in every summary;
no clipping, mirrored-height selection or IMU correction was applied. These are
observed failures in endpoint accuracy despite solvable linear systems.

Sequential solve medians were approximately 0.3–0.7 ms on this host; this includes
matrix construction, SVD and least squares, excludes CSV decoding and offline
truth scoring, and is not a general hardware latency guarantee.

## Reproduction and validation

From the project root in Windows PowerShell:

```powershell
& '.venv/Scripts/python.exe' 'imu-uwb-inekf/experiments/published_range_only/run.py'
```

One process is used, with OpenBLAS/MKL/OMP threads fixed to one before NumPy
import. No new dependency is required by the project environment. The existing
loader is imported for common data handling, but inertial integration is never
called. The dataset root follows `INEKF_TWR_DATASET` or the existing Delft path.

`results/protocol.json` records versions, input hashes, precise calibration,
author revision and setting provenance. `results/windows.csv` contains all 396
fits, estimated positions, scoring truth, event identity, rank, condition and
range diagnostic. `overall_summary.csv`, `per_log_summary.csv`, and
`duration_summary.csv` retain both layouts and all three settings.
`endpoint_error_cdf.png` displays the complete endpoint-error distributions.

The executable validation checks noiseless 3D polynomial recovery (largest
coefficient error `1.83e-14`) and compares all three implementations with the
**literal pinned author function**, loaded without unrelated CVXPY imports.
The real-data panel is 6 logs × 2 layouts × 3 settings at the fixed 10 s/4 s
window: maximum coefficient difference `1.91e-10`, relative difference
`6.18e-12`. It also asserts all 132 matched event counts. Details are retained
in `results/validation.json`.
