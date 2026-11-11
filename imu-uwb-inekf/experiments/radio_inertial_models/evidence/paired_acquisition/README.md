<!-- Paired endpoint and seed results; recorded quantities and scientific scope. -->
# Matched acquisition evidence

The frozen reference is `tex/recovery_bootstrap_paper.pdf` at `d6dd74e`. Its Profile-1 yaw candidate and two CV translation fits are identical on both sides. The new raw-range fit uses a rotating body accelerometer bias with the same fixed 0.10 m/s² prior throughout. Delft additionally uses the previously trained geometric radio mean. The four solved yaw bases are equivalent coordinates of the same data.

| File | What one row represents |
|---|---|
| `delft_same_profile.csv` | Same old/new Profile-1 endpoint; six flights, eleven physical windows per flight, two layouts and four yaw bases (528 rows). Both counts include the two CV fits. |
| `delft_new_objective_seeds.csv` | Same new raw objective with Local, Profile-1 or four uniform starts, including capped fits. An objective miss exceeds `1e-4 * (1 + best_observed)` above the best observed in those three arms. |
| `delft_frame0_controls.csv` | All matched endpoint controls at frame zero, by layout and axis, including free quadratic motion, static/geometric radio means and exclusion of the target motion from calibration. |
| `qdrone_window_controls.csv` | Ten QDrone recordings in four sites, one fixed middle endpoint per recording and windows of 2/4/8 s. The initial comparison and subsequent test of seed choice are labelled separately. Their four-start bias rows correspond to the same estimates and must not be pooled. |

Delft endpoint XY RMSE for the same Profile-1 policy falls from 1.625 to 0.159 m at 32,632 versus 32,789 function evaluations. Every flight improves in aggregate; individual windows can get worse. The raw objective changes, so its numerical value cannot be compared across models. Within the new objective the three seed policies miss 7, 0 and 0 out of 528. At the reference basis a free quadratic with the same radio calibration gives 0.154 m versus 0.159 m for the bias model.

For QDrone, the registered public position is the total-station prism, and its exact transform to IMU and radio is not known. All arms use the same IMU prefix and zero radio mean, without test-truth alignment or transferring Delft anchor coefficients. Profile-1 with the new motion model gives XY 1.631 m vs 2.930 m for the original affine model. Two-second windows do not improve in aggregate. No heading accuracy or closed-loop recovery claim follows from either endpoint table.
