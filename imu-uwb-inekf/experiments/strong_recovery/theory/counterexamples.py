"""Small algebraic recovery counterexamples, with no real dataset or fitted threshold."""

import json

import numpy as np

# Fixed geometric rotation and error scales define illustrative, not tuned, examples.
YAW = 1.1
OLD_VELOCITY_ERROR = 5.
RADIO_VARIANCE = .3**2


def main():
    """Verify the covariance/observability claims and print their exact numerical consequences."""
    actual = np.array([[1., .8], [.8, 1.]])
    naive = np.eye(2)
    conservative = 2*np.eye(2)
    assert np.linalg.eigvalsh(naive-actual)[0] < 0.
    assert np.linalg.eigvalsh(conservative-actual)[0] > 0.
    angle = YAW
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    times = np.linspace(0., 2., 41)
    path = np.column_stack((2+.2*times, np.sin(times), 1.4+.1*np.cos(times)))
    acceleration = np.column_stack((np.zeros_like(times), -np.sin(times), -.1*np.cos(times)))
    anchors = np.array([[0., 0., z] for z in (0., 1., 2., 3.)])
    original = np.linalg.norm(path[:, None]-anchors[None], axis=2)
    changed = np.linalg.norm((path@rotation.T)[:, None]-anchors[None], axis=2)
    gravity = np.array([0., 0., -9.81])
    original_specific = acceleration-gravity
    rotated_specific = ((acceleration@rotation.T)-gravity)@rotation
    range_difference = float(np.max(abs(original-changed)))
    imu_difference = float(np.max(abs(original_specific-rotated_specific)))
    assert range_difference < 1e-12 and imu_difference < 1e-12
    # A single radial direction identifies only radial p and v in this local linearization.
    direction = np.tile([1., 0., 0.], (len(times), 1))
    design = np.column_stack((direction, times[:, None]*direction))
    assert np.linalg.matrix_rank(design) == 2
    true_range = np.linalg.norm([0., 0., 1.])
    lost_range = np.linalg.norm([30., 0., 1.])
    lost_nis = (lost_range-true_range)**2/(.2**2+RADIO_VARIANCE)
    # Identical radio scale distinguishes predictions; NIS alone can reward a broad bad branch.
    good_residual, bad_residual, good_s, bad_s = .2, 2., .1**2, 10.**2
    good_nis, bad_nis = good_residual**2/good_s, bad_residual**2/bad_s
    good_nll = .5*(good_nis+np.log(good_s))
    bad_nll = .5*(bad_nis+np.log(bad_s))
    assert bad_nis < good_nis and good_nll < bad_nll
    print(json.dumps(dict(
        unknown_cross_naive_bound_min_eigenvalue=float(np.linalg.eigvalsh(naive-actual)[0]),
        unknown_cross_inflated_bound_min_eigenvalue=float(np.linalg.eigvalsh(conservative-actual)[0]),
        position_error_1s_after_perfect_p_only_reset_m=OLD_VELOCITY_ERROR,
        vertical_anchor_yaw_rotation_rad=YAW, maximum_range_change_m=range_difference,
        maximum_body_specific_force_change_mps2=imu_difference,
        radial_pv_jacobian_rank=int(np.linalg.matrix_rank(design)),
        available_pv_dimension=6, valid_radio_nis_after_30m_position_loss=float(lost_nis),
        good_branch_nis=good_nis, broad_bad_branch_nis=bad_nis,
        good_branch_gaussian_nll=good_nll, broad_bad_branch_gaussian_nll=bad_nll), indent=2))


if __name__ == '__main__':
    main()
