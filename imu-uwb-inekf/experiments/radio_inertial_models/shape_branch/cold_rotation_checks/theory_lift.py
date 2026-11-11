"""Direct squared-range full-rotation candidate without a radio trajectory prior."""

import numpy as np

from cases import GRAVITY


def lifted_rotation_seed(case, mean_correction=True):
    """Profile lifted translation/cross nuisances, then project a 3x3 fit onto SO(3)."""
    tau, h, anchors = case.tau, case.shape, case.anchors
    gravity_shape = .5*tau[:, None]**2*GRAVITY
    response = (.5*(case.ranges**2-np.sum(anchors*anchors, axis=1)-np.sum(h*h, axis=1))
                -.5*np.sum(gravity_shape*gravity_shape, axis=1)+np.sum(anchors*gravity_shape, axis=1))
    if mean_correction:
        response -= .5*case.sigmas**2
    # Free affine translation, its norm polynomial, and unknown C'p/C'v cross terms.
    nuisance = np.column_stack((np.ones(len(tau)), tau, tau**2, tau**3,
                                h, tau[:, None]*h, -anchors, -tau[:, None]*anchors))
    circular = np.einsum("ni,nj->nij", gravity_shape-anchors, h).reshape(-1, 9)
    weights = 1/np.sqrt(np.maximum(case.ranges, case.sigmas)**2*case.sigmas**2+.5*case.sigmas**4)
    linear = nuisance*weights[:, None]
    u, singular, _ = np.linalg.svd(linear, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    projected = np.column_stack((response, circular))*weights[:, None]
    projected -= u[:, :rank]@(u[:, :rank].T@projected)
    vector, _, projected_rank, rotation_singular = np.linalg.lstsq(projected[:, 1:], projected[:, 0], rcond=1e-10)
    raw_rotation = vector.reshape(3, 3)
    left, values, right = np.linalg.svd(raw_rotation)
    proper = left@np.diag([1., 1., np.linalg.det(left@right)])@right
    return proper, dict(nuisance_rank=rank, projected_rotation_rank=int(projected_rank),
                        projected_rotation_singular=rotation_singular.tolist(),
                        unconstrained_rotation_singular=values.tolist())
