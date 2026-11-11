"""Test the planar frontend on an exact coplanar mirror pair and an unobservable line."""

import numpy as np

from translation_bridge import projected_xy


def main():
    """Assert identical horizontal recovery under height reflection and a rank-aware fallback."""
    tau = np.linspace(-2.,0.,100)
    survey = np.array([[-2.,-3.,.6],[-1.,4.,.6],[3.,-2.,.6],[5.,3.,.6]])
    anchors = survey[np.arange(len(tau))%len(survey)]
    pv = np.array([.3,-.4,2.1,.7,.3,.2]); reflected = pv.copy()
    reflected[2] = 2*survey[0,2]-pv[2]; reflected[5] *= -1.
    paths = [s[:3]+tau[:,None]*s[3:] for s in (pv,reflected)]
    d = [np.linalg.norm(p-anchors,axis=1) for p in paths]
    std = np.full(len(tau), .25)
    ix = [0,1,3,4]
    estimates = [projected_xy(tau,anchors,r,std,p) for r,p in zip(d,(pv,reflected))]
    error = max(np.max(np.abs(v[ix]-pv[ix])) for v,_,_ in estimates)
    assert max(np.max(np.abs(d[0]-d[1])),error) < 1e-9
    line = anchors.copy(); line[:,1]=0.
    distances = np.linalg.norm(paths[0]-line,axis=1)
    _,_,fallback = projected_xy(tau,line,distances,std,pv)
    assert fallback and not any(v[2] for v in estimates)
    print(f'Coplanar mirror horizontal max error: {error:.3g} m or m/s; line rejected.')


if __name__ == '__main__':
    main()
