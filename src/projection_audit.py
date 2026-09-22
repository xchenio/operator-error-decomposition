"""Same-grid, same-sample orthogonal error decomposition (offline, uses labels)."""
import numpy as np


def projection_audit(features, truth, prediction, rcond=1e-12):
    """Matrices are (points, features) and (points, samples).

    Caller must include bias and apply any lifting to both targets and predictions.
    Numerical rank is explicit: truncated results are conditional on this tolerance.
    The in-space term is a coefficient/readout discrepancy, NOT pure optimization.
    """
    a, s, y = [np.asarray(v, dtype=np.float64) for v in (features, truth, prediction)]
    if a.ndim != 2 or s.ndim != 2 or s.shape != y.shape or a.shape[0] != s.shape[0]:
        raise ValueError("incompatible feature/solution shapes")
    if not all(np.isfinite(v).all() for v in (a, s, y)):
        raise ValueError("nonfinite input")
    u, sv, _ = np.linalg.svd(a, full_matrices=False)
    rank = int(np.sum(sv > rcond * sv[0])) if len(sv) else 0
    q = u[:, :rank]
    projected_s = q @ (q.T @ s)
    projected_y = q @ (q.T @ y)
    den = np.sum(s*s)
    if den <= 0:
        raise ValueError("zero target energy")
    rep = np.sum((s-projected_s)**2) / den
    coefficient = np.sum((projected_s-projected_y)**2) / den
    total = np.sum((s-y)**2) / den
    norms = np.linalg.norm(s, axis=0)
    if np.any(norms == 0):
        raise ValueError("zero-energy sample")
    return dict(rank=rank, rcond=rcond, representation_sq=float(rep),
                coefficient_sq=float(coefficient), total_sq=float(total),
                closure_error=float(total-rep-coefficient),
                prediction_outside_span=float(np.linalg.norm(y-projected_y)/np.sqrt(den)),
                projection_relative_fro=float(np.sqrt(rep)),
                actual_relative_fro=float(np.sqrt(total)),
                projection_mean_relative_l2=float(np.mean(np.linalg.norm(s-projected_s,axis=0)/norms)),
                actual_mean_relative_l2=float(np.mean(np.linalg.norm(s-y,axis=0)/norms)))
