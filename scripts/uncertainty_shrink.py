"""Continuous uncertainty-penalized coefficient risk; no new deletion threshold."""
import numpy as np


def moments(residual,delta,weights,spatial_dim):
    # delta: seed x trajectory x mode, residual: trajectory x mode
    return (weights[None,:,None]*residual[None,:,:]*delta/spatial_dim,
            weights[None,:,None]*delta**2/spatial_dim)


def bootstrap_moment(blocks,replicate,seed=2026093001):
    rng=np.random.default_rng(np.random.SeedSequence([seed,replicate]))
    # Trajectory-only: retain each fold and average its two probes before sampling rows.
    trajectory=[]
    for block in blocks:
        rows=rng.integers(0,block.shape[1],block.shape[1]);trajectory.append(block.mean(0)[rows].mean(0))
    # Hierarchical sensitivity: resample three fold-model blocks, one entire seed probe
    # per selected block, then held trajectories. Dependence within a probe is preserved.
    hierarchical=[]
    for f in rng.integers(0,len(blocks),len(blocks)):
        block=blocks[f];s=rng.integers(0,block.shape[0]);rows=rng.integers(0,block.shape[1],block.shape[1])
        hierarchical.append(block[s,rows].mean(0))
    return np.mean(trajectory,0),np.mean(hierarchical,0)


def coefficients(n,d,sd,kappa=1.96):
    n=np.asarray(n,dtype=float);d=np.asarray(d,dtype=float);sd=np.asarray(sd,dtype=float)
    if np.any(d<0) or np.any(sd<0) or not np.isfinite([n,d,sd]).all():raise ValueError('invalid moment')
    denominator=d+kappa*sd
    return np.clip(np.divide(n,denominator,out=np.zeros_like(n),where=denominator>1e-25),0,1)


def matched_global(old,new,d):
    denominator=float(np.sum(d*old**2));numerator=float(np.sum(d*new**2))
    factor=np.sqrt(numerator/denominator) if denominator>1e-25 else 0.
    return old*factor,float(factor)
