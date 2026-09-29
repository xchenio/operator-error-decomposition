"""Reproducible synthetic Allen-Cahn pairs; validation IDs never overlap data IDs."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.public_benchmark import atomic_json,file_hash
PARAMS=dict(n=256,dt=.001,final_time=.5,nu=.01,rho=5.,length=2.,seed=2026092904,modes=12)


def initial(ids,n=256):
    grid=np.arange(n)*2/n
    out=[]
    for i in ids:
        rng=np.random.default_rng(np.random.SeedSequence([PARAMS['seed'],int(i)]))
        k=np.arange(1,13);a=rng.normal(size=12)/k**1.5;b=rng.normal(size=12)/k**1.5
        a*=.35;b*=.35
        out.append(rng.uniform(-.4,.4)+np.cos(np.pi*grid[:,None]*k)@a+np.sin(np.pi*grid[:,None]*k)@b)
    return np.array(out)


def reaction(u,h,rho=5.):
    return u*np.exp(rho*h)/np.sqrt(1+u*u*np.expm1(2*rho*h))


def solve(u,dt=.001,final_time=.5,nu=.01,rho=5.,length=2.):
    u=np.asarray(u,dtype=np.float64).copy();steps=int(round(final_time/dt))
    if abs(steps*dt-final_time)>1e-12:raise ValueError('nonintegral time steps')
    k=2*np.pi*np.fft.rfftfreq(u.shape[1],d=length/u.shape[1]);factor=np.exp(-nu*k*k*dt/2)
    for _ in range(steps):
        u=np.fft.irfft(np.fft.rfft(u,axis=1)*factor,axis=1,n=u.shape[1])
        u=reaction(u,dt,rho)
        u=np.fft.irfft(np.fft.rfft(u,axis=1)*factor,axis=1,n=u.shape[1])
    if not np.isfinite(u).all():raise ValueError('solver nonfinite')
    return u


def qa():
    ids=list(range(100000,100016));x=initial(ids);coarse=solve(x,dt=.002);y=solve(x);fine=solve(x,dt=.0005)
    higher=solve(initial(ids,512),dt=.0005)[:,::2]
    norm=lambda a:np.linalg.norm(a,axis=1)
    t1=norm(coarse-fine)/norm(fine);t2=norm(y-fine)/norm(fine);space=norm(fine-higher)/norm(higher)
    constant=np.full((1,256),.25);analytic=reaction(constant,.5)
    zero=np.zeros((1,256));one=np.ones((1,256));minus=-one
    diffusion=solve(x,rho=0)
    k=2*np.pi*np.fft.rfftfreq(256,d=2/256)
    exact_diffusion=np.fft.irfft(np.fft.rfft(x,axis=1)*np.exp(-.01*k*k*.5),axis=1,n=256)
    # Independent method-of-lines integration for one held solver-QA sample.
    from scipy.integrate import solve_ivp
    rhs=lambda t,v:.01*np.fft.irfft(-k*k*np.fft.rfft(v),n=256)+5*v*(1-v*v)
    reference=solve_ivp(rhs,(0,.5),x[0],method='DOP853',rtol=1e-10,atol=1e-12).y[:,-1]
    result=dict(parameters=PARAMS,source_sha256=file_hash(Path(__file__)),qa_ids=ids,
        time_max_relative=float(t2.max()),coarse_max_relative=float(t1.max()),
        refinement_ratio_min=float((t1/np.maximum(t2,1e-20)).min()),space_max_relative=float(space.max()),
        constant_max_error=float(np.abs(solve(constant)-analytic).max()),
        equilibria_max_error=max(float(np.abs(solve(v)-v).max()) for v in (zero,one,minus)),
        diffusion_max_error=float(np.abs(diffusion-exact_diffusion).max()),
        independent_ode_relative=float(np.linalg.norm(y[0]-reference)/np.linalg.norm(reference)),
        nonlinear_departure=float(np.mean(norm(y-diffusion)/norm(y))))
    result['passed']=bool(result['time_max_relative']<1e-4 and result['space_max_relative']<1e-5 and
        result['refinement_ratio_min']>3 and result['constant_max_error']<1e-10 and
        result['equilibria_max_error']<1e-10 and result['diffusion_max_error']<1e-10 and result['independent_ode_relative']<1e-4)
    atomic_json(ROOT/'results/public_field_modal_v1/solver_qa.json',result)
    print(json.dumps(result,indent=2))
    if not result['passed']:raise ValueError('Allen-Cahn solver validation failed')
    return result


if __name__=='__main__':qa()
