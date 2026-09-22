"""Atomic local checkpoints with all random generators required for exact continuation."""
import os
import random
from pathlib import Path
import numpy as np
import torch


def random_state(rng):
    return dict(python=random.getstate(),numpy=np.random.get_state(),
                generator=rng.bit_generator.state,torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_random(state,rng):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    rng.bit_generator.state=state['generator']
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_save(state,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as f:
        torch.save(state,f)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)
