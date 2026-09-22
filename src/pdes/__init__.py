from .base import PDEProblem
from .burgers import BurgersProblem
from .burgers2d import Burgers2DProblem
from .diffusion_reaction import DiffusionReactionProblem
from .darcy import DarcyProblem

PROBLEMS = {
    "burgers": BurgersProblem,
    "burgers2d": Burgers2DProblem,
    "dr": DiffusionReactionProblem,
    "darcy": DarcyProblem,
}


def make_problem(name: str, **kwargs) -> PDEProblem:
    if name not in PROBLEMS:
        raise KeyError(f"unknown problem {name}, choose from {list(PROBLEMS)}")
    return PROBLEMS[name](**kwargs)
