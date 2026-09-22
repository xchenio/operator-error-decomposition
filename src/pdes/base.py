"""参数化 PDE 问题的统一接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PDEProblem(ABC):
    """子类需要设置的属性：
        n_coef        输入函数的参数个数
        n_sensors     branch 输入维度
        sensor_coords branch 输入所对应的采样位置
        grid          (Ng, trunk_dim) 评估网格（解的存储/测试位置）
        ranges        trunk 每个维度的 (lo, hi)，用于构造解析先验基
        prior_spec    trunk 每个维度的 (kind, K)
    """

    name = "base"
    trunk_dim = 2

    # 因果加权所需：时间维下标与时间区间；稳态问题置 None
    time_dim: int | None = None
    t_range: tuple = (0.0, 1.0)

    # ---------------- 数据生成 ----------------
    @abstractmethod
    def sample_coeffs(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """(n, n_coef)"""

    @abstractmethod
    def branch_input(self, coef: np.ndarray) -> np.ndarray:
        """(n, n_sensors)"""

    @abstractmethod
    def solve(self, coef: np.ndarray) -> np.ndarray:
        """(n, Ng)，与 self.grid 对应"""

    # ---------------- 训练采样与损失 ----------------
    @abstractmethod
    def sample_points(self, coef: np.ndarray, n_col: int, n_ic: int,
                      n_bc: int, rng: np.random.Generator) -> dict:
        """返回 dict，内容由子类自定义，由 losses() 解释。"""

    @abstractmethod
    def losses(self, evalr, pts: dict) -> dict:
        """返回 {损失名: 标量 tensor}"""

    # ---------------- 可选 ----------------
    def input_at(self, coef: np.ndarray, x: np.ndarray) -> np.ndarray:
        """输入函数在任意坐标上的取值，(n, *x.shape)。"""
        raise NotImplementedError

    def describe(self) -> str:
        return (f"{self.name}: trunk_dim={self.trunk_dim}, sensors={self.n_sensors}, "
                f"grid={self.grid.shape[0]}, coef={self.n_coef}")
