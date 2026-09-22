"""通用工具：随机种子、设备、指标、日志、计时。"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def rel_l2(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> float:
    """逐样本相对 L2，再取均值。pred/true: (B, N)。"""
    num = torch.linalg.norm(pred - true, dim=1)
    den = torch.linalg.norm(true, dim=1)
    return float((num / den.clamp_min(eps)).mean().item())


def code_fingerprint(root, globs=("src/**/*.py", "scripts/**/*.py")) -> str:
    """源码指纹：对 src/ 与 scripts/ 下所有 .py 的相对路径 + 内容做 sha256。

    动机（见论文 §5.9）：我们发现两个同样配置的 run 结果差 0.0124，
    排查后确认**不是**数值非确定性（重跑逐位一致），而是两次运行之间代码被改过，
    而没有任何记录说明哪个结果对应哪份代码。每次 run 记下源码指纹即可根治。
    """
    import hashlib
    root = Path(root)
    h = hashlib.sha256()
    for g in globs:
        for p in sorted(root.glob(g)):
            h.update(str(p.relative_to(root)).encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def env_fingerprint(device) -> dict:
    """运行环境指纹：库版本 + GPU 型号。"""
    import platform
    out = {"python": platform.python_version(), "torch": torch.__version__,
           "numpy": np.__version__, "device": str(device)}
    if torch.cuda.is_available():
        try:
            out["gpu"] = torch.cuda.get_device_name(0)
            out["cuda"] = torch.version.cuda
        except Exception:                                # noqa: BLE001
            pass
    return out


def ensure_dir(p) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


class Logger:
    def __init__(self, path=None, verbose: bool = True):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(self.path, "a", encoding="utf-8")
        else:
            self.f = None
        self.verbose = verbose

    def log(self, msg) -> None:
        msg = str(msg)
        if self.verbose:
            print(msg, flush=True)
        if self.f:
            self.f.write(msg + "\n")
            self.f.flush()

    def close(self) -> None:
        if self.f:
            self.f.close()


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def lap(self) -> float:
        return time.time() - self.t0


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return float("nan")
    rx = _rank(x)
    ry = _rank(y)
    return pearson(rx, ry)


def _rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    # 平均并列名次
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    return sums[inv] / cnt[inv]


def fmt_float(x, n: int = 4) -> str:
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)
