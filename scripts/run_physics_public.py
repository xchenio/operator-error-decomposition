"""Physics-informed DeepONet training on the public Burgers data (Nu = 0.01).

No supervision at the target time is used: the loss is the initial condition, the PDE
residual on interior collocation points, and periodicity. See
`docs/PHYSICS_INFORMED_V1.md` for the preregistered protocol, including the measured
diffusion coefficient and the prediction.

Why a separate script
---------------------
The supervised runner maps an initial field to one fixed later field with a one-dimensional
trunk. Here the trunk takes `(t, x)` so that time derivatives exist, which is a different
model shape and a different loss. Reusing the supervised runner's identity would also be
wrong, since its `source_hash` covers different files.

The evaluation cache is the existing Nu=0.01 `t = 1.0` family, so the input trajectory ids,
grid and reference targets are identical to the supervised comparison.

Usage
-----
    python scripts/run_physics_public.py --steps 1500 --device cuda
    python scripts/run_physics_public.py --steps 20 --out results/physics_smoke_v1  # timing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import DeepONet  # noqa: E402
from src.public_benchmark import atomic_json, file_hash  # noqa: E402
from src.resume import atomic_save, random_state, restore_random  # noqa: E402
from src.utils import env_fingerprint, set_seed  # noqa: E402

DATA = "external_data/pdebench/burgers_nu001/fixed_t100_v1"
NU = 3.083e-3          # preregistered pooled least-squares fit, see docs/PHYSICS_INFORMED_V1.md
TIME_MAX = 1.0         # default target time; override with --time-max to match the cache


def source_hash() -> str:
    files = [Path(__file__), ROOT / "src/models.py", ROOT / "src/derivatives.py",
             ROOT / "src/resume.py", ROOT / "src/public_benchmark.py",
             ROOT / "src/utils.py"]
    h = hashlib.sha256()
    for path in files:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def metrics(pred, true):
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    residual = pred - true
    return dict(mean_relative_l2=float(np.mean(
                    np.linalg.norm(residual, axis=1)
                    / np.maximum(np.linalg.norm(true, axis=1), 1e-12))),
                relative_fro=float(np.linalg.norm(residual) / max(np.linalg.norm(true), 1e-12)),
                rmse=float(np.sqrt(np.mean(residual ** 2))))


def train_one(cfg, arrays, protocol, folder, device):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    identity = dict(configuration=cfg, source_sha256=source_hash(),
                    data_protocol=dict(cache_sha256=protocol["cache_sha256"]),
                    environment=env_fingerprint(device))
    result_path = folder / "result.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous["identity"] != identity:
            raise ValueError(f"completed run identity mismatch: {folder}")
        print("[skip completed]", folder.name, flush=True)
        return previous
    manifest = folder / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["identity"] != identity:
        raise ValueError(f"run identity mismatch: {folder}")
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip()
    atomic_json(manifest, dict(status="running", identity=identity, git_commit=git_commit))

    def event(kind, **kw):
        with (folder / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(time_unix=time.time(), event=kind, **kw)) + "\n")
            f.flush()

    n = protocol["normalization"]
    grid = arrays["x"]                                  # (256,) cell-centred in [0, 1)
    u0_all = torch.tensor(arrays["train_x"], device=device, dtype=torch.float32)
    target = arrays["train_y"]                          # raw reference at T, for IC scaling
    xmean, xstd = n["input_mean"], max(n["input_std"], 1e-6)
    ymean, ystd = n["target_mean"], max(n["target_std"], 1e-6)

    model = DeepONet(branch_in=int(arrays["train_x"].shape[1]), trunk_in=2, width=64,
                     branch_hidden=128, trunk_hidden=128, branch_layers=3, trunk_layers=3,
                     trunk_type="fourier" if cfg["architecture"] == "deeponet_fourier" else "mlp"
                     ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(cfg["seed"] + 1000)
    cp = folder / "checkpoints/latest.pt"
    step, train_seconds, history = 0, 0.0, []
    if cp.exists():
        saved = torch.load(cp, map_location=device, weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("checkpoint identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_random(saved["random"], rng)
        step, train_seconds, history = saved["step"], saved["train_seconds"], saved["history"]
        event("resumed", step=step)
    else:
        event("started", step=0)

    @torch.no_grad()
    def predict_at(split, t_value):
        model.eval()
        u0 = torch.tensor((arrays[split + "_x"] - xmean) / xstd, device=device)
        coords = torch.tensor(np.column_stack([np.full(len(grid), t_value), grid]),
                              device=device, dtype=torch.float32).unsqueeze(0).expand(
                                  u0.shape[0], -1, -1)
        out = []
        for i in range(0, u0.shape[0], 32):
            out.append(model.predict_on_grid(u0[i:i + 32],
                                             torch.tensor(
                                                 np.column_stack(
                                                     [np.full(len(grid), t_value), grid]),
                                                 device=device, dtype=torch.float32)))
        model.train()
        return torch.cat(out).cpu().numpy() * ystd + ymean

    def save():
        atomic_save(dict(identity=identity, model=model.state_dict(),
                         optimizer=optimizer.state_dict(), random=random_state(rng),
                         step=step, history=history, train_seconds=train_seconds), cp)
        atomic_json(folder / "progress.json",
                    dict(status="running", step=step, total_steps=cfg["steps"],
                         train_seconds=train_seconds))

    n_ic = len(grid)
    while step < cfg["steps"]:
        started = time.perf_counter()
        idx = rng.choice(len(u0_all), size=cfg["batch_size"], replace=False)
        u0 = (u0_all[idx] - xmean) / xstd                      # (B, m) normalized input
        B = u0.shape[0]
        ev = model.evaluator(u0, order=2)

        # initial condition at t = 0 on the sensor grid
        c_ic = np.column_stack([np.zeros(n_ic), grid])
        # collocation in the interior
        t_col = rng.uniform(1e-4, cfg["time_max"], size=cfg["collocation"])
        x_col = rng.uniform(0.0, 1.0, size=cfg["collocation"])
        c_col = np.column_stack([t_col, x_col])
        # periodicity: same time, both endpoints
        t_bc = rng.uniform(0.0, cfg["time_max"], size=cfg["boundary_points"])
        c_bc = np.column_stack([np.repeat(t_bc, 2),
                                np.tile(np.array([0.0, 1.0]), cfg["boundary_points"])])

        coords = torch.tensor(np.vstack([c_ic, c_col, c_bc]), device=device,
                              dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
        u, du, d2u = ev(coords)
        u = u.squeeze() if u.shape[0] == 1 else u
        u_ic = u[:, :n_ic]
        u_col = u[:, n_ic:n_ic + cfg["collocation"]]
        u_bc = u[:, n_ic + cfg["collocation"]:]
        u_t = du[:, n_ic:n_ic + cfg["collocation"], 0]
        u_x = du[:, n_ic:n_ic + cfg["collocation"], 1]
        u_xx = d2u[:, n_ic:n_ic + cfg["collocation"], 1, 1]

        # work in physical units for the residual
        u_phys = u_col * ystd + ymean
        u_t_phys = u_t * ystd
        u_x_phys = u_x * ystd
        u_xx_phys = u_xx * ystd
        if cfg["equation"] == "advection":
            # linear advection: u_t + beta u_x = 0, beta taken from the file attributes
            residual = u_t_phys + cfg["beta"] * u_x_phys
        else:
            # Burgers: u_t + u u_x - nu u_xx = 0, nu measured from the data first
            residual = u_t_phys + u_phys * u_x_phys - cfg["nu"] * u_xx_phys

        loss_ic = ((u_ic * ystd + ymean - (u0 * xstd + xmean)) ** 2).mean()
        loss_pde = (residual ** 2).mean()
        periodic = u_bc.reshape(B, cfg["boundary_points"], 2)
        loss_bc = ((periodic[:, :, 0] - periodic[:, :, 1]) ** 2).mean()
        loss = loss_ic + loss_pde + loss_bc

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds += time.perf_counter() - started
        step += 1

        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            pred = predict_at("val", cfg["time_max"])
            val = metrics(pred, arrays["val_y"])
            row = dict(step=step, loss=float(loss.detach()),
                       loss_ic=float(loss_ic.detach()), loss_pde=float(loss_pde.detach()),
                       loss_bc=float(loss_bc.detach()), validation=val,
                       train_seconds=train_seconds)
            history.append(row)
            event("evaluation", **row)
            print(folder.name, step, "val=", round(val["mean_relative_l2"], 5),
                  "pde=", f"{float(loss_pde.detach()):.3e}",
                  "s=", round(train_seconds, 1), flush=True)
        if step % cfg["checkpoint_every"] == 0 or step == cfg["steps"]:
            save()

    scores = {}
    for split in ("train", "val", "test"):
        scores[split] = metrics(predict_at(split, cfg["time_max"]), arrays[split + "_y"])
    parameters = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
    result = dict(identity=identity, git_commit=git_commit, metrics=scores, history=history,
                  train_seconds=train_seconds, real_parameter_count=parameters,
                  selection="last fixed-budget checkpoint",
                  note="no supervision at the target time; initial condition, PDE residual "
                       "and periodicity only")
    atomic_json(result_path, result)
    atomic_json(manifest, dict(status="completed", identity=identity, git_commit=git_commit))
    atomic_json(folder / "progress.json",
                dict(status="completed", step=step, train_seconds=train_seconds))
    event("completed", step=step, test=scores["test"])
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--out", default="results/physics_nu001_t100_v1")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--architectures", default="deeponet_mlp,deeponet_fourier")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--collocation", type=int, default=256)
    ap.add_argument("--boundary-points", type=int, default=32)
    ap.add_argument("--nu", type=float, default=NU)
    ap.add_argument("--equation", default="burgers", choices=("burgers", "advection"),
                    help="advection uses u_t + beta u_x = 0 with beta from the file attributes")
    ap.add_argument("--beta", type=float, default=None,
                    help="required for --equation advection; normally read from the cache")
    ap.add_argument("--time-max", type=float, default=TIME_MAX,
                    help="target time; has to match the evaluation cache")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data_dir = ROOT / args.data_dir
    protocol = json.loads((data_dir / "protocol.json").read_text(encoding="utf-8"))
    if file_hash(data_dir / "pairs.npz") != protocol["cache_sha256"]:
        raise ValueError("pair cache changed")
    with np.load(data_dir / "pairs.npz") as z:
        arrays = {k: z[k].copy() for k in z.files}
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "protocol.json",
                dict(protocol, physics=dict(nu=args.nu, time_max=args.time_max,
                                            losses="initial condition + PDE residual + "
                                                   "periodicity, all weight 1")))
    beta = args.beta
    if args.equation == "advection":
        if beta is None:
            attrs = protocol.get("file_attrs") or {}
            if "beta" not in attrs:
                raise ValueError("advection needs --beta; the cache declares no file_attrs")
            beta = float(attrs["beta"])
        print(f"advection with beta={beta} from the cache", flush=True)
    else:
        beta = 0.0

    device = torch.device(args.device)
    for architecture in args.architectures.split(","):
        if architecture not in ("deeponet_mlp", "deeponet_fourier"):
            raise ValueError(architecture)
        for seed in (int(s) for s in args.seeds.split(",")):
            cfg = dict(architecture=architecture, seed=seed, steps=args.steps, lr=1e-3,
                       batch_size=args.batch_size, trunk_in=2,
                       collocation=args.collocation, boundary_points=args.boundary_points,
                       nu=args.nu, beta=beta, equation=args.equation, time_max=args.time_max,
                       checkpoint_every=50, eval_every=250)
            train_one(cfg, arrays, protocol, out / f"{architecture}_s{seed}", device)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
