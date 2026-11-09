"""Unified entry point for the MNIST course experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE = Path(__file__).resolve().parent


def execute(script, *args, env=None):
    command = [sys.executable, str(BASE / script), *map(str, args)]
    print("Running", script, flush=True)
    subprocess.run(command, check=True, cwd=BASE, env=env)


def verify():
    manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = BASE / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Missing or modified file: {name}")
    print(f"Verified {len(manifest)} bundled files.")
    execute("final_experiment/summarize.py", "--data", BASE / "mnist.npz",
            "--runs", BASE / "final_experiment/runs", "--out", BASE / "verified_summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "evaluate", "retrain"))
    parser.add_argument("--out", type=Path, help="New output folder for evaluate/retrain")
    args = parser.parse_args()
    if args.mode == "verify":
        verify()
        return
    target = (args.out or BASE / f"reproduced_{args.mode}").resolve()
    if target == BASE or BASE.is_relative_to(target):
        raise ValueError("Output must be a separate new folder.")
    target.mkdir(parents=True, exist_ok=False)
    (target / "final_experiment").mkdir()
    if args.mode == "evaluate":
        shutil.copytree(BASE / "final_experiment/runs", target / "final_experiment/runs")
        for pattern in ("linear_*.npz", "linear_*.npy", "limited_*.npz"):
            for path in BASE.glob(pattern):
                shutil.copy2(path, target / path.name)
        for name in ("linear.json", "limited.json"):
            shutil.copy2(BASE / name, target / name)
        execute("verify_inference.py", "--work", BASE)
    else:
        for seed in (123, 124, 125, 126, 127):
            execute("final_experiment/train.py", "--data", BASE / "mnist.npz",
                    "--out", target / "final_experiment/runs", "--seed", seed)
    env = os.environ.copy()
    env["MNIST_OUTPUT"] = str(target)
    if args.mode == "retrain":
        execute("linear.py", env=env)
        execute("limited.py", env=env)
    execute("final_experiment/summarize.py", "--data", BASE / "mnist.npz",
            "--runs", target / "final_experiment/runs", "--out", target / "final_experiment/summary.json")
    execute("stress_fast.py", env=env)
    execute("diagnostics.py", "--work", target, "--data", BASE / "mnist.npz",
            "--out", target / "diagnostics.json")
    execute("final_experiment/interpolation_sensitivity.py", "--data", BASE / "mnist.npz",
            "--runs", target / "final_experiment/runs", "--out", target / "interpolation_sensitivity.json")
    execute("final_experiment/make_figures.py", "--data", BASE / "mnist.npz",
            "--runs", target / "final_experiment/runs", "--out", target / "figures")
    execute("new_summary.py", env=env)
    print("Complete. Results saved to", target)


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    main()
