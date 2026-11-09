"""Independently reload all 55 checkpoints and check their recorded accuracy."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "final_experiment"))
from train import ANGLES, ARMS, SEEDS, load_data, features, rotate, logits

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--work", type=Path, required=True)
args = parser.parse_args()
root = args.work
_, _, images, truth = load_data(root / "mnist.npz")
linear = json.loads((root / "linear.json").read_text(encoding="utf-8"))
limited = json.loads((root / "limited.json").read_text(encoding="utf-8"))
checks = 0
for angle in ANGLES:
    x = features(images if angle == 0 else rotate(images, np.full(len(images), angle)))
    for arm in ARMS:
        for seed in SEEDS:
            path = root / "final_experiment/runs" / f"arm{arm:02}_seed{seed}"
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
                pred = logits(x, [saved[k] for k in ("w", "b", "v", "c")]).argmax(1)
            expected = np.load(path.with_suffix(".pred.npy"), allow_pickle=False)[ANGLES.index(angle)]
            if not np.array_equal(pred, expected):
                raise ValueError(f"Prediction mismatch: {path}, {angle}")
            checks += 1
    for row in linear:
        path = root / f"linear_{row['arm']}_{row['seed']}"
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
            pred = (x @ saved["w"] + saved["b"]).argmax(1)
        expected = np.load(path.with_suffix(".npy"), allow_pickle=False)[ANGLES.index(angle)]
        if not np.array_equal(pred, expected):
            raise ValueError(f"Prediction mismatch: {path}, {angle}")
        checks += 1
    if angle in limited["angles"]:
        for row in limited["rows"]:
            path = root / f"limited_{row['n']}_{row['arm']}_{row['seed']}.npz"
            with np.load(path, allow_pickle=False) as saved:
                pred = logits(x, [saved[k] for k in ("w", "b", "v", "c")]).argmax(1)
            observed = (pred == truth).mean() * 100
            expected = row["accuracy"][limited["angles"].index(angle)]
            if not np.isclose(observed, expected, atol=1e-10, rtol=0):
                raise ValueError(f"Accuracy mismatch: {path}, {angle}")
            checks += 1
print(f"All 55 checkpoints passed {checks} angle-condition checks.")
