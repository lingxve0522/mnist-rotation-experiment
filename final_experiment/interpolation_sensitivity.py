"""Evaluate the trained baseline and ±15-degree models with three interpolators."""
import argparse
import json
from pathlib import Path

from train import ANGLES, SEEDS, features, load_data, logits
import numpy as np
from PIL import Image


METHODS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
}
TEST_ANGLES = (-15, 15)
TEST_ARMS = (0, 15)


def load_checkpoints(runs):
    checkpoints = {}
    for arm in TEST_ARMS:
        for seed in SEEDS:
            stem = runs / f"arm{arm:02}_seed{seed}"
            with np.load(stem.with_suffix(".npz"), allow_pickle=False) as checkpoint:
                params = [checkpoint[name] for name in ("w", "b", "v", "c")]
            predictions = np.load(stem.with_suffix(".pred.npy"), allow_pickle=False)
            record = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
            if (record["arm"], record["seed"], tuple(record["angles"])) != (arm, seed, ANGLES):
                raise ValueError(f"Unexpected run metadata in {stem}")
            checkpoints[arm, seed] = (params, predictions)
    return checkpoints


def summarize(values):
    values = np.asarray(values)
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "by_seed": values.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _, _, images, labels = load_data(args.data)
    checkpoints = load_checkpoints(args.runs)
    methods = {}
    for name, resampling in METHODS.items():
        scores = {key: [] for key in checkpoints}
        for angle in TEST_ANGLES:
            rotated = np.stack([
                np.asarray(Image.fromarray(image).rotate(angle, resample=resampling, fillcolor=0))
                for image in images
            ])
            x = features(rotated)
            for key, (params, stored) in checkpoints.items():
                predictions = logits(x, params).argmax(axis=1)
                if name == "bilinear" and not np.array_equal(predictions, stored[ANGLES.index(angle)]):
                    raise ValueError(f"Bilinear predictions differ for {key} at {angle} degrees")
                scores[key].append(100 * np.mean(predictions == labels))
        baseline = np.array([scores[0, seed] for seed in SEEDS]).mean(axis=1)
        augmented = np.array([scores[15, seed] for seed in SEEDS]).mean(axis=1)
        methods[name] = {
            "baseline_percent": summarize(baseline),
            "rotation15_percent": summarize(augmented),
            "paired_difference_pp": summarize(augmented - baseline),
        }
    summary = dict(angles=TEST_ANGLES, seeds=SEEDS, methods=methods)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
