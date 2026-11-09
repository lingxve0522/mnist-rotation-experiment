import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from train import ANGLES, ARMS, SEEDS, EPOCHS, load_data, split_labels


def summarize(values):
    return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)), values=values.tolist())


def main():
    parser = argparse.ArgumentParser(description="Verify saved runs and compute paired summaries.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    images, labels, test, target = load_data(args.data)
    assert images.shape == (60000, 28, 28) and test.shape == (10000, 28, 28)
    train, val = split_labels(labels)
    assert len(train) == 55000 and len(val) == 5000
    assert np.array_equal(np.sort(np.concatenate([train, val])), np.arange(len(labels)))

    correctness = {}
    histories = {}
    elapsed = []
    for arm in ARMS:
        correctness[arm] = []
        histories[arm] = []
        for seed in SEEDS:
            path = args.runs / f"arm{arm:02}_seed{seed}"
            record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            assert (record["seed"], record["arm"], tuple(record["angles"])) == (seed, arm, ANGLES)
            assert (record["train_count"], record["validation_count"], record["test_count"]) == (
                len(train), len(val), len(target)
            )
            assert [item["epoch"] for item in record["history"]] == list(range(1, EPOCHS + 1))
            with np.load(path.with_suffix(".npz")) as weights:
                for name, shape in zip(("w", "b", "v", "c"), ((784,128),(128,),(128,10),(10,)), strict=True):
                    assert weights[name].shape == shape and np.isfinite(weights[name]).all()
            predictions = np.load(path.with_suffix(".pred.npy"), allow_pickle=False)
            assert predictions.shape == (len(ANGLES), len(target)) and predictions.max() < 10
            correct = predictions == target
            assert np.array_equal(correct.sum(axis=1), record["correct"])
            correctness[arm].append(correct)
            histories[arm].append(record["history"])
            elapsed.append(record["seconds"])
        correctness[arm] = np.stack(correctness[arm])

    endpoints = {
        "clean": [0],
        "plus_minus_15": [-15, 15],
        "all_nonzero": [a for a in ANGLES if a != 0],
        "plus_minus_45": [-45, 45],
    }
    stats = {}
    for arm, correct in correctness.items():
        stats[arm] = dict(
            by_angle={angle: summarize(correct[:, i, :].mean(axis=1) * 100)
                      for i, angle in enumerate(ANGLES)},
            endpoints={name: summarize(correct[:, [ANGLES.index(a) for a in angles], :].mean(axis=(1, 2)) * 100)
                       for name, angles in endpoints.items()},
            last_validation=summarize(np.array([h[-1]["validation"] for h in histories[arm]]) * 100),
        )
    primary = {}
    baseline = correctness[0]
    augmented = correctness[15]
    for name, angles in endpoints.items():
        ids = [ANGLES.index(a) for a in angles]
        paired = augmented[:, ids, :].astype(np.int8) - baseline[:, ids, :]
        delta = paired.mean(axis=(1, 2)) * 100
        info = summarize(delta)
        margin = 2.776445 * info["sd"] / np.sqrt(len(SEEDS))
        info["seed_t_ci95"] = [info["mean"] - margin, info["mean"] + margin]
        primary[name] = info
        if name == "plus_minus_15":
            # Resample image IDs, keeping both rotation directions and all seeds together.
            image_delta = paired.mean(axis=(0, 1)) * 100
            rng = np.random.default_rng(20261103)
            bootstrap = np.array([image_delta[rng.integers(len(target), size=len(target))].mean()
                                  for _ in range(5000)])
            info["fixed_models_image_bootstrap_ci95"] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
            signs = np.array(list(itertools.product((-1, 1), repeat=len(SEEDS))))
            info["two_sided_sign_flip_p"] = float((np.abs((signs * delta).mean(axis=1))
                                                   >= abs(delta.mean()) - 1e-10).mean())
    class_delta = {}
    ids = [ANGLES.index(a) for a in (-15, 15)]
    for digit in range(10):
        mask = target == digit
        class_delta[digit] = {
            "test_count": int(mask.sum()),
            "baseline": float(baseline[:, ids, :][:, :, mask].mean() * 100),
            "rotation15": float(augmented[:, ids, :][:, :, mask].mean() * 100),
        }
    out = dict(arms=stats, primary_comparison=primary, per_class_plus_minus_15=class_delta,
               train_seconds_total=sum(elapsed), train_seconds_per_run=elapsed,
               train_count=len(train), validation_count=len(val), test_count=len(target))
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
