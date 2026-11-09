import argparse
from pathlib import Path

import numpy as np

from train import ANGLES, features, load_data, logits, rotate


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate a saved model on the official MNIST test set.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    _, _, images, labels = load_data(args.data)
    with np.load(args.weights, allow_pickle=False) as saved:
        params = [saved[key] for key in ("w", "b", "v", "c")]
    for angle in ANGLES:
        batch = images if angle == 0 else rotate(images, np.full(len(images), angle))
        correct = int((logits(features(batch), params).argmax(1) == labels).sum())
        print(f"{angle:+3d} degrees: {correct / len(labels):.2%} ({correct}/{len(labels)})")


if __name__ == "__main__":
    main()
