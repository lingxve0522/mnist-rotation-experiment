import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from PIL import Image

SEEDS = (123, 124, 125, 126, 127)
ARMS = (0, 5, 15, 30)
ANGLES = (-45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45)
EPOCHS = 15
DATA_SHA = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"


def load_data(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError("Unexpected MNIST archive SHA-256")
    with np.load(path) as archive:
        return tuple(archive[name] for name in ("x_train", "y_train", "x_test", "y_test"))


def split_labels(labels):
    rng = np.random.default_rng(2026)
    counts = np.bincount(labels)
    quota = 5000 * counts / len(labels)
    sizes = quota.astype(int)
    sizes[np.argsort(-(quota - sizes), kind="stable")[:5000 - sizes.sum()]] += 1
    train, val = [], []
    for digit, size in enumerate(sizes):
        indices = rng.permutation(np.flatnonzero(labels == digit))
        val.extend(indices[:size])
        train.extend(indices[size:])
    return np.array(train), np.array(val)


def rotate(images, angles):
    result = np.empty_like(images)
    for i, (image, angle) in enumerate(zip(images, angles, strict=True)):
        result[i] = np.asarray(Image.fromarray(image).rotate(
            float(angle), resample=Image.Resampling.BILINEAR, fillcolor=0
        ))
    return result


def features(images):
    return images.reshape(-1, 784).astype(np.float32) / 255


def logits(x, params):
    w, b, v, c = params
    return np.maximum(x @ w + b, 0) @ v + c


def fit(images, labels, train, val, seed, arm):
    init = np.random.default_rng(seed)
    order = np.random.default_rng(seed + 1000)
    augment = np.random.default_rng(seed + 2000)
    params = [
        init.standard_normal((784, 128), dtype=np.float32) * np.float32(np.sqrt(2 / 784)),
        np.zeros(128, dtype=np.float32),
        init.standard_normal((128, 10), dtype=np.float32) * np.float32(np.sqrt(2 / 128)),
        np.zeros(10, dtype=np.float32),
    ]
    velocity = [np.zeros_like(param) for param in params]
    w, b, v, c = params
    x_val = features(images[val])
    x_clean = features(images) if arm == 0 else None
    history = []
    start = time.perf_counter()

    for epoch in range(EPOCHS):
        lr = 0.005 + 0.045 * (1 + np.cos(np.pi * epoch / (EPOCHS - 1))) / 2
        shuffled = order.permutation(train)
        total_loss = 0.0
        for begin in range(0, len(shuffled), 256):
            ids = shuffled[begin:begin + 256]
            if arm:
                x = features(rotate(images[ids], augment.uniform(-arm, arm, len(ids))))
            else:
                x = x_clean[ids]
            y = labels[ids]
            h = np.maximum(x @ w + b, 0)
            z = h @ v + c
            z -= z.max(axis=1, keepdims=True)
            exp = np.exp(z)
            normalizer = exp.sum(axis=1, keepdims=True)
            total_loss += float((np.log(normalizer[:, 0]) - z[np.arange(len(y)), y]).sum())
            dz = exp / normalizer
            dz[np.arange(len(y)), y] -= 1
            dz /= len(y)
            dh = (dz @ v.T) * (h > 0)
            gradients = (x.T @ dh, dh.sum(axis=0), h.T @ dz, dz.sum(axis=0))
            for param, speed, gradient in zip(params, velocity, gradients, strict=True):
                speed *= 0.9
                speed += gradient
                param -= lr * speed
        accuracy = float((logits(x_val, params).argmax(1) == labels[val]).mean())
        history.append(dict(epoch=epoch + 1, loss=total_loss / len(train), validation=accuracy))
        print(f"seed={seed} arm={arm} epoch={epoch + 1} val={accuracy:.4f}", flush=True)
    return params, history, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description="Train paired rotation-ablation models on official MNIST.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--arms", type=int, nargs="+", choices=ARMS, default=ARMS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    images, labels, test, target = load_data(args.data)
    train, val = split_labels(labels)
    tests = [features(test if a == 0 else rotate(test, np.full(len(test), a))) for a in ANGLES]
    for arm in args.arms:
        path = args.out / f"arm{arm:02}_seed{args.seed}"
        params, history, seconds = fit(images, labels, train, val, args.seed, arm)
        np.savez_compressed(path.with_suffix(".npz"), w=params[0], b=params[1], v=params[2], c=params[3])
        with np.load(path.with_suffix(".npz")) as saved:
            restored = [saved[key] for key in ("w", "b", "v", "c")]
        predictions = np.array([logits(x, restored).argmax(1) for x in tests], dtype=np.uint8)
        np.save(path.with_suffix(".pred.npy"), predictions, allow_pickle=False)
        record = dict(seed=args.seed, arm=arm, angles=ANGLES, history=history,
                      correct=(predictions == target).sum(axis=1).tolist(), seconds=seconds,
                      train_count=len(train), validation_count=len(val), test_count=len(target))
        path.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
