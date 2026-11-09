"""Render reproducible, print-resolution figures from saved MNIST predictions."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from train import ANGLES, ARMS, SEEDS, load_data, rotate

INK = "#14263A"
MUTED = "#536679"
GRID = "#DDE5EC"
BASELINE = "#4B5B6C"
IMPROVED = "#1765AA"
COLORS = {0: BASELINE, 5: "#A85812", 15: IMPROVED, 30: "#00816D"}
LABELS = {0: "Baseline", 5: "±5° train", 15: "±15° train", 30: "±30° train"}
WHITE = "#FFFFFF"
FONT_CANDIDATES = [
    (Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
     Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arialbd.ttf"),
    (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
     Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
]


def font(size, bold=False):
    for regular, heavy in FONT_CANDIDATES:
        path = heavy if bold else regular
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


class Figure:
    def __init__(self, width, height):
        self.image = Image.new("RGB", (width, height), WHITE)
        self.draw = ImageDraw.Draw(self.image, "RGBA")

    def text(self, x, y, value, size=38, anchor="lm", bold=False, color=INK):
        self.draw.text((x, y), value, font=font(size, bold), fill=color, anchor=anchor)

    def save(self, path):
        self.image.save(path, dpi=(300, 300))


class Axes:
    def __init__(self, figure, rect, xs, ys, xticks, yticks, yformat=str):
        self.figure = figure
        self.left, self.top, self.right, self.bottom = rect
        self.xs, self.ys = xs, ys
        for tick in yticks:
            y = self.map(xs[0], tick)[1]
            figure.draw.line((self.left, y, self.right, y), fill=GRID, width=2)
            figure.text(self.left - 20, y, yformat(tick), 35, "rm", color=MUTED)
        for tick in xticks:
            x = self.map(tick, ys[0])[0]
            figure.draw.line((x, self.bottom, x, self.bottom + 10), fill=MUTED, width=2)
            figure.text(x, self.bottom + 31, str(tick), 35, "mm", color=MUTED)
        figure.draw.line((self.left, self.top, self.left, self.bottom, self.right, self.bottom),
                         fill=MUTED, width=3)

    def map(self, x, y):
        lo, hi = self.xs
        bottom, top = self.ys
        return (self.left + (x - lo) / (hi - lo) * (self.right - self.left),
                self.bottom - (y - bottom) / (top - bottom) * (self.bottom - self.top))

    def band(self, x, center, spread, color):
        upper = [self.map(i, j) for i, j in zip(x, center + spread, strict=True)]
        lower = [self.map(i, j) for i, j in zip(x, center - spread, strict=True)]
        rgba = tuple(bytes.fromhex(color[1:])) + (32,)
        self.figure.draw.polygon(upper + lower[::-1], fill=rgba)

    def series(self, x, values, color, marker=True):
        points = [self.map(i, j) for i, j in zip(x, values, strict=True)]
        self.figure.draw.line(points, fill=color, width=6, joint="curve")
        if marker:
            for cx, cy in points:
                self.figure.draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8),
                                         fill=WHITE, outline=color, width=4)


def legend(fig, y):
    for x, arm in zip((235, 610, 985, 1370), ARMS, strict=True):
        fig.draw.line((x, y, x + 62, y), fill=COLORS[arm], width=6)
        fig.draw.ellipse((x + 23, y - 8, x + 39, y + 8), fill=WHITE,
                         outline=COLORS[arm], width=4)
        fig.text(x + 78, y, LABELS[arm], 37)


def summaries(data):
    return data.mean(axis=0), data.std(axis=0, ddof=1)


def plot_angle(runs, destination):
    fig = Figure(1960, 1020)
    legend(fig, 940)
    data = {arm: np.array([run["correct"] for run in runs[arm]]) * 100 /
                 runs[arm][0]["test_count"] for arm in ARMS}
    summary = {arm: summaries(data[arm]) for arm in ARMS}
    ylo = max(0, 10 * math.floor(min((avg - sd).min() for avg, sd in summary.values()) / 10))
    ax = Axes(fig, (145, 96, 1880, 790), (-46, 46), (ylo, 100), ANGLES,
              range(ylo, 101, 10))
    fig.text(145, 52, "Test accuracy (%)", 42, bold=True)
    fig.text(1010, 874, "Test rotation angle (degrees)", 42, "mm")
    zx = ax.map(0, ylo)[0]
    for yy in range(ax.top, ax.bottom, 19):
        fig.draw.line((zx, yy, zx, min(ax.bottom, yy + 10)), fill="#AAB8C5", width=2)
    for arm in ARMS:
        avg, sd = summary[arm]
        ax.band(ANGLES, avg, sd, COLORS[arm])
    for arm in ARMS:
        ax.series(ANGLES, summary[arm][0], COLORS[arm])
    fig.save(destination)


def linear_ticks(low, high, step):
    return np.arange(step * math.floor(low / step), step * math.ceil(high / step) + step / 2, step)


def plot_validation(runs, destination):
    fig = Figure(1960, 1010)
    legend(fig, 928)
    panels = (
        ("validation", "Clean validation accuracy (%)", 100, 1, "{:.0f}"),
        ("loss", "Training cross-entropy", 1, 0.1, "{:.1f}"),
    )
    for left, right, (key, title, multiplier, step, formatter) in zip(
        (156, 1132), (932, 1880), panels, strict=True
    ):
        curves = {}
        for arm in ARMS:
            curves[arm] = summaries(np.array([[h[key] for h in row["history"]]
                                             for row in runs[arm]]) * multiplier)
        low = min((avg - sd).min() for avg, sd in curves.values())
        high = max((avg + sd).max() for avg, sd in curves.values())
        if (high - low) / step > 8:
            step *= 2
        ticks = linear_ticks(low, high, step)
        epochs = np.array([row["epoch"] for row in runs[0][0]["history"]])
        ax = Axes(fig, (left, 102, right, 765), (0.65, epochs[-1] + 0.35),
                  (ticks[0], ticks[-1]), (1, 5, 10, 15), ticks, formatter.format)
        fig.text(left, 54, title, 40, bold=True)
        fig.text((left + right) / 2, 851, "Epoch", 40, "mm")
        for arm in ARMS:
            ax.band(epochs, *curves[arm], COLORS[arm])
        for arm in ARMS:
            ax.series(epochs, curves[arm][0], COLORS[arm])
    fig.save(destination)


def plot_class(predictions, truth, destination):
    selected = [ANGLES.index(-15), ANGLES.index(15)]
    correct_base = predictions[0][:, selected, :] == truth
    correct_aug = predictions[15][:, selected, :] == truth
    gain = np.stack([
        (correct_aug[:, :, truth == digit].mean(axis=(1, 2))
         - correct_base[:, :, truth == digit].mean(axis=(1, 2))) * 100
        for digit in range(10)
    ], axis=1)
    avg, sd = summaries(gain)
    ticks = linear_ticks(min(-1, (avg - sd).min()), max(1, (avg + sd).max()), 1)
    if len(ticks) > 11:
        ticks = linear_ticks(ticks[0], ticks[-1], 2)
    fig = Figure(1960, 860)
    ax = Axes(fig, (150, 98, 1880, 697), (-0.65, 9.65), (ticks[0], ticks[-1]),
              range(10), ticks, "{:.0f}".format)
    fig.text(150, 50, "Accuracy gain at ±15° test rotation (pp)", 42, bold=True)
    fig.text(1010, 797, "True digit", 42, "mm")
    zero = ax.map(0, 0)[1]
    fig.draw.line((ax.left, zero, ax.right, zero), fill=MUTED, width=3)
    for digit in range(10):
        x, middle = ax.map(digit, avg[digit])
        top, bottom = ax.map(digit, avg[digit] + sd[digit])[1], ax.map(digit, avg[digit] - sd[digit])[1]
        fig.draw.rectangle((x - 46, min(middle, zero), x + 46, max(middle, zero)),
                           fill=IMPROVED if avg[digit] >= 0 else COLORS[5])
        fig.draw.line((x, top, x, bottom), fill=INK, width=4)
        fig.draw.line((x - 14, top, x + 14, top), fill=INK, width=4)
        fig.draw.line((x - 14, bottom, x + 14, bottom), fill=INK, width=4)
    fig.save(destination)
    return [
        dict(digit=digit, count=int((truth == digit).sum()),
             gain_pp_by_seed=gain[:, digit].tolist(),
             mean_gain_pp=float(avg[digit]), seed_sd_pp=float(sd[digit]))
        for digit in range(10)
    ]


def distinct_examples(mask, truth, count):
    # Select the earliest qualifying image of each class, then take the earliest four.
    choices = []
    for digit in range(10):
        eligible = np.flatnonzero(mask & (truth == digit))
        if eligible.size:
            choices.append(int(eligible[0]))
    return sorted(choices)[:count]


def confusion_changes(predictions, truth):
    selected = [ANGLES.index(-15), ANGLES.index(15)]
    changes = []
    for arm in (0, 15):
        pred = predictions[arm][:, selected, :]
        key = (10 * truth + pred).ravel()
        matrix = np.bincount(key, minlength=100).reshape(10, 10)
        changes.append(matrix / (len(SEEDS) * len(selected)))
    baseline, augmented = changes
    rows = [
        dict(digit=digit, prediction=prediction,
             baseline_mean_count=float(baseline[digit, prediction]),
             augmented_mean_count=float(augmented[digit, prediction]),
             delta_mean_count=float(augmented[digit, prediction] - baseline[digit, prediction]))
        for digit in range(10) for prediction in range(10) if digit != prediction
    ]
    return sorted(rows, key=lambda row: row["delta_mean_count"])


def plot_examples(images, truth, predictions, destination):
    angle = 15
    at = ANGLES.index(angle)
    baseline, augmented = predictions[0][0, at], predictions[15][0, at]
    categories = (
        ("Corrected by augmentation", (baseline != truth) & (augmented == truth), IMPROVED),
        ("New errors after augmentation", (baseline == truth) & (augmented != truth), COLORS[5]),
    )
    fig = Figure(1960, 1120)
    records = []
    for row, (title, mask, accent) in enumerate(categories):
        y = 28 + row * 557
        fig.text(44, y + 37, f"{title}  ·  {int(mask.sum())} images", 42, bold=True, color=accent)
        indices = distinct_examples(mask, truth, 4)
        rotated = rotate(images[indices], np.full(len(indices), angle))
        for col, (idx, rimg) in enumerate(zip(indices, rotated, strict=True)):
            x = 44 + col * 475
            rect = (x, y + 86, x + 446, y + 506)
            fig.draw.rounded_rectangle(rect, radius=14, fill="#F3F6F9", outline=GRID, width=2)
            fig.text(x + 118, y + 131, "Original", 34, "mm", color=MUTED)
            fig.text(x + 326, y + 131, "+15°", 34, "mm", color=MUTED)
            for offset, image in ((31, images[idx]), (239, rimg)):
                tile = Image.fromarray(image).resize((174, 174), Image.Resampling.NEAREST)
                fig.image.paste(tile.convert("RGB"), (x + offset, y + 157))
            fig.text(x + 27, y + 376, f"True {truth[idx]}   ·   Test #{idx}", 35, bold=True)
            fig.text(x + 27, y + 445, f"Prediction: {baseline[idx]} → {augmented[idx]}", 36)
            records.append(dict(category=title, index=idx, digit=int(truth[idx]),
                                baseline=int(baseline[idx]), augmented=int(augmented[idx])))
    fig.save(destination)
    return dict(seed=SEEDS[0], angle=angle, index_origin=0,
                selection="first eligible image per true class, then earliest four by test index",
                corrected=int(categories[0][1].sum()), regressed=int(categories[1][1].sum()),
                examples=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _, _, images, truth = load_data(args.data)
    runs, predictions = {}, {}
    for arm in ARMS:
        runs[arm], arm_predictions = [], []
        for seed in SEEDS:
            path = args.runs / f"arm{arm:02}_seed{seed}"
            run = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            pred = np.load(path.with_suffix(".pred.npy"), allow_pickle=False)
            if (run["arm"] != arm or run["seed"] != seed or tuple(run["angles"]) != ANGLES
                or pred.shape != (len(ANGLES), len(truth))
                or (pred == truth).sum(1).tolist() != run["correct"]):
                raise ValueError(f"Invalid run data at {path}")
            runs[arm].append(run)
            arm_predictions.append(pred)
        predictions[arm] = np.stack(arm_predictions)
    plot_angle(runs, args.out / "angle_accuracy.png")
    plot_validation(runs, args.out / "validation_curves.png")
    classes = plot_class(predictions, truth, args.out / "class_delta.png")
    examples = plot_examples(images, truth, predictions, args.out / "error_examples.png")
    (args.out / "figure_data.json").write_text(
        json.dumps(dict(class_delta=classes, error_examples=examples,
                        confusion_changes=confusion_changes(predictions, truth)), indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
