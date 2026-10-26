"""
Assignment 2: training pipeline.

    uv run python main.py train

Start to finish, with no manual steps:

1. Download the dataset version named in config.py from Roboflow Universe.
2. Describe it: images and instances per class per split, object sizes.
3. Finetune the pretrained YOLO checkpoint and copy the best weights to weights/.
4. Evaluate on the test split: finetuned model, COCO zero-shot, predict-nothing.
5. Score every test can, sweep the thresholds, and price each operating point.
6. Save everything app.py needs to artifacts/, so the app never trains.
"""

import gc
import json
import os
import platform
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

import config
import costing
from api_key import load_api_key

SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --- 1. Data ---------------------------------------------------------------

def download_dataset() -> Path:
    target = config.DATA_DIR / f"{config.PROJECT}-{config.VERSION}"
    if (target / "data.yaml").exists():
        print(f"Dataset already at {target}, reusing it.")
        return target
    from roboflow import Roboflow

    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(config.WORKSPACE).project(config.PROJECT).version(config.VERSION)
    version.download("yolov11", location=str(target))
    return target


def prepare_data_yaml(dataset: Path) -> tuple[Path, list[str]]:
    """Rewrite Roboflow's data.yaml with absolute paths and a test split."""
    raw = yaml.safe_load((dataset / "data.yaml").read_text())
    names = raw["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    for split in SPLITS:
        if not (dataset / split / "images").is_dir():
            raise SystemExit(
                f"{dataset.name} has no {split}/ split. The brief compares models on a "
                f"held-out test split, so pick a dataset version that has one."
            )
    data_yaml = dataset / "data_abs.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(dataset),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": dict(enumerate(names)),
    }))
    return data_yaml, names


def defect_class_ids(names: list[str]) -> list[int]:
    ids = [i for i, n in enumerate(names) if n.strip().lower() not in config.GOOD_CLASSES]
    if not ids:
        raise SystemExit(f"Every class {names} is listed in GOOD_CLASSES; nothing is a defect.")
    return ids


def read_labels(label_file: Path) -> list[tuple[int, float]]:
    """(class id, box area as a share of the image) for each object."""
    objects = []
    if not label_file.exists():
        return objects
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, coords = int(parts[0]), list(map(float, parts[1:]))
        if len(coords) == 4:
            area = coords[2] * coords[3]
        else:  # polygon: use its bounding box
            xs, ys = coords[0::2], coords[1::2]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        objects.append((cls, area))
    return objects


def split_images(dataset: Path, split: str) -> list[Path]:
    folder = dataset / split / "images"
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def label_path(image: Path) -> Path:
    return image.parent.parent / "labels" / f"{image.stem}.txt"


# --- 2. Describe -----------------------------------------------------------

def describe_dataset(dataset: Path, names: list[str], defect_ids: list[int]) -> dict:
    stats = {"names": names, "defect_classes": [names[i] for i in defect_ids], "splits": {}}
    per_class_areas = {n: [] for n in names}
    for split in SPLITS:
        images = split_images(dataset, split)
        instances = {n: 0 for n in names}
        images_with = {n: 0 for n in names}
        defective_images = empty_images = 0
        for image in images:
            objects = read_labels(label_path(image))
            if not objects:
                empty_images += 1
            classes = {c for c, _ in objects}
            if classes & set(defect_ids):
                defective_images += 1
            for c in classes:
                images_with[names[c]] += 1
            for c, area in objects:
                instances[names[c]] += 1
                per_class_areas[names[c]].append(area)
        stats["splits"][split] = {
            "images": len(images),
            "defective_images": defective_images,
            "images_without_labels": empty_images,
            "instances": instances,
            "images_containing": images_with,
        }
    stats["object_area_share"] = {
        n: {
            "count": len(a),
            "p10": float(np.percentile(a, 10)) if a else None,
            "median": float(np.median(a)) if a else None,
            "p90": float(np.percentile(a, 90)) if a else None,
        }
        for n, a in per_class_areas.items()
    }

    rows = [
        {"class": n, **{s: stats["splits"][s]["instances"][n] for s in SPLITS},
         "median_area_%": round(100 * (stats["object_area_share"][n]["median"] or 0), 2)}
        for n in names
    ]
    print("\nInstances per class and split:\n" + pd.DataFrame(rows).to_string(index=False))
    for s in SPLITS:
        d = stats["splits"][s]
        print(f"{s:>5}: {d['images']} images, {d['defective_images']} defective, "
              f"{d['images_without_labels']} without labels")
    return stats


# --- 3. Finetune -----------------------------------------------------------

def finetune(data_yaml: Path) -> tuple[Path, dict]:
    model = YOLO(config.BASE_MODEL)
    started = time.time()
    model.train(
        data=str(data_yaml),
        epochs=config.EPOCHS,
        imgsz=config.IMGSZ,
        batch=config.BATCH,
        patience=config.PATIENCE,
        seed=config.SEED,
        project=str(config.ROOT / "runs"),
        name="finetune",
        exist_ok=True,
    )
    config.WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, config.WEIGHTS_PATH)
    info = {
        "base_model": config.BASE_MODEL,
        "epochs_requested": config.EPOCHS,
        "epochs_run": int(model.trainer.epoch) + 1,
        "imgsz": config.IMGSZ,
        "batch": config.BATCH,
        "patience": config.PATIENCE,
        "seed": config.SEED,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "platform": platform.platform(),
        "minutes": round((time.time() - started) / 60, 1),
    }
    # Saved now, so a crash later in the pipeline does not lose it.
    (config.ARTIFACTS_DIR / "train_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nSaved best weights to {config.WEIGHTS_PATH.relative_to(config.ROOT)}")
    # Free the trainer and its dataloaders before evaluation needs the RAM.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return config.WEIGHTS_PATH, info


def reuse_weights() -> tuple[Path, dict]:
    """Skip training and evaluate the weights already in weights/."""
    if not config.WEIGHTS_PATH.exists():
        raise SystemExit(f"REUSE_WEIGHTS is set but {config.WEIGHTS_PATH} does not exist.")
    info_path = config.ARTIFACTS_DIR / "train_info.json"
    if not info_path.exists():
        raise SystemExit(f"REUSE_WEIGHTS is set but {info_path} does not exist.")
    print(f"REUSE_WEIGHTS set: evaluating {config.WEIGHTS_PATH.relative_to(config.ROOT)} "
          f"without retraining.")
    return config.WEIGHTS_PATH, json.loads(info_path.read_text())


# --- 4. Evaluate -----------------------------------------------------------

def evaluate_detection(weights: Path, data_yaml: Path, names: list[str]) -> dict:
    """Ultralytics' own test-split metrics: mAP, per-class P/R/AP, confusion."""
    results = YOLO(str(weights)).val(
        data=str(data_yaml), split="test", imgsz=config.IMGSZ, batch=config.BATCH,
        project=str(config.ROOT / "runs"), name="test_eval", exist_ok=True,
    )
    box = results.box
    per_class = []
    for i, cls in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        per_class.append({
            "class": names[int(cls)], "precision": float(p), "recall": float(r),
            "ap50": float(ap50), "ap50_95": float(ap),
        })
    for name in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        src = Path(results.save_dir) / name
        if src.exists():
            shutil.copy(src, config.ARTIFACTS_DIR / name)
    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": per_class,
        # Ultralytics convention: rows are predicted class, columns true class.
        "confusion_matrix": {
            "labels": names + ["background"],
            "matrix": results.confusion_matrix.matrix.astype(int).tolist(),
        },
    }


def score_images(model: YOLO, images: list[Path], class_ids: list[int] | None) -> np.ndarray:
    """Highest confidence of any box of `class_ids` per image (0 if none)."""
    scores = []
    for image in images:
        # One image at a time keeps memory flat on large test splits.
        result = model.predict(str(image), conf=0.01, imgsz=config.IMGSZ, verbose=False)[0]
        conf = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        if class_ids is not None:
            conf = conf[np.isin(cls, class_ids)]
        scores.append(float(conf.max()) if conf.size else 0.0)
    return np.array(scores)


def auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """Probability a random defective can outscores a random good one."""
    pos, neg = scores[y_true], scores[~y_true]
    if not len(pos) or not len(neg):
        return None
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def image_level(y_true: np.ndarray, scores: np.ndarray) -> dict:
    at_half = costing.summarise(y_true, scores, 0.5, 0.5)
    flagged = scores >= 0.5
    sweep = costing.sweep_single(y_true, scores)
    best = sweep.loc[sweep["cost_per_1000"].idxmin()]
    return {
        "auc": auc(y_true, scores),
        "recall_at_0.5": at_half["recall"],
        "precision_at_0.5": float(y_true[flagged].mean()) if flagged.any() else None,
        "best_threshold": float(best["low"]),
        "best_cost_per_1000": float(best["cost_per_1000"]),
    }


# --- 5. Cost sweep ---------------------------------------------------------

def choose_operating_points(y_true: np.ndarray, scores: np.ndarray) -> dict:
    single = costing.sweep_single(y_true, scores)
    band = costing.sweep_band(y_true, scores)
    single.to_csv(config.ARTIFACTS_DIR / "sweep_single.csv", index=False)
    band.to_csv(config.ARTIFACTS_DIR / "sweep_band.csv", index=False)

    best_single = single.loc[single["cost_per_1000"].idxmin()]
    best_band = band.loc[band["cost_per_1000"].idxmin()]
    staffed = band[band["reviewers_needed"] <= config.REVIEWERS_AVAILABLE]
    best_staffed = staffed.loc[staffed["cost_per_1000"].idxmin()]

    def point(row):
        return costing.summarise(y_true, scores, row["low"], row["high"])

    return {
        "model_alone": point(best_single),
        "unconstrained_band": point(best_band),
        "chosen": point(best_staffed),
        **costing.alternatives(y_true),
    }


def sensitivity(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """How the best single threshold moves when the key assumptions move."""
    rows = []
    for base_rate in (0.002, 0.005, 0.01, 0.02, 0.05, 0.10):
        for cost_fn in (5.0, 20.0, 50.0, 200.0, 1000.0):
            sweep = costing.sweep_single(y_true, scores, base_rate, cost_fn)
            best = sweep.loc[sweep["cost_per_1000"].idxmin()]
            rows.append({
                "base_rate": base_rate, "cost_fn": cost_fn,
                "best_threshold": best["low"], "cost_per_1000": best["cost_per_1000"],
            })
    return pd.DataFrame(rows)


def plot_cost_curve(operating: dict) -> None:
    single = pd.read_csv(config.ARTIFACTS_DIR / "sweep_single.csv")
    alone = operating["model_alone"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(single["low"], single["cost_per_1000"], label="model decides alone")
    ax.axvline(alone["low"], ls="--", c="grey")
    ax.scatter([alone["low"]], [alone["cost_per_1000"]], c="red", zorder=3,
               label=f"minimum: t={alone['low']:.2f}, €{alone['cost_per_1000']:.1f}/1000")
    ax.axhline(operating["do_nothing_ship_all"]["cost_per_1000"], ls=":", c="black",
               label="ship everything")
    ax.set(xlabel="confidence threshold", ylabel="expected cost (€ per 1,000 cans)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.ARTIFACTS_DIR / "cost_curve.png", dpi=120)
    plt.close(fig)


# --- Pipeline --------------------------------------------------------------

def main() -> None:
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    dataset = download_dataset()
    data_yaml, names = prepare_data_yaml(dataset)
    defect_ids = defect_class_ids(names)
    print(f"Classes: {names}\nTreated as defect (positive): {[names[i] for i in defect_ids]}")

    stats = describe_dataset(dataset, names, defect_ids)
    (config.ARTIFACTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))

    # REUSE_WEIGHTS=1 re-runs evaluation and costing on the committed weights,
    # e.g. after changing a cost in config.py. The default trains from scratch.
    if os.environ.get("REUSE_WEIGHTS") == "1":
        weights, train_info = reuse_weights()
    else:
        weights, train_info = finetune(data_yaml)
    detection = evaluate_detection(weights, data_yaml, names)

    test_images = split_images(dataset, "test")
    y_true = np.array([
        any(c in defect_ids for c, _ in read_labels(label_path(p))) for p in test_images
    ])
    if y_true.all() or not y_true.any():
        raise SystemExit(
            "The test split has only one kind of can (all defective or all good), so "
            "false positives or false negatives cannot be counted. Pick another version."
        )
    scores = {
        "finetuned": score_images(YOLO(str(weights)), test_images, defect_ids),
        # COCO has no defect classes, so any detection at all is its best guess.
        "coco_zero_shot": score_images(YOLO(config.BASE_MODEL), test_images, None),
        "predict_nothing": np.zeros(len(test_images)),
    }
    pd.DataFrame({"image": [p.name for p in test_images], "defective": y_true, **scores}) \
        .to_csv(config.ARTIFACTS_DIR / "test_scores.csv", index=False)

    baselines = {
        "finetuned": {"map50": detection["map50"], "map50_95": detection["map50_95"]},
        # No COCO class is a can defect, so it cannot match a single defect box.
        "coco_zero_shot": {"map50": 0.0, "map50_95": 0.0},
        "predict_nothing": {"map50": 0.0, "map50_95": 0.0},
    }
    for name, s in scores.items():
        baselines[name].update(image_level(y_true, s))
    print("\nTest split, all three models:\n" + pd.DataFrame(baselines).T.to_string())

    operating = choose_operating_points(y_true, scores["finetuned"])
    sensitivity(y_true, scores["finetuned"]).to_csv(
        config.ARTIFACTS_DIR / "sensitivity.csv", index=False
    )
    plot_cost_curve(operating)

    metrics = {
        "dataset": {"workspace": config.WORKSPACE, "project": config.PROJECT,
                    "version": config.VERSION},
        "names": names,
        "positive_classes": [names[i] for i in defect_ids],
        "test_images": len(test_images),
        "test_defective": int(y_true.sum()),
        "train": train_info,
        "detection": detection,
        "baselines": baselines,
    }
    (config.ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (config.ARTIFACTS_DIR / "operating_point.json").write_text(json.dumps(operating, indent=2))

    c = operating["chosen"]
    print(
        f"\nChosen band: pass < {c['low']:.2f} <= review < {c['high']:.2f} <= fail\n"
        f"  €{c['cost_per_1000']:.2f} per 1,000 cans, {100 * c['review_share']:.2f}% reviewed, "
        f"{c['reviewers_needed']:.2f} inspectors\n"
        f"Artifacts written to {config.ARTIFACTS_DIR.relative_to(config.ROOT)}/"
    )


if __name__ == "__main__":
    main()
