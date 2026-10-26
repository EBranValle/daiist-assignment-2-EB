"""
Assignment 2: Gradio dashboard.

    uv run python main.py app

Reads only what train.py produced: the committed weights in weights/ and the
metrics, test-split scores and sweeps in artifacts/. It never trains anything.
Moving the thresholds re-prices the saved test-split scores, so the numbers on
screen are the same calculation train.py used to choose the operating point.
"""

import json

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import config
import costing


def load_json(name: str) -> dict:
    path = config.ARTIFACTS_DIR / name
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run `uv run python main.py train` first.")
    return json.loads(path.read_text())


if not config.WEIGHTS_PATH.exists():
    raise SystemExit(f"{config.WEIGHTS_PATH} is missing. Commit your trained weights.")
MODEL = YOLO(str(config.WEIGHTS_PATH))
METRICS = load_json("metrics.json")
OPERATING = load_json("operating_point.json")
STATS = load_json("dataset_stats.json")
SCORES = pd.read_csv(config.ARTIFACTS_DIR / "test_scores.csv")
SWEEP = pd.read_csv(config.ARTIFACTS_DIR / "sweep_single.csv")

Y_TRUE = SCORES["defective"].to_numpy(bool)
Y_SCORE = SCORES["finetuned"].to_numpy(float)
NAMES = METRICS["names"]
DEFECT_IDS = [NAMES.index(n) for n in METRICS["positive_classes"]]
IMGSZ = METRICS["train"]["imgsz"]
CHOSEN = OPERATING["chosen"]


def decide(score: float, low: float, high: float) -> str:
    if score < low:
        return "PASS"
    return "FAIL" if score >= high else "HUMAN REVIEW"


def predict(image, low, high):
    if image is None:
        return None, "Upload an image of a can."
    high = max(low, high)
    result = MODEL.predict(image, conf=0.01, imgsz=IMGSZ, verbose=False)[0]
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    defect_conf = conf[np.isin(cls, DEFECT_IDS)]
    score = float(defect_conf.max()) if defect_conf.size else 0.0
    shown = result[conf >= low]
    lines = [f"**Decision: {decide(score, low, high)}** (defect score {score:.2f})", ""]
    lines += [f"- {NAMES[c]}: {p:.2f}" for c, p in zip(cls[conf >= low], conf[conf >= low])]
    return shown.plot()[:, :, ::-1], "\n".join(lines)


def operating_view(low, high):
    high = max(low, high)
    point = costing.summarise(Y_TRUE, Y_SCORE, low, high)
    table = costing.confusion_table(point["rates"])
    summary = (
        f"### €{point['cost_per_1000']:.2f} expected cost per 1,000 cans\n"
        f"- Sent for review: **{100 * point['review_share']:.2f}%** "
        f"= {point['reviews_per_day']:,.0f} cans/day "
        f"= **{point['reviewers_needed']:.2f} inspectors** "
        f"(available: {config.REVIEWERS_AVAILABLE})\n"
        f"- Defects caught (failed or reviewed): {100 * point['recall']:.1f}%\n"
        f"- Good cans wrongly rejected: {100 * point['false_reject_rate']:.2f}%\n"
        f"- Assumes base rate {100 * config.BASE_RATE:.1f}%, "
        f"FN €{config.COST_FN}, FP €{config.COST_FP}, review €{config.COST_REVIEW:.3f}/can"
    )
    designs = pd.DataFrame([
        {"design": "Current sliders", "cost_per_1000": point["cost_per_1000"],
         "inspectors": point["reviewers_needed"]},
        {"design": f"Chosen band ({CHOSEN['low']:.2f}-{CHOSEN['high']:.2f})",
         "cost_per_1000": CHOSEN["cost_per_1000"], "inspectors": CHOSEN["reviewers_needed"]},
        {"design": f"Model alone (t={OPERATING['model_alone']['low']:.2f})",
         "cost_per_1000": OPERATING["model_alone"]["cost_per_1000"], "inspectors": 0.0},
        {"design": "Do nothing: ship everything",
         "cost_per_1000": OPERATING["do_nothing_ship_all"]["cost_per_1000"], "inspectors": 0.0},
        {"design": "Person checks everything",
         "cost_per_1000": OPERATING["person_checks_everything"]["cost_per_1000"],
         "inspectors": OPERATING["person_checks_everything"]["reviewers_needed"]},
    ]).round(2)
    return table, summary, designs, cost_curve(low, high)


def cost_curve(low, high):
    alone = OPERATING["model_alone"]
    fig = go.Figure()
    fig.add_scatter(x=SWEEP["low"], y=SWEEP["cost_per_1000"], mode="lines",
                    name="model decides alone")
    fig.add_scatter(x=[alone["low"]], y=[alone["cost_per_1000"]], mode="markers",
                    marker=dict(size=12, color="red"),
                    name=f"minimum t={alone['low']:.2f}")
    fig.add_vrect(x0=CHOSEN["low"], x1=CHOSEN["high"], fillcolor="orange", opacity=0.2,
                  line_width=0, annotation_text="chosen review band")
    fig.add_vline(x=low, line_dash="dot", annotation_text="low")
    fig.add_vline(x=max(low, high), line_dash="dot", annotation_text="high",
                  annotation_position="bottom right")
    fig.update_layout(xaxis_title="confidence threshold",
                      yaxis_title="expected cost (€ per 1,000 cans)", height=420)
    return fig


def per_class_tables():
    det = METRICS["detection"]
    test_instances = STATS["splits"]["test"]["instances"]
    per_class = pd.DataFrame(det["per_class"])
    per_class["test_instances"] = per_class["class"].map(test_instances)
    per_class["positive"] = per_class["class"].isin(METRICS["positive_classes"])
    baselines = pd.DataFrame(METRICS["baselines"]).T.reset_index(names="model")
    cm = det["confusion_matrix"]
    heat = go.Figure(go.Heatmap(z=cm["matrix"], x=cm["labels"], y=cm["labels"],
                                text=cm["matrix"], texttemplate="%{text}",
                                colorscale="Blues"))
    heat.update_layout(xaxis_title="true class", yaxis_title="predicted class",
                       yaxis_autorange="reversed", height=450)
    return per_class.round(3), baselines.round(3), heat


def dataset_table():
    rows = []
    for name in NAMES:
        area = STATS["object_area_share"][name]
        rows.append({
            "class": name,
            **{s: STATS["splits"][s]["instances"][name] for s in STATS["splits"]},
            "median box % of image": round(100 * (area["median"] or 0), 2),
        })
    splits = pd.DataFrame(STATS["splits"]).T[["images", "defective_images",
                                               "images_without_labels"]]
    return pd.DataFrame(rows), splits.reset_index(names="split")


with gr.Blocks(title="Canned food defect inspection", analytics_enabled=False) as demo:
    ds = METRICS["dataset"]
    gr.Markdown(
        f"# Canned food defect inspection\n"
        f"Dataset `{ds['workspace']}/{ds['project']}/{ds['version']}` · "
        f"positive = {', '.join(METRICS['positive_classes'])} · "
        f"test split: {METRICS['test_images']} cans, {METRICS['test_defective']} defective"
    )
    with gr.Row():
        low = gr.Slider(0, 1, value=CHOSEN["low"], step=0.01,
                        label="Low threshold: below this, PASS")
        high = gr.Slider(0, 1, value=CHOSEN["high"], step=0.01,
                         label="High threshold: at or above this, FAIL (between: review)")

    with gr.Tab("Operating point"):
        with gr.Row():
            confusion = gr.Dataframe(label="Expected cans per 1,000 (actual × outcome)")
            summary = gr.Markdown()
        designs = gr.Dataframe(label="Priced against the alternatives")
        curve = gr.Plot(label="Cost against threshold")

    with gr.Tab("Try an image"):
        with gr.Row():
            image_in = gr.Image(type="pil", label="Upload a can")
            image_out = gr.Image(label="Boxes at or above the low threshold")
        verdict = gr.Markdown()
        gr.Button("Predict").click(predict, [image_in, low, high], [image_out, verdict])

    with gr.Tab("Per-class results"):
        det = METRICS["detection"]
        gr.Markdown(f"Test split: mAP50 **{det['map50']:.3f}**, "
                    f"mAP50-95 **{det['map50_95']:.3f}**")
        per_class, baselines, heat = per_class_tables()
        gr.Dataframe(per_class, label="Per class (test split)")
        gr.Plot(heat, label="Detection confusion matrix (test split)")
        gr.Dataframe(baselines, label="Finetuned vs baselines (same test split)")

    with gr.Tab("Dataset"):
        classes, splits = dataset_table()
        gr.Dataframe(classes, label="Instances per class and split")
        gr.Dataframe(splits, label="Images per split")

    outputs = [confusion, summary, designs, curve]
    for slider in (low, high):
        slider.change(operating_view, [low, high], outputs)
        slider.change(predict, [image_in, low, high], [image_out, verdict])
    demo.load(operating_view, [low, high], outputs)


if __name__ == "__main__":
    demo.launch()
