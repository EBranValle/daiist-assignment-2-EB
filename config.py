"""
Every choice the pipeline depends on, in one place.

train.py and app.py both import this, so the dashboard always prices errors
with the same figures the training run used.

The cost and operating figures below are PLACEHOLDERS so the pipeline runs end
to end. They are your decisions to make and defend in REPORT.md: replace each
one with your own figure and write down where it came from.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
WEIGHTS_DIR = ROOT / "weights"
ARTIFACTS_DIR = ROOT / "artifacts"
WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"

# --- Dataset (Roboflow Universe) -------------------------------------------
# https://universe.roboflow.com/canned-food-surface-defect-classification/canned-food-surface-defect/6
# Version 6 is the full set of raw images: no "Isolate" crops, no augmented copies.
WORKSPACE = "canned-food-surface-defect-classification"
PROJECT = "canned-food-surface-defect"
VERSION = 6

# Class names that mean "a sellable can". Every other class is a defect, and a
# can with at least one defect box is a positive. Matched case-insensitively.
# TODO: confirm against the class names the training run prints.
GOOD_CLASSES = {"no defect", "no-defect", "no_defect", "normal", "good"}

# --- Training --------------------------------------------------------------
# Environment overrides exist only for a quick smoke run; the real run uses
# the defaults.
BASE_MODEL = os.environ.get("BASE_MODEL", "yolo11n.pt")
EPOCHS = int(os.environ.get("EPOCHS", "20"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))
BATCH = int(os.environ.get("BATCH", "16"))
PATIENCE = 15
SEED = 0

# --- Cost matrix, in euros per can (PLACEHOLDERS) --------------------------
COST_FP = 0.40   # good can rejected: the can and its contents are scrapped
COST_FN = 50.0   # defective can shipped: complaint, return, share of a recall
COST_TP = 0.0    # defective can rejected: lost either way, so no extra cost
COST_TN = 0.0    # good can shipped: the normal case

# --- Operating assumptions (PLACEHOLDERS) ----------------------------------
BASE_RATE = 0.02            # share of cans on the line that are defective
CANS_PER_DAY = 100_000      # line throughput
REVIEW_SECONDS = 10         # time for a person to check one flagged can
REVIEWER_HOURLY_EUR = 25.0  # loaded hourly cost of an inspector
SHIFT_HOURS = 8             # productive hours per inspector per day
REVIEWERS_AVAILABLE = 2     # inspectors you can actually put on the line

COST_REVIEW = REVIEW_SECONDS / 3600 * REVIEWER_HOURLY_EUR  # per reviewed can
