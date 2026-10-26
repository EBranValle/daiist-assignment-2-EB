"""
Turn per-can defect scores into costs, shared by train.py and app.py.

Each test image is one can. Its score is the highest confidence of any defect
box the model drew on it (0 if none), and it is positive if its labels contain
any defect box. Two thresholds split cans into three outcomes:

    score <  low            -> pass   (shipped)
    low <= score < high     -> review (a person decides, assumed correct)
    score >= high           -> fail   (rejected)

With low == high there is no review band and the model decides alone.

Rates are measured on the test split, then re-weighted to the assumed
production base rate, because the test split's defect share is whatever the
dataset author collected and not what the line produces.
"""

import numpy as np
import pandas as pd

import config


def outcome_rates(y_true, scores, low, high):
    """Share of positives and of negatives that end up in each outcome."""
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    passed = scores < low
    failed = scores >= high
    review = ~passed & ~failed
    rates = {}
    for name, mask in (("pos", y_true), ("neg", ~y_true)):
        n = max(int(mask.sum()), 1)
        rates[name] = {
            "pass": passed[mask].sum() / n,
            "review": review[mask].sum() / n,
            "fail": failed[mask].sum() / n,
        }
    return rates


def expected_cost(rates, base_rate=config.BASE_RATE, cost_fn=config.COST_FN):
    """Expected euros per can at the given base rate."""
    pos, neg = rates["pos"], rates["neg"]
    # A reviewed can costs the review, then the person gets it right.
    cost_pos = (
        pos["pass"] * cost_fn
        + pos["review"] * (config.COST_REVIEW + config.COST_TP)
        + pos["fail"] * config.COST_TP
    )
    cost_neg = (
        neg["pass"] * config.COST_TN
        + neg["review"] * (config.COST_REVIEW + config.COST_TN)
        + neg["fail"] * config.COST_FP
    )
    return base_rate * cost_pos + (1 - base_rate) * cost_neg


def review_share(rates, base_rate=config.BASE_RATE):
    return base_rate * rates["pos"]["review"] + (1 - base_rate) * rates["neg"]["review"]


def reviewers_needed(share):
    """Inspectors needed to check `share` of the day's cans."""
    hours = share * config.CANS_PER_DAY * config.REVIEW_SECONDS / 3600
    return hours / config.SHIFT_HOURS


def summarise(y_true, scores, low, high, base_rate=config.BASE_RATE, cost_fn=config.COST_FN):
    """Everything the report and dashboard show for one pair of thresholds."""
    rates = outcome_rates(y_true, scores, low, high)
    share = review_share(rates, base_rate)
    cost = expected_cost(rates, base_rate, cost_fn)
    return {
        "low": float(low),
        "high": float(high),
        "cost_per_can": float(cost),
        "cost_per_1000": float(1000 * cost),
        "review_share": float(share),
        "reviews_per_day": float(share * config.CANS_PER_DAY),
        "reviewers_needed": float(reviewers_needed(share)),
        "recall": float(rates["pos"]["fail"] + rates["pos"]["review"]),
        "false_reject_rate": float(rates["neg"]["fail"]),
        "rates": rates,
    }


# Starts at 0.01 so a can with no detection at all always passes.
THRESHOLDS = np.round(np.linspace(0.01, 1, 100), 2)


def sweep_single(y_true, scores, base_rate=config.BASE_RATE, cost_fn=config.COST_FN):
    """Model decides alone: cost at every single threshold."""
    rows = [summarise(y_true, scores, t, t, base_rate, cost_fn) for t in THRESHOLDS]
    return pd.DataFrame([{k: v for k, v in r.items() if k != "rates"} for r in rows])


def sweep_band(y_true, scores, base_rate=config.BASE_RATE):
    """Every (low, high) pair with low <= high."""
    rows = [
        summarise(y_true, scores, lo, hi, base_rate)
        for i, lo in enumerate(THRESHOLDS)
        for hi in THRESHOLDS[i:]
    ]
    return pd.DataFrame([{k: v for k, v in r.items() if k != "rates"} for r in rows])


def alternatives(y_true, base_rate=config.BASE_RATE):
    """The three designs every operating point is priced against, per can."""
    nothing = base_rate * config.COST_FN + (1 - base_rate) * config.COST_TN
    everyone = config.COST_REVIEW + base_rate * config.COST_TP + (1 - base_rate) * config.COST_TN
    return {
        "do_nothing_ship_all": {
            "cost_per_1000": 1000 * nothing,
            "reviewers_needed": 0.0,
        },
        "person_checks_everything": {
            "cost_per_1000": 1000 * everyone,
            "reviewers_needed": reviewers_needed(1.0),
        },
    }


def confusion_table(rates, base_rate=config.BASE_RATE, per=1000):
    """Expected cans per `per` in each (actual, outcome) cell."""
    rows = []
    for actual, key, weight in (("defective", "pos", base_rate), ("good", "neg", 1 - base_rate)):
        rows.append({
            "actual": actual,
            **{o: round(per * weight * rates[key][o], 2) for o in ("pass", "review", "fail")},
        })
    return pd.DataFrame(rows)
