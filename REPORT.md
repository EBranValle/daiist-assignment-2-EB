# Assignment 2 Report

- **Name**: Edgar Andres Bran Valle
- **Student ID**: 18789
- **Email**: ebran.ieu2022@student.ie.edu 
- **Group**:  BBADBA 5B 

## Business problem, and the cost of being wrong

**Decision.** A canning factory packing a premium preserved product (gourmet
tinned fish, about €15 retail per can) photographs every can after sealing them. For
each can the system decides: **ship it, reject it, or send it to a quality
inspector**. Today that decision is made by people looking at the cans. But this process could be made more efficient. 

**Who acts on it.** The reject gate on the line acts on "fail"; a quality
inspector at the end of the line handles "review".

**Positive class.** A *defective* can: any can labelled Critical, Major **or
Minor** defect. Minor (cosmetic) defects are rejected too, because the product
is premium and food-grade safety is not something the line can take lightly.

**Cost of each outcome (€ per can):**

| | Shipped | Rejected |
|---|---|---|
| **Defective can** | **€50** (FN): penalty, shipping and return handling | **€7.5** (TP): the cost of producing the can, now lost |
| **Good can** | **€0** (TN): sold normally | **€15** (FP): the sale price the can would have earned |

**The asymmetry.** What drives the decision is the gap between the two choices
for each kind of can. Shipping a defective can instead of rejecting it costs
€50 − €7.5 = **€42.5** more; rejecting a good can instead of shipping it costs
**€15** more. A miss is therefore about **2.8× worse** than a false alarm.

In the hypothetical case, shipping a defect would cost the company significantly more, since it could incur penalties and shipping and return shipping costs. This is considerably less expensive than a wasted good, since only the hypothetical cost of production would be less than the costs mentioned before. For this reason, even if a defect is considered minor, it should be rejected to avoid the 50 euros cost and to align with the company's hypothetical premium can and reputational standards. Because a miss costs more than a false alarm, the model should reject cans on weak evidence, meaning a low confidence threshold.

## Dataset

- **Roboflow Universe:** workspace `canned-food-surface-defect-classification`,
  project `canned-food-surface-defect`, **version 6** ("last version").
- **Link:** https://universe.roboflow.com/canned-food-surface-defect-classification/canned-food-surface-defect/6
- **License:** CC BY 4.0. Used as published, not forked.
- **Task:** object detection, 4 classes: `Critical Defect`, `Major Defect`,
  `Minor Defect`, `No defect`.
- **What one image is:** 
  A photograph of a can against a dark background with a bounding box over it or its defect.¡, labeled with the level of severity. One image equals one can that equals on ship/Review/reject decision. 
- **Why version 6:** it is the full set of raw images. Versions 2–4 "isolate"
  (crop) each object, which does not look like a camera on a line, and version
  5 is ~3× augmented copies of the same images.
 - **Why the Dataset **
   The dataset has a large amount of available images (8,096 images with 829 in the test split and 200 per class) to create a solution to the problem above. With the dataset, the model is able to grade the severity of damage and good cans for the hypothetical business case of a premium canned food manufacturing plant. However, it differs from a real manufacturing plant as the images are on a dark background and the cans are all standardized, instead of being, for example, on a conveyor belt. 

## What the dataset's composition implies

**Instances per class and split** (`artifacts/dataset_stats.json`):

| Class | Train | Valid | Test | Median box, % of image |
|---|---|---|---|---|
| Critical Defect | 1,406 | 401 | 202 | 21.6% |
| Major Defect | 1,358 | 385 | 200 | 10.6% |
| Minor Defect | 1,530 | 437 | 227 | 18.6% |
| No defect | 1,364 | 386 | 199 | 14.7% |
| **Images** | **5,658** | **1,609** | **829** | |

- Exactly one labelled object per image; one test image has no label at all
  and is counted as a good can.
- Test split: **629 defective / 200 good cans = 76% defective.**
- Objects are large (median 11–22% of the image), so small-object detection is
  not the difficulty here.
- Classes are balanced, as there are around 200 instances of each, which is enough to trust per-class recall to a few percentage points.
- The test split is 76% defective; a real line is assumed to have a realistic amount of  defective cans of about 2%. Every cost figure is therefore re-weighted to the 2% base rate rather than taken from the raw test counts.
- There are only 200 good test cans in the dataset; a single false reject moves the false positive rate by 0.5%.

## Finetuning and baselines

**Model:** `yolo11n` (detection), finetuned from the pretrained COCO
checkpoint. Detection rather than classification because the dataset is
labelled with boxes; the decision is made per can from the most confident
defect box.

**Training:** Google Colab, Tesla T4 GPU, 20 epochs, image size 640, batch 16,
seed 0, early-stopping patience 15 (not triggered). 134 minutes. Settings are in
`config.py`; the run's record is `artifacts/train_info.json`. Validation
mAP50-95 was still rising at epoch 20 (0.699 → 0.795 → 0.863 → 0.892 → 0.906
at epochs 1/5/10/15/20), so more epochs would likely help a little.

**Results on the same 829-image test split:**

| Model | mAP50 | mAP50-95 | AUC (defective vs good can) | Lowest expected cost, € / 1,000 cans |
|---|---|---|---|---|
| **Finetuned yolo11n** | **0.961** | **0.903** | **0.996** | **€223.50** (threshold 0.06) |
| COCO yolo11n zero-shot | 0 | 0 | 0.549 | €971.62 |
| Trivial: predict nothing (ship every can) | 0 | 0 | 0.500 | €1,000.00 |

COCO has no defect classes, so its mAP on these classes is 0 by construction;
its image-level score is "highest confidence of any COCO detection", which is
barely better than chance (AUC 0.55).

The gap is large and expected. The COCO model was never trained on defects, so it doesn't have classes for them, and the scores ended up close to random guessing since the AUC is low. The fine-tuned model learned exactly what defects look like from the labelled boxes, which allowed it to separate the good and defective cans almost perfectly with an AUC of 0.996. This cut the expected costs from about 972 euros to 223 euros per 1000 cans. 

## Per-class evaluation and failure modes

**Positive class:** defective (Critical + Major + Minor).

| Class | Precision | Recall | AP50 | AP50-95 | Test instances |
|---|---|---|---|---|---|
| Critical Defect | 0.971 | 1.000 | 0.990 | 0.946 | 202 |
| Major Defect | 0.948 | 0.993 | 0.993 | 0.967 | 200 |
| Minor Defect | 0.931 | 0.947 | 0.986 | 0.951 | 227 |
| No defect | 0.952 | **0.734** | **0.874** | **0.746** | 199 |

**Confusion between classes** (test split, rows = predicted, columns = true;
`artifacts/confusion_matrix.png`):

| Predicted ↓ / True → | Critical | Major | Minor | No defect |
|---|---|---|---|---|
| Critical | 202 | 1 | 2 | 0 |
| Major | 0 | 198 | 7 | 0 |
| Minor | 0 | 1 | 218 | 0 |
| No defect | 0 | 0 | 0 | 199 |

**At the can level** (highest defect score per can):
- Lowest score of any defective can: **0.168**. So any threshold up to ~0.16
  catches every defective can in the test split.
- Good cans: 197 of 200 score below 0.06, but one good can
  (`IMG_20220929_174528_jpg.rf.b89af3d9…`) scores **0.956**, a confident false
  alarm.
- The large "background" column in Ultralytics' confusion matrix comes from
  counting every faint duplicate box at confidence 0.001; it is not the error
  rate at our thresholds.
- Fed random noise, the model reports a *Major Defect* at 0.79: it never says
  "this is not a can".

The headline number comes from the defect classes, which all score very high, with Critical Defect being the strongest. The model catches every critical can. "No defect" is the weak class so the model often misses or poorly places the box for a good can, but this doesn't really matter as we are focusing on the cans that have defects. Most mistakes between classes are from over grading, Minor was called Major 7 times and Critical 2 times, and Major was called Critical once. There was only one under-grade (Major called Minor). No defective can is ever predicted as “No defect,” so a severity mistake can change how a can is classified but never lets a defective can ship.

## Cost matrix and operating point

**Figures** (see the business section for where each comes from): FN €50,
FP €15, TP €7.5, TN €0. Human review costs 10 s × €25/h = **€0.069 per can**.

**Base rate:** 2% of cans defective. Test-split rates are re-weighted to this,
because the test split is 76% defective.

**Floor:** at 2% defective and €7.5 per rejected defect, even a perfect decision
costs **€150 per 1,000 cans**. The model only controls the cost above that.

**Sweep, model deciding alone** (`artifacts/sweep_single.csv`,
`artifacts/cost_curve.png`):

| Threshold | € / 1,000 cans | Defects caught | Good cans rejected |
|---|---|---|---|
| **0.06 (minimum)** | **223.50** | 100% | 0.5% |
| 0.30 | 230.26 | 99.2% | 0.5% |
| 0.50 | 238.36 | 98.3% | 0.5% |
| 0.90 | 369.45 | 82.8% | 0.5% |
| 0.96 | 935.14 | 7.6% | 0.0% |

**Sensitivity** (`artifacts/sensitivity.csv`): best threshold by miss cost and
base rate.

| Base rate | FN €10 | €25 | **€50** | €100 | €250 |
|---|---|---|---|---|---|
| 0.2% | 0.96 | 0.96 | 0.06 | 0.06 | 0.06 |
| 0.5% – 2% | 0.96 | 0.06 | **0.06** | 0.06 | 0.06 |
| 5% – 10% | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 |

**One cost for every defect grade.** A shipped Critical defect plausibly costs
far more than €50. Raising the miss cost only lowers the best threshold, and
0.06 is already the lowest threshold that catches every test defect, so pricing
Critical higher would not change this operating point on this test split.

The operating point is a threshold of 0,06 whic catches every defective can and rejects only 0.5% of the good cans for an expected cost of €223.50 per 1,000 cans. The assumption that would change this the most is the cost of a missed defect. This rests on the assumption that a shipped defect costs €50. If that cost were actually below about €25 (or below €50 when defects are very rare), the best threshold would jump to 0.96, and it would be cheaper to let borderline cans ship. A higher cost, such as €100, would change nothing, because at 0.06 the model misses no defects at all.


## What a person still has to check

**Chosen thresholds:** pass < **0.06** ≤ human review < **0.96** ≤ fail. This is
the cheapest pair in the full two-threshold sweep (`artifacts/sweep_band.csv`)
that the 2 available inspectors can staff.

| | Per day (100,000 cans) |
|---|---|
| Share sent for review | **2.34%** |
| Cans reviewed | **~2,340** |
| Time per check | 10 s |
| Inspector hours | ~6.5 h |
| Inspectors needed (8 productive h each) | **0.81** of 2 available |

**Priced against the alternatives** (€ per 1,000 cans; above the €150 floor in
brackets):

| Design | € / 1,000 cans | Inspectors |
|---|---|---|
| **Model + human review band (chosen)** | **151.62** (+1.62) | 0.81 |
| Person checks everything | 219.44 (+69.44) | 34.7 |
| Model decides alone (t = 0.06) | 223.50 (+73.50) | 0 |
| Do nothing (ship every can) | 1,000.00 (+850) | 0 |

**Where the band's saving comes from.** Holding the low edge at 0.06 and moving
the high edge down: 0.96 → €151.62, 0.90 → €223.74, 0.50 → €223.52. The whole
saving comes from sending the one confidently-wrong good can (0.956) to a person
instead of rejecting it. Meanwhile 92% of defective cans fall inside the band,
so most of the inspector's time goes on confirming defects the model already
found.

One inspector is realistic since the review would take 6.5 hours a day. It would be good not to ship the band as it is because its upper edge depends on just one test image, so its better to check it on more data first. Doing nothing is the worst option because every defect would ship. The model alone is not cheaper, but it can still reject some good cans. But checking it all by hand costs the same as needing around 35 people but we only have 2, so the differnce is not big when comparing costs. The model plus a small human review is the cheapest option and the best choice. 


## Limitations & next steps

- The test split is easier than production: single cans, lab lighting, likely
  the same physical cans in train and test.
- The cost result rests on very few test errors (one false reject, zero
  misses); the miss cost is untestable on the split.
- The band's upper edge (0.96) is set by one image.
- One miss cost for all severities.
- The model has no "not a can" answer for unusual input in case something would get into the production line.
- Training stopped at 20 epochs while validation mAP was still rising. Training was also slow as it took almost 2 hours to run.
- Computational power would have to be taken into account if the model is to be implemented since GPUs are expensive. 

## Generative AI use disclosure

Claude (Anthropic) via Claude Code was used to write the code (`train.py`,
`app.py`, `costing.py`, `config.py`), to debug the Colab run, and to fill this
report's tables and facts from the pipeline's output files. The problem, the
dataset version, the cost figures and the decision to reject Minor defects
were my own choices. The report apart from the factual data was also done on my part. The model was also trained by me on a Colab Notebook using a T4 GPU runtime

