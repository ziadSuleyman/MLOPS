# Model results - notebook 6

**Final model:** logistic regression (C=0.3, class_weight="balanced")
on 31 features, trained on 64,320 orders (up to 2018-03-31).
**Label:** the calendar-day rule (notebook 2).

## Results

| | validation | test |
|---|---|---|
| orders | 13,547 | 18,603 |
| late rate | 5.53% | 3.61% |
| **ROC-AUC** | **0.7739** | **0.7107** |
| PR-AUC | 0.1709 | 0.0849 |
| recall @ 5% | 20.7% | 15.0% |
| lift @ 5% | 4.14x | 3.01x |

The test set was opened once, after every decision had been fixed.

## Comparison against the baselines (on validation)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| "nothing is late" | - (accuracy 94.5% with no usefulness at all) | - |
| B1: promise_per_100km alone | 0.6953 | 0.1433 |
| Random Forest | 0.7303 | 0.149 |
| Gradient Boosting (depth=2) | 0.7526 | 0.1709 |
| **logistic regression (winner)** | **0.7739** | **0.1709** |

The Task 1 reference (purchase-time features, a different label and split): ROC-AUC = 0.679.

## Top five features (permutation importance on validation)

promised_days           0.1261
customer_state_te       0.0649
same_state              0.0436
distance_km             0.0255
main_seller_state_te    0.0202

## Two results that contradicted the expectation

1. **The trees lost to logistic regression** despite the notebook 4 prediction. The likely cause: the feature
   engineering already handled the non-linearity (target encoding is monotonic by construction), and the moving
   base rate punishes flexibility. The evidence: boosting performance degrades as rounds increase - memorising
   the training period rather than learning.
2. **`promised_days` is the most important feature** even though its univariate AUC was 0.487.
   Its effect was masked by the effect of distance (correlation 0.65); once the model sees both together it surfaces.
   Conversely `promise_per_100km` became nearly redundant - it was needed to see the phenomenon, not to represent it.

## The operational decision

At an alert budget of **5%**: we catch **15.0%** of the late orders
with a lift of **3.01x** over random targeting (on the test set).
It is the **budget percentage** that is carried over, not the numeric threshold, because the base rate shifts between periods.

## Known limitations

- The split is by purchase date while the label is known at delivery: ~6% of train was delivered after the cut-off (notebook 3).
- The model is trained on train alone; in production it would be retrained on train+val with the encoders rebuilt.
- The base rate is non-stationary (7.95% -> 5.53% -> 3.61%) - any deployment needs drift monitoring and recalibration.
- **ROC-AUC fell from 0.7739 to 0.7107** between validation and test. The metric is a ranking one, so the drop is real
  and not a base-rate artefact: the test period sits 3-5 months beyond the last thing the model saw. The conclusion: periodic retraining, not a one-off deployment.
