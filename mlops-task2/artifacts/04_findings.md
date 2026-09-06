# EDA findings - notebook 4

Data source: `03_train.parquet` only (64,320 orders,
2016-09-15 -> 2018-03-31).
Base rate: 7.95% late. Neither val nor test was opened.

## The strongest signals

| Feature | AUC alone | Range |
|---|---|---|
| `promise_per_100km` (derived) | 0.612 | 14.8% -> 3.4% |
| `distance_km` | 0.589 | 3.7% -> 12.8% across the deciles |
| `freight_total` | 0.588 | - |
| `same_state` | - | 4.61% against 9.66% |
| `customer_state` | - | 4.72% (PR) -> 19.43% (MA) |

## Main discoveries

1. **The raw promise carries no signal (AUC 0.487) because its relationship is an inverted U:**
   short promises 4.9% and long ones 4.7%, with the danger in the middle (10.4%).
   The cause: the correlation between promise and distance = 0.65 - the company stretches the promise with distance, so the effect cancels out.
2. **`promise_per_100km` solves the puzzle** and becomes the strongest signal in the data: from 14.8% at the most ambitious promise down to 3.4% at the most generous.
3. **Missing customer coordinates are a signal, not a gap**: 173 orders with a late rate of 10.98% against 7.94% - added as a binary indicator.
4. **Product attributes predict nothing** (price/weight/volume/item count all ~0.5):
   lateness is a route problem, not a parcel problem.
5. **The month is a strong signal but a trap** - it does not generalise forward, deliberately excluded.
6. **The time within a day or a week is completely flat** - 7.38% to 8.55%.

## The model decision

Tree models first (a non-monotonic relationship + heavy skew + real interactions),
with a logistic regression as a baseline for comparison. The target: beat the ROC-AUC = 0.679 measured in Task 1.

## Known limitations

- 3 orders have a distance greater than the width of Brazil - corrupt coordinates that survived the median.
- The state of SP makes up 71% of the sellers - state features are effectively "SP versus the rest".
- 28 rare product categories (<100 orders) cover 1.5% - grouped into "rare".
