# Benchmark networks (PEMS-BAY, METR-LA, Chengdu, Shenzhen)

Besides Munich, TTA is evaluated on four public speed-forecasting benchmarks. Scoring uses the same
`models/tta.py`, with the settings in `reference_results/benchmarks/<dataset>/d15/config.json`.
The raw data are not included in this repository. Data download, model training and scoring scripts
for these networks are to be added.

## Data

The four networks are the ones used in ST-GFSL (Lu et al., KDD 2022, https://arxiv.org/abs/2205.13947).
The authors share them in one archive, linked from https://github.com/RobinLu1209/ST-GFSL. Each network
is two NumPy files, `dataset.npy` (traffic) and `matrix.npy` (road graph); keep each pair in its own
folder, e.g. `data/metr_la/`.

| Network | Quantity | Detectors | Step | Period (steps) | Test period (last 20 %) | Forecast lead |
|---|---|---|---|---|---|---|
| METR-LA | speed (mph), Los Angeles highways | 207 | 5 min | 2012-03-01 to 2012-06-27 (34,272) | 2012-06-04 to 2012-06-27 | 5 min |
| PEMS-BAY | speed (mph), Bay Area highways | 325 | 5 min | 2017-01-01 to 2017-06-30 (52,116) | 2017-05-25 to 2017-06-30 | 5 min |
| Chengdu | speed, Didi GAIA road segments | 524 | 10 min | 2018-01-01 to 2018-04-30 (17,280) | 2018-04-07 to 2018-04-30 | 10 min |
| Shenzhen | speed, Didi GAIA road segments | 627 | 10 min | 2018-01-01 to 2018-04-30 (17,280) | 2018-04-07 to 2018-04-30 | 10 min |

Setup used here (it differs from the few-shot, cross-city setup of ST-GFSL):

- Split in time: first 70 % training, next 10 % validation, last 20 % test. Nothing is trained on the test period, and there is no cross-city transfer.
- Input: the last 12 steps; target: the next step, so the lead is one step (5 or 10 min).
- The files hold values in physical units. In METR-LA a speed of 0 means "missing": set it to NaN before scoring (PEMS-BAY has no zeros). Model outputs must be de-normalised before scoring.
- The files carry no dates. For METR-LA, Chengdu and Shenzhen, rebuild them from the start date in the table and a fixed step (no gaps). PEMS-BAY is in local Pacific time and skips the hour lost to daylight saving on 2017-03-12 (52,116 steps, not 52,128), so use `pd.date_range("2017-01-01", "2017-06-30 23:55", freq="5min", tz="US/Pacific")`. The time-of-day channel of PEMS-BAY is a plain step counter and does not skip that hour; do not use it for dates there.
- The "normal" profile is built from the full series, with the public holidays (and, for Chengdu and Shenzhen, the make-up workdays) listed in `config.json`.
- `TTAConfig` is the Munich one (`direction="down"`, 15 min / 15 %), except `min_normal_level` is not set (speed, not flow).

## What the models see (METR-LA example)

Both files below are real METR-LA data; the other three networks have the same layout, with `N` and
the step changed.

**`dataset.npy`** has shape `[T, N, 2]` = `[34272, 207, 2]` (float64): channel 0 is the speed, channel 1
the time of day as a fraction of the day (`0.2917` = 07:00). Eight steps of the first four detectors,
Thursday 2012-03-01 (speed, mph):

| time | det 0 | det 1 | det 2 | det 3 |
|---|---|---|---|---|
| 06:30 | 67.2 | 65.9 | 66.9 | 52.9 |
| 06:35 | 68.4 | 65.6 | 67.1 | 54.6 |
| 06:40 | 68.2 | 64.6 | 61.2 | 52.8 |
| 06:45 | 66.9 | 66.3 | 57.4 | 52.2 |
| 06:50 | 67.0 | 63.0 | **24.8** | 50.6 |
| 06:55 | 69.2 | 64.7 | 31.6 | 53.9 |
| 07:00 | 68.2 | 65.1 | 31.8 | 49.8 |
| 07:05 | 68.1 | 65.1 | 26.9 | 51.0 |

Detector 2 slows from 57 to 25 mph at 06:50: this is the kind of event TTA times. The deep models take
windows of 12 steps of this array (`[12, 207, 2]`) and predict the next-step speed of every detector
(`[207]`).

**Tree models** get one row per (detector, target time), built from data before the target. For
detector 2 and target 07:00 above:

| `lag1` | `lag2` | `lag3` | `lag4` | `lag5` | `lag6` | `tod_sin` | `tod_cos` | `weekend` | `normal_target` | `dev_last` | target |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 31.6 | 24.8 | 57.4 | 61.2 | 67.1 | 66.9 | 0.966 | -0.259 | 0 | from profile | from profile | 31.8 |

`lag1` to `lag6` are the six previous steps, `tod_sin`/`tod_cos` encode the minute of day, `normal_target`
is the profile's normal speed for the target time (training period) and `dev_last` is `lag1` minus its normal
value. The profile needs the full series, so those two are not shown from this slice.

**`matrix.npy`** is the road graph, `[N, N]` = `[207, 207]` (float32): a weighted, directed adjacency
matrix with 1,722 non-zero entries (1,515 links and 207 self-loops), weights from 0.10 to 1.00. It is
not symmetric. Top-left corner:

```
[[1.   0.   0.   0.   0.  ]
 [0.   1.   0.39 0.   0.  ]
 [0.   0.72 1.   0.   0.  ]
 [0.   0.   0.   1.   0.63]
 [0.   0.   0.   0.63 1.  ]]
```

## Reference results

`reference_results/benchmarks/<dataset>/` holds, for each network, `d15/` (`summary.csv`,
`summary_all_h.csv`, `matched_far.csv`, `pairs.csv`, `config.json`), `duration_sweep.csv` and `sanity.csv`.
