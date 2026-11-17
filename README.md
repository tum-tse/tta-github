# Time to Adapt (TTA): replication package

Code for the paper *Time to Adapt: A Reaction-Time Metric for Short-Term Traffic Prediction under Disruptions* (H. Begic, S. N. Lone, J.Yang, C. Antoniou).

TTA measures how many minutes a forecast takes to register a disruption after the detector data show it. The same one-sided CUSUM rule finds the onset in the observations and in each forecast. Disruptions are found automatically in the data, and TTA is reported with the share of events registered and a false-alarm rate (FAR). The metric needs only observations and forecasts.

This repository contains the metric, the full Munich pipeline (data loading, models, trained checkpoints, scoring), and the scripts for every figure and table. It does not contain raw data or per-event outputs. Running the steps below regenerates them.

## Contents

```
models/
  tta.py                        the metric: profile, CUSUM tally, events, TTA, FAR, summaries, bootstrap
  test_models_tta_all.py        Munich: builds every forecast and scores all models (main pipeline)
  test_tta_duration_sweep.py    Munich: re-scores at d_min = 5, 10, 15, 30 min from the cached forecasts
  plot_tta_blindspot.py         Munich: forecast made perfect during every disruption (Sec. 5.2)
  tta_data.py                   loading helpers used by the two scripts above
  spatial/gnn.py                data loading, windowing, training loop shared by the deep models
  spatial/gcn_simple.py         GCN
  temporal/bilstm.py            BiLSTM
  spatio_temporal/gnn_bilstm_a.py  GCN-BiLSTM
imputation.py                   imputation of missing detector values
global/a8-a99-a9_cleaned/       detector graph of the A8-A99-A9 corridor (nodes, directed edges)
meta_fixed.csv                  detector sites and their lane units
checkpoints/                    trained Munich models used in the paper (best_model.pth + run_summary.json)
figures/                        one script per figure (Figs. 1-7) and the trade-off curves
reference_results/              the paper's summary tables, for comparison with a rerun
benchmarks/                     the four benchmark networks: data, setup and an input example (benchmarks/README.md)
```

## 1. Environment

Python 3.10 and the versions in `requirements.txt`. A CUDA GPU is recommended for the deep models; the tree models and the scoring run on CPU.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run every command below from the repository root.

Check the metric on synthetic data (about one minute, no data needed):

```bash
python models/tta.py --selftest        # ends with "self-test passed"
```

## 2. Data (Munich)

The Munich data are 1-minute loop-detector records of Die Autobahn GmbH des Bundes on the A8-A99-A9 motorway triangle, available from the SCUDO Open Data Platform:
https://opendataplatform.scudo-project.eu/?uc=traffic-management&cat=05-traffic-conditions

Put all lane files in one folder, `data/autobahn/`, or point `TTA_DATA_DIR` to another folder. Two periods are used:

| Period | File suffix | Used for |
|---|---|---|
| 2024 | `2024-01-01_bis_2025-01-01` | evaluation (held out for every model) |
| 2025 | `2025-01-01_bis_2026-01-01` | training of all learned models |

Each file is one lane of one detector, named `FG1_Kurz_<unit>_agg1min_<suffix>.csv`, semicolon-separated, with columns `t_start;q_kfz_h;q_lkw_h;q_pkw_h;v_kfz;v_lkw;v_pkw;b;t_netto;s_kfz`. The lane units of each detector are listed in `meta_fixed.csv`.

The four benchmark networks (METR-LA, PEMS-BAY, Chengdu, Shenzhen) use public data from the ST-GFSL release; where to get them, the setup and an example of the model input are in `benchmarks/README.md`.

## 3. Reproducing the Munich results

**Step 1 (optional): train the deep models.** The checkpoints used in the paper are in `checkpoints/`, so this step can be skipped. To retrain:

```bash
python models/spatial/gcn_simple.py
python models/temporal/bilstm.py
python models/spatio_temporal/gnn_bilstm_a.py
```

Each trains on 2025 data and writes `checkpoints/<model>/<timestamp>/best_model.pth`. To score a retrained model, set its path in `DL_MODELS` in `models/test_models_tta_all.py`. Training on a GPU is not bit-for-bit deterministic, so retrained models give slightly different numbers.

**Step 2: forecasts and scores.** This step trains Random Forest and XGBoost, runs the three checkpoints over 2024, adds the reference forecasts and scores everything:

```bash
python models/test_models_tta_all.py
```

Output in `models/test_results_tta/`: `summary.csv` (Table 5, Munich rows), `summary_all_h.csv` (every forecast threshold h_p), `matched_far.csv` (Table 6, Munich rows), `pairs.csv` (paired comparisons), `events.csv`, `per_event.csv.gz`, `config.json` (Table 3, Munich row), `fig_example_event_*` (Fig. 6). Forecasts are cached in `models/test_results_tta/cache/`, so later steps do not recompute them.

**Step 3: sensitivity and the counterfactual.**

```bash
python models/test_tta_duration_sweep.py   # d_min = 5/10/15/30 (Sec. 5.1)
python models/plot_tta_blindspot.py        # perfect forecast during disruptions: MAE -3 %, TTA -87 % (Sec. 5.2)
```

Compare the outputs with `reference_results/munich/`.

## 4. Figures

The figure scripts read summary tables from `reference_results/` by default. To draw them from your own rerun, set `TTA_RESULTS` to a folder with the same layout. Output goes to `figures/out/`.

| Paper | Script |
|---|---|
| Fig. 1 | `python figures/make_fig1_why_simple.py` |
| Fig. 2 | `python figures/make_fig2_operator.py` |
| Fig. 3 | `python figures/make_fig3_rank.py` |
| Fig. 4 | `python figures/make_fig4_models.py` |
| Fig. 5 | `python figures/make_fig5_persistence.py` |
| Fig. 6 | `python figures/plot_example_event_ieee.py` |
| Fig. 7 | `python figures/plot_matched_far.py` |

## 5. Where each result comes from

| Paper | File |
|---|---|
| Table 3 (events) | `n_events` in `summary.csv` and `events_per_detector_day` in `config.json` of each run |
| Table 4 (parameters) | `TTAConfig` defaults in `models/tta.py`; `min_normal_level=50` on Munich only |
| Table 5 | `reference_results/summary_all_datasets.csv` |
| Table 6 | `reference_results/matched_far_common.csv` |
| Kendall's tau (Fig. 3) | computed in `figures/make_fig3_rank.py` from Table 5 |
| Paired differences (Sec. 5.1, 5.3) | `pairs.csv` of each run |
| Duration sensitivity | `duration_sweep.csv` of each run |
| Sanity checks (Sec. 4.4) | `reference_results/benchmarks/<dataset>/sanity.csv`; persistence and profile rows of Table 5 |

Confidence intervals of a model's median TTA are bootstrapped over events (`tta.py`); confidence intervals of paired differences are bootstrapped over days (`test_models_tta_all.py`).

## 6. Using TTA on your own forecasts

```python
import sys; sys.path.insert(0, "models")
import tta

profile = tta.build_profile(history, history_times)            # time-of-week median and spread
res = tta.evaluate(obs, times, {"MyModel": {"t+5min": (pred, 5.0)}}, profile, cfg=tta.TTAConfig())
print(tta.summary_table(res))
```

`obs` and `pred` have shape `[T, N]` (time × detector) in physical units, and `pred[t]` is the forecast valid at `times[t]`. The second value is the lead time in minutes. Persistence and the historical profile are added automatically as references. Report mean TTA and FAR next to MAE, RMSE and MAPE, with the parameters and a persistence row.

## Citation

[to be added on publication]

## License

[to be added]
