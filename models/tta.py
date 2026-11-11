"""
tta.py — Time to Adapt (TTA)
============================

A reaction-time metric for short-term traffic forecasts, to be reported
ALONGSIDE MAE / RMSE / MAPE.

    MAE/RMSE/MAPE : how LARGE the forecast error is (average over minutes)
    TTA           : how LONG the forecast takes to register a disruption
                    (one value per disruption event, in minutes)

Idea
----
1. From the OBSERVED series, find when each detector entered a disrupted state
   (traffic unusually far from what is normal for that time of week). These
   are the events. No incident log is needed.
2. Apply the SAME onset rule to each model's PREDICTED series.
3. TTA = (time the prediction entered the disrupted state)
       − (time the observation entered it).

The onset rule is a one-sided CUSUM on the standardised deviation from a
historical time-of-week profile:

    z_t   = (normal_t − x_t) / spread_t        (direction="down", e.g. flow, speed)
    S_t   = max(0, S_{t−1} + z_t − k),  S_0 = 0
    onset = first t with S_t ≥ h

"Normal" is the median for that weekday class and time-of-day slot, so the
daily rush hour is normal and never counts as a disruption. An observed
departure only counts as a disruption if it is SUSTAINED (>= 15 min by default)
and SUBSTANTIAL (>= 15 % below normal on average); shorter or shallower dips
are ordinary fluctuation.

Dependencies: numpy, pandas. matplotlib only for the plot_* helpers.
Self-test:    python tta.py --selftest
Guide:        see HOW_TO_RUN.md
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "TTAConfig", "Profile", "TTAResult",
    "build_profile", "evaluate", "summary_table", "compare", "ranking_agreement",
    "windows_to_valid_time",
    "plot_event", "plot_adaptation_curve", "plot_error_vs_tta", "plot_horizon", "plot_tta_far",
]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TTAConfig:
    """All times are in MINUTES; they are converted to steps of your data."""
    direction: str = "down"        # "down": disruption = values below normal (flow, speed)
                                   # "up":   disruption = values above normal (occupancy, travel time)
    k: float = 0.5                 # CUSUM slack, in spread units
    h_catalogue: float = 5.0       # threshold on the OBSERVED series (defines events)
    h_report: float = 5.0          # threshold on PREDICTED series for the main table
    h_sweep: tuple = (2.0, 3.0, 5.0, 8.0, 12.0)   # pred thresholds for the TTA-vs-FAR curve
    z_cap: float | None = 4.0      # winsorise |z| so one bad interval cannot fire alone
    warmup_min: float = 30.0       # operator runs this long before the window (data must exist)
    max_lead_min: float = 10.0     # a pred onset may precede the observed onset by at most this
    max_wait_min: float = 60.0     # censoring cap (data must exist this long after the window)
    fa_grace_min: float = 15.0     # pred alarm within this after an observed episode is not false
    # What counts as a DISRUPTION (event catalogue). Short or shallow dips are
    # ordinary fluctuation: a model that ignores them is right to do so.
    min_event_duration_min: float = 15.0   # observed episode must last at least this long
    min_event_depth: float = 0.15          # mean relative departure from normal over the first
                                           # depth_window_min of the episode (0.15 = 15 %)
    depth_window_min: float = 15.0
    recovery_min: float = 5.0              # traffic counts as recovered after this long back at normal
    min_normal_level: float | None = None  # only count disruptions (and false alarms) where the NORMAL
                                           # level at that detector and time is at least this, in your
                                           # units (e.g. 50 veh/5 min). A 15 % dip in a near-empty road
                                           # is noise, not a disruption. None = no filter.
    n_boot: int = 2000             # bootstrap resamples over events
    seed: int = 0


# =============================================================================
# Historical profile ("what is normal for this time of week")
# =============================================================================

@dataclass
class Profile:
    median: np.ndarray             # [n_keys, N]
    spread: np.ndarray             # [n_keys, N] robust std (1.4826 * MAD), floored
    bin_min: int
    day_classes: str
    holidays: tuple = ()

    def keys(self, times) -> np.ndarray:
        return _profile_keys(pd.DatetimeIndex(times), self.bin_min, self.day_classes, self.holidays)

    def lookup(self, times) -> tuple[np.ndarray, np.ndarray]:
        k = self.keys(times)
        return self.median[k], self.spread[k]


def _dt_minutes(times: pd.DatetimeIndex) -> float:
    d = np.diff(times.values).astype("timedelta64[s]").astype(float) / 60.0
    if d.size == 0:
        raise ValueError("need at least two timestamps")
    dt = float(np.median(d))
    if not np.allclose(d, dt):
        warnings.warn(f"time grid is not regular ({int((~np.isclose(d, dt)).sum())} irregular steps); "
                      f"TTA assumes a regular grid of {dt:g} min.")
    return dt


def _profile_keys(times: pd.DatetimeIndex, bin_min: int, day_classes: str, holidays) -> np.ndarray:
    per_day = 1440 // bin_min
    tod = (times.hour * 60 + times.minute) // bin_min
    hol = times.normalize().isin(pd.to_datetime(list(holidays))) if len(holidays) else np.zeros(len(times), bool)
    if day_classes == "weekday_weekend":
        cls = ((times.weekday >= 5) | hol).astype(int)
    elif day_classes == "day_of_week":
        cls = np.where(hol, 6, times.weekday)
    else:
        raise ValueError("day_classes must be 'weekday_weekend' or 'day_of_week'")
    return np.asarray(cls) * per_day + np.asarray(tod)


def build_profile(history: np.ndarray, history_times, *, bin_min: int | None = None,
                  day_classes: str = "weekday_weekend", holidays: Iterable = (),
                  exclude_dates: Iterable = (), zero_is_missing: bool = False,
                  min_samples: int = 8, spread_floor_rel: float = 0.05,
                  spread_floor_abs: float = 1.0) -> Profile:
    """
    history       : [T, N] the SAME quantity, units and aggregation as your model
                    target (e.g. 5-min flow sums, or speed), NaN = missing.
                    Typically the training period (or a full year).
    history_times : length-T timestamps.
    bin_min       : time-of-day bin (default = data resolution).
    exclude_dates : dates to leave out (e.g. the test days themselves).
    zero_is_missing: True if your raw files store gaps as 0.
    spread_floor_abs : smallest spread allowed, in YOUR units (e.g. 1 veh, 1 km/h).
    """
    x = np.asarray(history, dtype=np.float32).copy()
    if x.ndim == 1:
        x = x[:, None]
    times = pd.DatetimeIndex(history_times)
    if bin_min is None:
        bin_min = max(1, int(round(_dt_minutes(times))))
    if zero_is_missing:
        x[x == 0] = np.nan
    key = _profile_keys(times, bin_min, day_classes, tuple(holidays))
    excl = pd.to_datetime(list(exclude_dates)) if len(list(exclude_dates)) else pd.DatetimeIndex([])
    keep = ~times.normalize().isin(excl)

    n_keys = (7 if day_classes == "day_of_week" else 2) * (1440 // bin_min)
    N = x.shape[1]
    med = np.full((n_keys, N), np.nan, np.float32)
    spr = np.full((n_keys, N), np.nan, np.float32)
    idx = np.flatnonzero(keep)
    order = idx[np.argsort(key[idx], kind="stable")]
    bounds = np.searchsorted(key[order], np.arange(n_keys + 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for b in range(n_keys):
            rows = order[bounds[b]:bounds[b + 1]]
            if rows.size < 2:
                continue
            v = x[rows]
            m = np.nanmedian(v, axis=0)
            mad = np.nanmedian(np.abs(v - m), axis=0)
            m[np.sum(np.isfinite(v), axis=0) < min_samples] = np.nan
            med[b], spr[b] = m, 1.4826 * mad
    floor = np.maximum(spread_floor_rel * np.abs(np.nan_to_num(med)), spread_floor_abs)
    spr = np.where(np.isfinite(med), np.maximum(np.nan_to_num(spr), floor), np.nan).astype(np.float32)
    return Profile(med, spr, bin_min, day_classes, tuple(str(d) for d in holidays))


# =============================================================================
# Onset operator (CUSUM)
# =============================================================================

def _z(x, mu, sd, cfg: TTAConfig) -> np.ndarray:
    """Deviation from normal in spread units, positive = towards a disruption."""
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (mu - x) / sd if cfg.direction == "down" else (x - mu) / sd
    if cfg.z_cap is not None:
        z = np.clip(z, -cfg.z_cap, cfg.z_cap)
    return z


def _cusum(x, mu, sd, cfg: TTAConfig) -> np.ndarray:
    z = _z(x, mu, sd, cfg)
    inc = np.nan_to_num(z - cfg.k, nan=0.0)          # missing -> no evidence either way
    S = np.empty(inc.shape, np.float32)
    s = np.zeros(inc.shape[1])
    for t in range(inc.shape[0]):
        s = np.maximum(0.0, s + inc[t])              # increment may be negative: S decays & resets
        S[t] = s
    return S


def _episodes(S: np.ndarray, h: float) -> list[list[tuple[int, int]]]:
    """Per node: (onset, end). Onset = crossing of h while armed; end = S back to 0."""
    W, N = S.shape
    eps: list[list[tuple[int, int]]] = [[] for _ in range(N)]
    armed = np.ones(N, bool)
    cur = np.full(N, -1, int)
    for t in range(W):
        fire = armed & (S[t] >= h)
        cur[fire] = t
        armed &= ~fire
        reset = (~armed) & (S[t] <= 0.0)
        for n in np.flatnonzero(reset):
            eps[n].append((int(cur[n]), t))
        cur[reset] = -1
        armed |= reset
    for n in np.flatnonzero(~armed):
        eps[n].append((int(cur[n]), W))
    return eps


def _first_at_or_after(S: np.ndarray, h: float, start: np.ndarray, node: np.ndarray) -> np.ndarray:
    """For each (start, node): smallest t >= start with S[t, node] >= h, else W.
    Memory O(events) instead of O(W x N), so it scales to a year of 1-minute data."""
    W = S.shape[0]
    out = np.full(start.shape, W, np.int64)
    for n in np.unique(node):
        sel = node == n
        above = np.flatnonzero(S[:, n] >= h)
        if above.size:
            k = np.searchsorted(above, start[sel])
            ok = k < above.size
            res = np.full(k.shape, W, np.int64)
            res[ok] = above[k[ok]]
            out[sel] = res
    return out


def _next_at_or_above(S: np.ndarray, h: float) -> np.ndarray:
    W, N = S.shape
    nxt = np.full((W + 1, N), W, np.int64)
    above = S >= h
    for t in range(W - 1, -1, -1):
        nxt[t] = np.where(above[t], t, nxt[t + 1])
    return nxt[:W]


# =============================================================================
# Helpers for getting predictions onto the valid-time grid
# =============================================================================

def windows_to_valid_time(preds: np.ndarray, target_index: np.ndarray, T: int) -> list[np.ndarray]:
    """
    Convert window-shaped model output to one valid-time series per horizon.

    preds        : [n_windows, H, N] predictions (physical units)
    target_index : [n_windows, H] grid index each prediction is VALID at,
                   e.g. issue_idx[:, None] + np.arange(1, H + 1)[None, :]
    T            : length of the observation grid
    returns      : list of H arrays [T, N], NaN where no prediction exists
    """
    preds = np.asarray(preds)
    n_w, H, N = preds.shape
    out = []
    for h in range(H):
        g = np.full((T, N), np.nan, np.float32)
        ti = np.asarray(target_index)[:, h]
        ok = (ti >= 0) & (ti < T)
        g[ti[ok]] = preds[ok, h]
        out.append(g)
    return out


# =============================================================================
# Main entry point
# =============================================================================

@dataclass
class TTAResult:
    """Long-format results; several results (scenarios) can be pooled."""
    per_event: pd.DataFrame        # one row per (model, horizon, h_pred, event)
    alarms: pd.DataFrame           # false alarms and detector-days per (model, horizon, h_pred)
    errors: pd.DataFrame           # error sums per (model, horizon) for exact pooling
    events: pd.DataFrame           # the event catalogue
    config: dict
    series: dict = field(default_factory=dict, repr=False)   # for plot_event (not pooled)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({k: getattr(self, k).to_dict(orient="records")
                       for k in ("per_event", "alarms", "errors", "events")} | {"config": self.config},
                      f, indent=1, default=str)


def evaluate(obs: np.ndarray, times, predictions: Mapping[str, Mapping[str, tuple]],
             profile: Profile, window: tuple | None = None, cfg: TTAConfig | None = None,
             scenario: str = "scenario", node_ids: Sequence | None = None,
             add_baselines: bool = True) -> TTAResult:
    """
    obs         : [T, N] observed target, physical units, on a regular time grid.
                  Must cover [window_start − warmup, window_end + max_wait].
    times       : length-T timestamps of that grid.
    predictions : {model_name: {horizon_label: (pred[T, N], lead_min)}}
                  pred[t] is the prediction VALID at times[t] (NaN if none).
                  lead_min = time from the last observation the model could use
                  to the valid time (persistence has TTA ≈ lead_min).
    profile     : from build_profile() on history of the same quantity.
    window      : (start, end) — disruptions whose observed onset falls in
                  [start, end) are scored; errors are computed on it too.
                  None (default) = the whole series, minus the warm-up at the
                  start and the censoring cap at the end.
    add_baselines: adds "Persistence" (obs shifted by each lead) and
                  "HistProfile" (the profile median) reference rows.
    """
    cfg = cfg or TTAConfig()
    obs = np.asarray(obs, np.float32)
    if obs.ndim == 1:
        obs = obs[:, None]
    times = pd.DatetimeIndex(times)
    T, N = obs.shape
    dt = _dt_minutes(times)
    steps = lambda minutes: int(round(minutes / dt))
    node_ids = list(node_ids) if node_ids is not None else list(range(N))

    if window is None:
        ev_lo = min(T, steps(cfg.warmup_min))
        ev_hi = max(ev_lo, T - steps(cfg.max_wait_min))
    else:
        ev_lo = int(np.searchsorted(times, pd.Timestamp(window[0])))
        ev_hi = int(np.searchsorted(times, pd.Timestamp(window[1])))
    if ev_hi <= ev_lo:
        raise ValueError(f"[{scenario}] nothing to score: the series is shorter than warm-up + cap "
                         f"({cfg.warmup_min:g} + {cfg.max_wait_min:g} min)")
    g0 = max(0, ev_lo - steps(cfg.warmup_min))
    g1 = min(T, ev_hi + steps(cfg.max_wait_min))
    if ev_lo - g0 < steps(cfg.warmup_min):
        warnings.warn(f"[{scenario}] only {(ev_lo - g0) * dt:g} min of warm-up data before the window "
                      f"(want {cfg.warmup_min:g}); pass a longer obs series.")
    if g1 - ev_hi < steps(cfg.max_wait_min):
        warnings.warn(f"[{scenario}] only {(g1 - ev_hi) * dt:g} min of data after the window "
                      f"(want {cfg.max_wait_min:g}); late reactions will be censored early.")
    lo, hi, W = ev_lo - g0, ev_hi - g0, g1 - g0

    zero_share = float(np.mean(obs[np.isfinite(obs)] == 0)) if np.isfinite(obs).any() else 0.0
    if cfg.direction == "down" and zero_share > 0.01:
        warnings.warn(f"[{scenario}] {100 * zero_share:.1f}% of observed values are exactly 0. If 0 means "
                      f"'missing' in your data (e.g. METR-LA / PEMS-BAY speeds), set them to NaN first, "
                      f"or every gap will look like a disruption.")
    mu_full, sd_full = profile.lookup(times)
    mu, sd, x_obs = mu_full[g0:g1], sd_full[g0:g1], obs[g0:g1]
    valid = np.isfinite(mu).any(0) & np.isfinite(sd).any(0)

    # ---- event catalogue (observations only) ---------------------------------
    S_obs = _cusum(x_obs, mu, sd, cfg)
    obs_eps = _episodes(S_obs, cfg.h_catalogue)
    # Duration = how long the TRAFFIC is abnormal: from the first abnormal step of
    # the episode (where the tally started rising) until the series has been back
    # at normal for recovery_min. Not the tally's own episode, which also includes
    # the time the tally needs to drain after traffic has already recovered.
    z_obs = _z(x_obs, mu, sd, cfg)
    ev_node, ev_on, ev_dur, ev_depth, ongoing = [], [], [], [], 0
    n_candidates = n_short = n_shallow = n_low = 0
    sgn = 1.0 if cfg.direction == "down" else -1.0
    dur_s, depth_w = steps(cfg.min_event_duration_min), max(1, steps(cfg.depth_window_min))
    rec_s = max(1, steps(cfg.recovery_min))
    for n in np.flatnonzero(valid):
        if not obs_eps[n]:
            continue
        abnormal = np.nan_to_num(z_obs[:, n], nan=0.0) > cfg.k
        zero = np.flatnonzero(S_obs[:, n] <= 0.0)
        for (o, e) in obs_eps[n]:
            if lo <= o < hi:
                n_candidates += 1
                if cfg.min_normal_level is not None and not (np.nan_to_num(mu[o, n]) >= cfg.min_normal_level):
                    n_low += 1
                    continue
                j = np.searchsorted(zero, o) - 1          # last minute the tally was 0
                s0 = zero[j] + 1 if j >= 0 else 0
                ok = ~abnormal[o:min(W, e + rec_s)]
                if ok.size >= rec_s:
                    run = np.convolve(ok.astype(np.int8), np.ones(rec_s, np.int8), "valid") == rec_s
                    k_rec = int(np.argmax(run)) if run.any() else ok.size
                else:
                    k_rec = ok.size
                dur = o + k_rec - s0
                if dur < dur_s:
                    n_short += 1
                    continue
                w = slice(o, min(W, o + depth_w))
                with np.errstate(invalid="ignore", divide="ignore"):
                    rel = sgn * (mu[w, n] - x_obs[w, n]) / np.abs(mu[w, n])
                depth = float(np.nanmean(rel)) if np.isfinite(rel).any() else np.nan
                if not np.isfinite(depth) or depth < cfg.min_event_depth:
                    n_shallow += 1
                    continue
                ev_node.append(n); ev_on.append(o); ev_dur.append(dur * dt); ev_depth.append(depth)
            elif o < lo < e:
                ongoing += 1
    ev_node, ev_on = np.asarray(ev_node, int), np.asarray(ev_on, int)
    lead_s0, grace_s0 = steps(cfg.max_lead_min), steps(cfg.fa_grace_min)
    explain = {}                       # per node: sorted interval starts and running max of ends
    for n in np.flatnonzero(valid):
        if obs_eps[n]:
            st = np.array([o - lead_s0 for o, _ in obs_eps[n]])
            en = np.maximum.accumulate(np.array([e + grace_s0 for _, e in obs_eps[n]]))
            explain[n] = (st, en)
    events = pd.DataFrame({"scenario": scenario, "node": [node_ids[n] for n in ev_node],
                           "onset": times[g0 + ev_on] if ev_on.size else pd.DatetimeIndex([]),
                           "duration_min": np.asarray(ev_dur, float), "depth": np.asarray(ev_depth, float)})
    if ongoing:
        warnings.warn(f"[{scenario}] {ongoing} detector(s) were already disrupted when the window opened "
                      f"and are not scored; consider starting the window earlier.")

    # ---- assemble series to score -------------------------------------------
    series: dict[str, dict[str, tuple[np.ndarray, float]]] = {}
    if add_baselines:
        leads = sorted({float(v[1]) for m in predictions.values() for v in m.values()}) or [dt]
        series["Persistence"] = {}
        for L in leads:
            p = np.full_like(obs, np.nan)
            s = steps(L)
            if s > 0:
                p[s:] = obs[:-s]
            else:
                p[:] = obs
            series["Persistence"][f"lead {L:g}min"] = (p, L)
        series["HistProfile"] = {"any": (mu_full.astype(np.float32), leads[0])}
    for m, hz in predictions.items():
        series[m] = {str(k): (np.asarray(v[0], np.float32).reshape(T, -1), float(v[1])) for k, v in hz.items()}

    h_list = sorted(set(cfg.h_sweep) | {cfg.h_report})
    lead_s, wait_s, grace_s = steps(cfg.max_lead_min), steps(cfg.max_wait_min), steps(cfg.fa_grace_min)
    if cfg.min_normal_level is not None:
        det_days = float(np.sum(np.nan_to_num(mu[lo:hi][:, valid]) >= cfg.min_normal_level)) * dt / 1440.0
    else:
        det_days = float(valid.sum()) * (hi - lo) * dt / 1440.0
    pe_rows, al_rows, er_rows = [], [], []

    for model, hz in series.items():
        for hlabel, (pred_full, L) in hz.items():
            if pred_full.shape != obs.shape:
                raise ValueError(f"{model}/{hlabel}: prediction shape {pred_full.shape} != obs {obs.shape}")
            pred = pred_full[g0:g1]

            # errors on the window (sums, so pooling across scenarios is exact)
            yt, yp = x_obs[lo:hi], pred[lo:hi]
            ok = np.isfinite(yt) & np.isfinite(yp)
            e = (yt - yp)[ok]
            nz = ok & (yt != 0)
            er_rows.append(dict(scenario=scenario, model=model, horizon=hlabel, lead_min=L,
                                n=int(ok.sum()), sum_abs=float(np.abs(e).sum()), sum_sq=float((e ** 2).sum()),
                                n_ape=int(nz.sum()),
                                sum_ape=float(np.abs((yt[nz] - yp[nz]) / yt[nz]).sum()),
                                coverage=float(ok.sum() / max(1, np.isfinite(yt).sum()))))

            S = _cusum(pred, mu, sd, cfg)
            for hp in h_list:
                start = np.maximum(ev_on - lead_s, 0)
                p = _first_at_or_after(S, hp, start, ev_node) if ev_on.size else np.zeros(0, int)
                # prediction already in alarm max_lead before the observed onset:
                # not anticipation (it was signalling a disruption that had not
                # happened yet). Kept out of the TTA statistics, reported as a share.
                early = (p == start) & (ev_on - lead_s >= 0) & (S[start, ev_node] >= hp) \
                    & (S[np.maximum(start - 1, 0), ev_node] >= hp)
                tta_steps = (p - ev_on).astype(float)
                cens = ((p >= W) | (tta_steps > wait_s)) & ~early
                tta_min = np.where(cens, cfg.max_wait_min, tta_steps * dt)
                tta_min = np.where(early, np.nan, tta_min)
                if ev_on.size:
                    pe_rows.append(pd.DataFrame({
                        "scenario": scenario, "model": model, "horizon": hlabel, "lead_min": L, "h_pred": hp,
                        "node": [node_ids[n] for n in ev_node], "onset": times[g0 + ev_on],
                        "tta_min": tta_min.astype(float), "censored": cens.astype(bool),
                        "pre_alarmed": early.astype(bool)}))
                n_fa = 0
                for n_i, eps in enumerate(_episodes(S, hp)):
                    if not valid[n_i] or not eps:
                        continue
                    pon = np.array([o for o, _ in eps])
                    pon = pon[(pon >= lo) & (pon < hi)]
                    if cfg.min_normal_level is not None and pon.size:
                        pon = pon[np.nan_to_num(mu[pon, n_i]) >= cfg.min_normal_level]
                    if pon.size == 0:
                        continue
                    if n_i not in explain:
                        n_fa += int(pon.size)
                        continue
                    st, en = explain[n_i]
                    k = np.searchsorted(st, pon, side="right") - 1
                    ok = k >= 0
                    explained = np.zeros(pon.size, bool)
                    explained[ok] = pon[ok] <= en[k[ok]]
                    n_fa += int((~explained).sum())
                al_rows.append(dict(scenario=scenario, model=model, horizon=hlabel, lead_min=L,
                                    h_pred=hp, false_alarms=n_fa, detector_days=det_days))

    cols = ["scenario", "model", "horizon", "lead_min", "h_pred", "node", "onset",
            "tta_min", "censored", "pre_alarmed"]
    per_event = pd.concat(pe_rows, ignore_index=True) if pe_rows else pd.DataFrame(columns=cols)
    keep_series = {"times": times[g0:g1], "obs": x_obs, "mu": mu, "sd": sd, "node_ids": node_ids,
                   "pred": {(m, h): v[0][g0:g1] for m, hz in series.items() for h, v in hz.items()}}
    return TTAResult(per_event, pd.DataFrame(al_rows), pd.DataFrame(er_rows), events,
                     asdict(cfg) | {"dt_min": dt, "n_detectors_scored": int(valid.sum()),
                                    "n_ongoing_excluded": ongoing,
                                    "n_candidate_episodes": n_candidates,
                                    "n_dropped_low_level": n_low,
                                    "n_dropped_short": n_short, "n_dropped_shallow": n_shallow,
                                    "events_per_detector_day": float(len(ev_on) / det_days) if det_days else np.nan},
                     keep_series)


# =============================================================================
# Summaries (pool scenarios by passing a list of results)
# =============================================================================

def _as_list(results) -> list[TTAResult]:
    return list(results) if isinstance(results, (list, tuple)) else [results]


def _boot_ci(x: np.ndarray, n_boot: int, seed: int, stat=np.median) -> tuple[float, float]:
    if x.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = stat(x[rng.integers(0, x.size, (n_boot, x.size))], axis=1)
    return tuple(float(v) for v in np.percentile(b, [2.5, 97.5]))


def summary_table(results, h_pred: float | str | None = None) -> pd.DataFrame:
    """
    One row per (model, horizon): error measures and TTA side by side.
    Pass a list of results to pool scenarios: events are pooled (not medians
    averaged), error sums are pooled. h_pred=None -> cfg.h_report; "all" -> every sweep value.
    """
    rs = _as_list(results)
    cfg = rs[0].config
    pe = pd.concat([r.per_event for r in rs], ignore_index=True)
    al = pd.concat([r.alarms for r in rs], ignore_index=True)
    er = pd.concat([r.errors for r in rs], ignore_index=True)
    hs = sorted(pe["h_pred"].unique()) if h_pred == "all" else [cfg["h_report"] if h_pred is None else h_pred]

    order = list(dict.fromkeys(er["model"]))
    rows = []
    for (model, horizon), eg in er.groupby(["model", "horizon"], sort=False):
        L = float(eg["lead_min"].iloc[0])
        n = eg["n"].sum()
        err = dict(MAE=eg["sum_abs"].sum() / n if n else np.nan,
                   RMSE=np.sqrt(eg["sum_sq"].sum() / n) if n else np.nan,
                   MAPE=100 * eg["sum_ape"].sum() / eg["n_ape"].sum() if eg["n_ape"].sum() else np.nan)
        for hp in hs:
            ev = pe[(pe.model == model) & (pe.horizon == horizon) & (pe.h_pred == hp)]
            a = al[(al.model == model) & (al.horizon == horizon) & (al.h_pred == hp)]
            pre = ev["pre_alarmed"].to_numpy(bool)
            x = ev["tta_min"].to_numpy(float)[~pre]
            cens = ev["censored"].to_numpy(bool)[~pre]
            far = a["false_alarms"].sum() / a["detector_days"].sum() if a["detector_days"].sum() else np.nan
            row = dict(model=model, horizon=horizon, lead_min=L, h_pred=hp, **err,
                       n_events=int(len(ev)), pre_alarmed_pct=100 * float(pre.mean()) if len(ev) else np.nan)
            if x.size:
                med = float(np.median(x))
                ci = _boot_ci(x, cfg["n_boot"], cfg["seed"])
                row.update(TTA_median=med, TTA_q1=float(np.percentile(x, 25)), TTA_q3=float(np.percentile(x, 75)),
                           TTA_ci_lo=ci[0], TTA_ci_hi=ci[1], median_is_lower_bound=bool(cens.mean() >= 0.5),
                           TTA_mean_capped=float(x.mean()),
                           caught_pct=100 * float((~cens).mean()),
                           A_idx=1 - med / L if L > 0 else np.nan, FAR_per_det_day=far)
            else:
                row.update(TTA_median=np.nan, FAR_per_det_day=far)
            rows.append(row)
    out = pd.DataFrame(rows)
    out["_o"] = out["model"].map({m: i for i, m in enumerate(order)})
    return out.sort_values(["_o", "lead_min", "h_pred"]).drop(columns="_o").reset_index(drop=True)


def compare(results, model_a: str, model_b: str, horizon_a: str, horizon_b: str | None = None,
            h_pred: float | None = None) -> dict:
    """
    Paired comparison on the SAME events: is model A faster than model B?
    Returns the median paired difference (A − B, minutes; negative = A faster)
    with a bootstrap CI, and the share of events where each was faster.
    """
    rs = _as_list(results)
    cfg = rs[0].config
    hp = cfg["h_report"] if h_pred is None else h_pred
    pe = pd.concat([r.per_event for r in rs], ignore_index=True)
    pe = pe[pe.h_pred == hp]
    key = ["scenario", "node", "onset"]
    a = pe[(pe.model == model_a) & (pe.horizon == horizon_a)].set_index(key)["tta_min"]
    b = pe[(pe.model == model_b) & (pe.horizon == (horizon_b or horizon_a))].set_index(key)["tta_min"]
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    d = (j["a"] - j["b"]).to_numpy(float)
    if d.size == 0:
        return {"n_events": 0}
    ci = _boot_ci(d, cfg["n_boot"], cfg["seed"])
    return {"n_events": int(d.size), "median_diff_min": float(np.median(d)), "ci_lo": ci[0], "ci_hi": ci[1],
            "A_faster_pct": 100 * float((d < 0).mean()), "tie_pct": 100 * float((d == 0).mean()),
            "B_faster_pct": 100 * float((d > 0).mean())}


def ranking_agreement(table: pd.DataFrame, horizon: str, error: str = "MAE",
                      exclude=("Persistence", "HistProfile")) -> dict:
    """Kendall's tau between the error ranking and the TTA ranking of the models
    at one horizon. tau = 1: TTA adds no new ordering; tau ≤ 0: it reorders the models."""
    t = table[(table.horizon == horizon) & (~table.model.isin(exclude))]
    t = t.drop_duplicates("model")
    if len(t) < 3:
        return {"n_models": len(t), "kendall_tau": np.nan}
    tau = t[[error, "TTA_mean_capped"]].corr(method="kendall").iloc[0, 1]
    return {"n_models": len(t), "kendall_tau": float(tau),
            "rank_by_" + error: list(t.sort_values(error)["model"]),
            "rank_by_TTA": list(t.sort_values("TTA_mean_capped")["model"])}


# =============================================================================
# Plots (matplotlib, optional)
# =============================================================================

_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
_BASE = {"Persistence": dict(color="#6b6b6b", ls="--"), "HistProfile": dict(color="#a8a8a8", ls=":")}


def _style(models: Sequence[str]) -> dict:
    out, i = {}, 0
    for m in models:
        if m in _BASE:
            out[m] = _BASE[m]
        else:
            out[m] = dict(color=_COLORS[i % len(_COLORS)], ls="-")
            i += 1
    return out


def _plt():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
    return plt


def plot_event(result: TTAResult, event: int = 0, models: Sequence[str] | None = None,
               horizon: str | None = None, ax=None, pad_min: float = 45):
    """One event: observed vs predicted series, the normal profile, and each onset.
    This is the picture that explains TTA to a reader."""
    plt = _plt()
    s, cfg = result.series, result.config
    ev = result.events.iloc[event]
    n = s["node_ids"].index(ev["node"])
    times = s["times"]
    t0 = times.get_loc(ev["onset"])
    w = int(round(pad_min / cfg["dt_min"]))
    sl = slice(max(0, t0 - w), min(len(times), t0 + 2 * w))
    ax = ax or plt.subplots(figsize=(8, 3.6))[1]
    tt = times[sl]
    ax.fill_between(tt, s["mu"][sl, n] - s["sd"][sl, n], s["mu"][sl, n] + s["sd"][sl, n],
                    color="#d9d9d9", alpha=0.6, lw=0, label="normal ±1 spread")
    ax.plot(tt, s["obs"][sl, n], color="#1f1f1f", lw=2.4, label="observed", zorder=5)
    ax.axvline(ev["onset"], color="#1f1f1f", lw=1)
    pe = result.per_event
    pe = pe[(pe.h_pred == cfg["h_report"]) & (pe.node == ev["node"]) & (pe.onset == ev["onset"])]
    models = models or [m for m in dict.fromkeys(pe.model) if m != "HistProfile"]
    sty = _style(models)
    for m in models:
        r = pe[pe.model == m]
        if horizon is not None:
            r = r[r.horizon == horizon]
        if r.empty:
            continue
        r = r.iloc[0]
        lab = "already in alarm" if r.pre_alarmed else f"TTA {'≥' if r.censored else ''}{r.tta_min:.0f} min"
        ax.plot(tt, s["pred"][(m, r.horizon)][sl, n], lw=1.6, color=sty[m]["color"], ls=sty[m]["ls"],
                label=f"{m}: {lab}")
        if not (r.censored or r.pre_alarmed):
            ax.axvline(ev["onset"] + pd.Timedelta(minutes=r.tta_min), color=sty[m]["color"], lw=1, ls=sty[m]["ls"])
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title(f"{ev['scenario']} · detector {ev['node']} · observed onset {ev['onset']:%H:%M}", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="best")
    return ax


def plot_adaptation_curve(results, horizon_by_model: Mapping[str, str] | None = None,
                          models: Sequence[str] | None = None, ax=None, h_pred: float | None = None):
    """Share of disruption events each model has registered within x minutes of the
    observed onset (a cumulative curve; censored events never count). Higher and
    further left = adapts faster. Shows the whole distribution, handles censoring."""
    plt = _plt()
    rs = _as_list(results)
    hp = rs[0].config["h_report"] if h_pred is None else h_pred
    cap = rs[0].config["max_wait_min"]
    pe = pd.concat([r.per_event for r in rs], ignore_index=True)
    pe = pe[pe.h_pred == hp]
    models = models or list(dict.fromkeys(pe.model))
    sty = _style(models)
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    grid = np.linspace(-rs[0].config["max_lead_min"], cap, 400)
    for m in models:
        r = pe[pe.model == m]
        if horizon_by_model and m in horizon_by_model:
            r = r[r.horizon == horizon_by_model[m]]
        elif r.horizon.nunique() > 1:
            r = r[r.horizon == r.horizon.iloc[0]]
        r = r[~r.pre_alarmed]
        x = np.where(r.censored, np.inf, r.tta_min)
        if x.size == 0:
            continue
        ax.step(grid, [(x <= g).mean() for g in grid], where="post", lw=2, label=m, **sty[m])
    ax.axvline(0, color="#1f1f1f", lw=0.8)
    ax.set_xlabel("minutes after the observed onset")
    ax.set_ylabel("share of events registered")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_error_vs_tta(table: pd.DataFrame, horizon_by_model: Mapping[str, str] | None = None,
                      error: str = "MAE", ax=None, include_baselines: bool = True,
                      y: str = "TTA_mean_capped"):
    """Each model as a point: x = error, y = TTA. If TTA said the same thing as the
    error measure, the points would lie on a rising line. Default y is the mean TTA
    with never-registered events counted at the cap, which stays finite and
    comparable even when many events are censored; y="TTA_median" shows the
    median with its IQR."""
    plt = _plt()
    t = table.copy()
    if horizon_by_model:
        t = t[[horizon_by_model.get(m, h) == h for m, h in zip(t.model, t.horizon)]]
    t = t.drop_duplicates("model")
    if not include_baselines:
        t = t[~t.model.isin(_BASE)]
    sty = _style(list(t.model))
    ax = ax or plt.subplots(figsize=(5.5, 4.2))[1]
    offsets = [(6, 4), (6, -10), (-6, 8), (-6, -12)]
    for i, (_, r) in enumerate(t.sort_values(error).iterrows()):
        c = sty[r.model]["color"]
        if y == "TTA_median":
            ax.errorbar(r[error], r.TTA_median, yerr=[[r.TTA_median - r.TTA_q1], [r.TTA_q3 - r.TTA_median]],
                        fmt="o", ms=8, color=c, ecolor=c, elinewidth=1, capsize=0)
        else:
            ax.plot(r[error], r[y], "o", ms=8, color=c)
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(r.model, (r[error], r[y]), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                    ha="left" if dx > 0 else "right")
    ax.set_xlabel(f"{error} (lower is better)")
    ax.set_ylabel(("median TTA" if y == "TTA_median" else "mean TTA, never-adapted at cap")
                  + ", min (lower is better)")
    return ax


def plot_horizon(table: pd.DataFrame, models: Sequence[str] | None = None, axes=None):
    """(a) median TTA vs lead time with the persistence diagonal (TTA = lead) and
    full anticipation (TTA = 0); (b) anticipation index vs lead time."""
    plt = _plt()
    models = models or list(dict.fromkeys(table.model))
    models = [m for m in models if m != "HistProfile"]
    sty = _style(models)
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    Lmax = table.lead_min.max()
    axes[0].plot([0, Lmax], [0, Lmax], color="#6b6b6b", ls="--", lw=1, label="echo (TTA = lead)")
    axes[0].axhline(0, color="#1f1f1f", lw=0.8)
    for m in models:
        r = table[table.model == m].sort_values("lead_min")
        if m == "Persistence":
            continue
        axes[0].plot(r.lead_min, r.TTA_median, marker="o", ms=6, lw=2, label=m, **sty[m])
        axes[1].plot(r.lead_min, r.A_idx, marker="o", ms=6, lw=2, label=m, **sty[m])
    axes[1].axhline(0, color="#6b6b6b", ls="--", lw=1)
    axes[1].axhline(1, color="#1f1f1f", lw=0.8)
    axes[0].set(xlabel="lead time (min)", ylabel="median TTA (min)", title="(a) Time to Adapt")
    axes[1].set(xlabel="lead time (min)", ylabel="anticipation index", title="(b) Lead time used")
    axes[0].legend(frameon=False, fontsize=8)
    return axes


def plot_tta_far(table_all_h: pd.DataFrame, horizon_by_model: Mapping[str, str] | None = None, ax=None):
    """Median TTA vs false-alarm rate, traced by sweeping the prediction threshold
    (use summary_table(results, h_pred='all')). A model is faster than another only
    if its curve lies below over the false-alarm range of interest."""
    plt = _plt()
    t = table_all_h[table_all_h.model != "HistProfile"]
    models = list(dict.fromkeys(t.model))
    sty = _style(models)
    ax = ax or plt.subplots(figsize=(5.5, 4.2))[1]
    for m in models:
        r = t[t.model == m]
        h = horizon_by_model.get(m) if horizon_by_model else r.horizon.iloc[0]
        r = r[r.horizon == h].sort_values("FAR_per_det_day")
        ax.plot(r.FAR_per_det_day, r.TTA_median, marker="o", ms=5, lw=2, label=m, **sty[m])
    ax.set_xlabel("false alarms per detector-day")
    ax.set_ylabel("median TTA (min)")
    ax.legend(frameon=False, fontsize=8)
    return ax


# =============================================================================
# Self-test on synthetic data with known answers
# =============================================================================

def _selftest(make_plots: bool = False) -> None:
    rng = np.random.default_rng(0)
    dt, N = 1, 12
    times = pd.date_range("2025-03-03", periods=21 * 1440, freq="min")      # 3 weeks of history+test
    tod = (times.hour * 60 + times.minute).to_numpy()
    base = 60 + 40 * np.exp(-((tod - 8 * 60) / 90.0) ** 2) + 45 * np.exp(-((tod - 17 * 60) / 100.0) ** 2)
    base = np.where(times.weekday >= 5, 0.7 * base, base)
    y = base[:, None] * rng.uniform(0.8, 1.2, N)[None, :] + rng.normal(0, 3, (len(times), N))

    # test day: a capacity drop at 16:30 propagating upstream 2 min per detector, lasting 50 min
    test_day = pd.Timestamp("2025-03-19")
    on0 = times.get_loc(test_day + pd.Timedelta("16:30:00"))
    for n in range(8):
        y[on0 + 2 * n: on0 + 2 * n + 50, n] *= 0.55
    profile = build_profile(y, times, exclude_dates=[test_day])

    sl = slice(times.get_loc(test_day + pd.Timedelta("10:00:00")), times.get_loc(test_day + pd.Timedelta("21:00:00")))
    obs, tt = y[sl], times[sl]
    mu, _ = profile.lookup(tt)

    def lagged(lag):
        p = np.full_like(obs, np.nan); p[lag:] = obs[:-lag]; return p

    dev = obs - mu                          # "Slow": knows the normal pattern, but takes
    sdev = np.zeros_like(dev)               # ~25 min (time constant) to absorb a deviation
    for t in range(1, len(dev)):
        sdev[t] = sdev[t - 1] + (dev[t - 1] - sdev[t - 1]) / 25.0
    slow = mu + sdev
    preds = {"Perfect": {"t+5": (obs.copy(), 5.0)},
             "Lag-5": {"t+5": (lagged(5), 5.0)},
             "Slow": {"t+5": (slow, 5.0)},
             "ProfileOnly": {"t+5": (mu.copy(), 5.0)}}
    res = evaluate(obs, tt, preds, profile, window=(test_day + pd.Timedelta("11:00:00"),
                                                    test_day + pd.Timedelta("20:00:00")),
                   scenario="synthetic")
    tab = summary_table(res)
    print(f"events found: {len(res.events)} (8 disruptions injected)")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tab[["model", "horizon", "MAE", "n_events", "TTA_median", "TTA_q1", "TTA_q3",
                   "caught_pct", "pre_alarmed_pct", "A_idx", "FAR_per_det_day"]].round(2).to_string(index=False))
    get = lambda m: tab[tab.model == m].iloc[0]
    true_on = {n: test_day + pd.Timedelta("16:30:00") + pd.Timedelta(minutes=2 * n) for n in range(8)}
    found = {n for n, o in zip(res.events.node, res.events.onset)
             if n in true_on and pd.Timedelta(0) <= o - true_on[n] <= pd.Timedelta(minutes=5)}
    assert found == set(range(8)), f"injected disruptions not all detected: {sorted(found)}"
    assert get("Perfect").TTA_median == 0 and get("Perfect").caught_pct == 100
    assert get("Lag-5").TTA_median == 5 and abs(get("Persistence").TTA_median - 5) <= 1
    assert get("ProfileOnly").caught_pct == 0 and get("HistProfile").caught_pct == 0
    assert get("Slow").TTA_mean_capped > get("Lag-5").TTA_mean_capped
    print("compare Lag-5 vs Slow:", {k: (round(v, 2) if isinstance(v, float) else v)
                                     for k, v in compare(res, "Lag-5", "Slow", "t+5").items()})
    print("self-test passed")
    if make_plots:
        plt = _plt()
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        plot_event(res, event=3, models=["Perfect", "Lag-5", "Slow", "ProfileOnly"], ax=axs[0, 0])
        plot_adaptation_curve(res, ax=axs[0, 1])
        plot_error_vs_tta(tab, ax=axs[1, 0])
        plot_tta_far(summary_table(res, h_pred="all"), ax=axs[1, 1])
        fig.tight_layout()
        fig.savefig("tta_selftest.png", dpi=130)
        print("wrote tta_selftest.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Time to Adapt (TTA) metric")
    ap.add_argument("--selftest", action="store_true", help="run the synthetic check")
    ap.add_argument("--plots", action="store_true", help="also save tta_selftest.png")
    a = ap.parse_args()
    if a.selftest:
        _selftest(a.plots)
    else:
        ap.print_help()