"""GTA VI trophy earn-rate forecaster, Python 3.13, standard library only.

The rate Exophase prints is earners / tracked players. Both move: the tracked
pool grows as new PSN profiles are linked (which dilutes every rate), while the
number of players who actually earned a trophy grows through the week. We model
those two forces separately instead of extrapolating the printed rate directly:

    earners_i(t) ~= rate_i(t) * players_tracked(t) / 100
    log earners_i(t) = a_i + b_i * log(1 + hours since release)
    log tracked(t)   = A   + B   * log(1 + hours since release)

Then rate_i(target) = 100 * earners_i(target) / tracked(target). Slopes are
shrunk toward a pooled estimate because a case has only a handful of distinct
Exophase refreshes. Quantiles are placed on the logit scale with a spread built
from the fit residuals, the slope uncertainty and the extrapolation distance.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader runs with -P, so the directory holding this file is off sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# The rules record that Exophase's first refresh, at 04:30 UTC on launch day,
# moved the tracked pool from 638 to 3,868 with every rate falling. Practice
# cases stop just before it, so with no visible refresh we use that as the prior.
PRIOR_N_BEFORE = 638.0
PRIOR_N_AFTER = 3868.0


def clamp(x, lo, hi):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def expit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def hours_between(a, b):
    """ Hours from ISO-8601 a to b. """
    from datetime import datetime
    ta = datetime.fromisoformat(a.replace("Z", "+00:00"))
    tb = datetime.fromisoformat(b.replace("Z", "+00:00"))
    return (tb - ta).total_seconds() / 3600.0


def wls(xs, ys, ws):
    """ Weighted least squares y = intercept + slope * x. Returns (a, b, slope_se). """
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return (ys[0] if ys else 0.0), 0.0, 1.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx < 1e-9:
        return my, 0.0, 1.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    s2 = sum(w * r * r for w, r in zip(ws, resid)) / max(1.0, sw - 2.0)
    return a, b, math.sqrt(max(s2, 0.0) / sxx)


def distinct_events(case):
    """ Collapse consecutive identical reads: Exophase refreshes in steps. """
    events = []
    prev_key = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == prev_key:
            continue
        prev_key = key
        events.append(read)
    return events


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name}: unexpected levels {case['levels']}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name}: no history")

    last = history[-1]
    # Time is measured from the first recorded read, so a case's own clock is enough.
    t0 = case["history"][0]["at"]
    t_last = max(0.0, hours_between(t0, last["at"]))
    t_target = t_last + case["horizon_minutes"] / 60.0
    lx_last = math.log1p(t_last)
    lx_target = math.log1p(t_target)
    dx = lx_target - lx_last

    events = distinct_events(case)
    n_trophies = len(case["trophies"])

    # Late events matter more: a half-life of 30 hours over a two-day history.
    newest_t = max(0.0, hours_between(t0, events[-1]["at"])) if events else t_last
    weights = []
    for ev in events:
        te = max(0.0, hours_between(t0, ev["at"]))
        weights.append(0.35 + 0.65 * math.exp(-(newest_t - te) / 30.0))

    # Prior for a flat history: the rules' documented first refresh.
    if len(events) < 3:
        ratio = PRIOR_N_BEFORE / PRIOR_N_AFTER
        rows = []
        for i, trophy in enumerate(case["trophies"]):
            r = last["rates"][i]
            if r is None or r <= 0:
                r = 0.01
            if r >= 99.99:
                pred = r
                sigma = 0.05
            else:
                pred = clamp(r * ratio, 0.01, 99.99)
                sigma = 0.85
            rows.append({"id": trophy["id"], "q": [round4(clamp(expit(logit(pred) + z * sigma), 0.0, 100.0)) for z in Z]})
        return {"trophies": rows}

    # Track the growth of the tracked-player pool.
    ev_x = [math.log1p(max(0.0, hours_between(t0, ev["at"]))) for ev in events]
    ev_n = [max(1.0, float(ev["players_tracked"])) for ev in events]
    n_a, n_b, n_se = wls(ev_x, [math.log(v) for v in ev_n], weights)
    n_b = clamp(n_b, -0.2, 2.0)
    log_n_last = math.log(max(1.0, float(last["players_tracked"])))
    log_n_target = max(log_n_last, n_a + n_b * lx_target)

    # Per-trophy earner curves, read from the same refresh events.
    slopes = []
    residuals = []
    fitted = []
    for i in range(n_trophies):
        xs, ys, ws = [], [], []
        for ev, x, w in zip(events, ev_x, weights):
            r = ev["rates"][i]
            n = max(1.0, float(ev["players_tracked"]))
            if r is None or r <= 0:
                continue
            earners = r * n / 100.0 + 0.5
            xs.append(x)
            ys.append(math.log(earners))
            ws.append(w)
        if len(xs) >= 2:
            a, b, se = wls(xs, ys, ws)
            res = math.sqrt(sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) /
                            max(1.0, sum(ws) - 2.0))
        else:
            a, b, se, res = (ys[0] if ys else 0.0), 0.5, 0.5, 0.3
        slopes.append(b)
        residuals.append(res)
        fitted.append((a, b, se))

    positive = [b for b in slopes if -3.0 < b < 4.0]
    pool = sorted(positive)[len(positive) // 2] if positive else 0.5
    pool = clamp(pool, -0.5, 1.5)
    median_res = sorted(residuals)[len(residuals) // 2] if residuals else 0.3

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = last["rates"][i]
        if r is None or r <= 0:
            r = 0.01
        if r >= 99.99:
            rows.append({"id": trophy["id"], "q": [99.99] * 5})
            continue

        a, b, se = fitted[i]
        n_obs = sum(1 for ev in events if ev["rates"][i] is not None and ev["rates"][i] > 0)
        shrink = n_obs / (n_obs + 4.0)
        b = clamp(shrink * b + (1.0 - shrink) * pool, -0.75, 1.75)

        log_e_last = math.log(max(0.5, r * float(last["players_tracked"]) / 100.0 + 0.5))
        # Blend the observed last point with the fitted line to damp refresh noise.
        log_e_fit_last = a + b * lx_last
        log_e_last = 0.6 * log_e_last + 0.4 * log_e_fit_last
        log_e_target = log_e_last + b * dx

        # logit(rate in percent) = log(earners) - log(tracked players).
        pred_logit = clamp(log_e_target - log_n_target, -20.0, 20.0)
        pred = clamp(expit(pred_logit), 0.01, 99.99)

        sigma = math.sqrt(residuals[i] ** 2 + (se * dx) ** 2 + 0.30 ** 2)
        sigma = clamp(sigma, 0.20, 1.30)
        q = [round4(clamp(expit(logit(pred) + z * sigma), 0.0, 100.0)) for z in Z]
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"id": trophy["id"], "q": q})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast_case(case, path.name)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
