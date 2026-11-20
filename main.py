"""GTA VI trophy earn-rate forecaster (Python 3.13, standard library only).

Exophase prints rate_i = earners_i / tracked_players. Both sides move: linking
new PSN accounts dilutes every rate, while players earn trophies over the week.
We forecast the two counts separately and divide:

    earners_i(t) = rate_i(t) * tracked(t) / 100
    log earners_i, log tracked and log(total trophies earned) are fit against
    log(1 + hours since release) across the distinct Exophase refreshes in the
    case, shrunk toward a prior for the trophy's tier and rarity band.

Quantiles sit on the logit scale. Their spread combines each trophy's fit
residual, slope uncertainty, and the uncertainty of extrapolating the player
pool, so the five levels stay honest without being needlessly wide.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# The rules record Exophase's first refresh: 638 -> 3,868 tracked players at
# 04:30 UTC on launch night, with every rate falling. Practice cases stop just
# before it, so a flat history uses that as its prior.
PRIOR_N_BEFORE = 638.0
PRIOR_N_AFTER = 3868.0
# The recency-weighted fit already follows the recent growth rate, so the log-log
# slope is carried forward without any extra damping.
DAMP = 1.0


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
    ta = datetime.fromisoformat(a.replace("Z", "+00:00"))
    tb = datetime.fromisoformat(b.replace("Z", "+00:00"))
    return (tb - ta).total_seconds() / 3600.0


def wls(xs, ys, ws):
    """Weighted least squares; returns (intercept, slope, slope standard error)."""
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
    """Consecutive identical reads are one Exophase refresh."""
    events = []
    prev = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != prev:
            events.append(read)
            prev = key
    return events


def group_key(trophy, rate):
    """Tier and rarity band give a prior for a trophy's growth slope."""
    band = 0 if rate < 0.1 else (1 if rate < 1.0 else 2)
    return (trophy["tier"], band)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name}: unexpected levels {case['levels']}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name}: no history")

    last = history[-1]
    t0 = history[0]["at"]
    t_last = max(0.0, hours_between(t0, last["at"]))
    t_target = t_last + case["horizon_minutes"] / 60.0
    lx_last = math.log1p(t_last)
    dx = math.log1p(t_target) - lx_last
    trophies = case["trophies"]
    n_trophies = len(trophies)

    events = distinct_events(case)
    newest = max(0.0, hours_between(t0, events[-1]["at"])) if events else t_last
    weights = [0.3 + 0.7 * math.exp(-(newest - max(0.0, hours_between(t0, ev["at"]))) / 30.0)
               for ev in events]
    ev_x = [math.log1p(max(0.0, hours_between(t0, ev["at"]))) for ev in events]

    # Flat pre-refresh history: use the launch-night prior from the rules.
    if len(events) < 3:
        ratio = PRIOR_N_BEFORE / PRIOR_N_AFTER
        rows = []
        for i, trophy in enumerate(trophies):
            r = last["rates"][i]
            if r is None or r <= 0:
                r = 0.01
            if r >= 99.99:
                rows.append({"id": trophy["id"], "q": [99.99] * 5})
                continue
            pred = clamp(r * ratio, 0.01, 99.99)
            q = [round4(clamp(expit(logit(pred) + z * 0.25), 0.0, 100.0)) for z in Z]
            rows.append({"id": trophy["id"], "q": q})
        return {"trophies": rows}

    # Forecast the tracked pool and the total number of trophies earned.
    _, b_n, se_n = wls(ev_x, [math.log(max(1.0, float(ev["players_tracked"]))) for ev in events], weights)
    b_n = clamp(b_n, -0.2, 1.6)
    n_last = max(1.0, float(last["players_tracked"]))
    log_n_target = math.log(n_last) + DAMP * b_n * dx

    # Per-trophy earner curves.
    slopes, residuals, fitted, observed = [], [], [], []
    late_cut = math.log1p(max(0.0, newest - 24.0)) if newest else 0.0
    for i in range(n_trophies):
        xs, ys, ws = [], [], []
        for ev, x, w in zip(events, ev_x, weights):
            r, n = ev["rates"][i], float(ev["players_tracked"])
            if r is None or r <= 0:
                continue
            xs.append(x)
            ys.append(math.log(r * n / 100.0 + 0.5))
            ws.append(w)
        if len(xs) >= 2:
            a, b, se = wls(xs, ys, ws)
            res = math.sqrt(sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) /
                            max(1.0, sum(ws) - 2.0))
        else:
            a, b, se, res = (ys[0] if ys else 0.0), 0.5, 0.5, 0.3

        # When earner growth decelerates, carry the recent slope forward rather
        # than the slope fitted across the whole history.
        late = [(x, y, w) for x, y, w in zip(xs, ys, ws) if x >= late_cut]
        if len(late) >= 2 and len(xs) - len(late) >= 2:
            _, b_late, _ = wls([p[0] for p in late], [p[1] for p in late], [p[2] for p in late])
            if b_late < b:
                b = 0.5 * b + 0.5 * b_late
                res = max(res, 0.10)
        slopes.append(b)
        residuals.append(res)
        fitted.append((a, b, se))
        observed.append(len(xs))

    groups = {}
    for i, trophy in enumerate(trophies):
        r = last["rates"][i]
        if r is None or r <= 0:
            continue
        groups.setdefault(group_key(trophy, r), []).append(slopes[i])
    priors = {k: sorted(v)[len(v) // 2] for k, v in groups.items()}
    all_priors = sorted(priors.values())
    fallback = all_priors[len(all_priors) // 2] if all_priors else 0.5

    raw = []
    for i, trophy in enumerate(trophies):
        r = last["rates"][i]
        if r is None or r <= 0:
            r = 0.01
        a, b, se = fitted[i]
        prior = priors.get(group_key(trophy, r), fallback)
        w = observed[i] / (observed[i] + 4.0)
        b = clamp(w * b + (1.0 - w) * prior, -0.75, 1.75)
        log_e_last = math.log(max(0.5, r * float(last["players_tracked"]) / 100.0 + 0.5))
        log_e_last = 0.6 * log_e_last + 0.4 * (a + b * lx_last)
        raw.append(math.exp(log_e_last + DAMP * b * dx))

    rows = []
    for i, trophy in enumerate(trophies):
        rate = last["rates"][i]
        if rate is None or rate <= 0:
            rate = 0.01
        if rate >= 99.99:
            rows.append({"id": trophy["id"], "q": [99.99] * 5})
            continue
        a, b, se = fitted[i]
        rate_target = clamp(100.0 * raw[i] / math.exp(log_n_target), 0.01, 99.99)
        pred = logit(rate_target)

        sigma = math.sqrt(residuals[i] ** 2 + (se * DAMP * dx) ** 2 +
                          (0.06 * dx) ** 2 + 0.12 ** 2)
        sigma = clamp(sigma, 0.15, 1.20)
        q = [round4(clamp(expit(pred + z * sigma), 0.0, 100.0)) for z in Z]
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
