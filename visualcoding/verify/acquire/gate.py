"""Acceptance gate for an acquired-instrument candidate: raw sandbox rows -> answers (adapters) -> range rule
(fit_rule with the model's declared witnesses) -> leave-one-task-out -> verdict + feedback material."""
from __future__ import annotations

import collections

import numpy as np

from .. import instruments as I
from . import adapters as A
from . import fit_rule as F

DEFAULTS = dict(min_precision=0.95, min_cover=3, max_atoms=2, loo_precision=0.90, stability=0.80,
                min_margin=0.02,     # v2: numeric thresholds must not hug an observed value (gap >= 2 % of the range)
                gain_tier=True, gain_precision=0.5, gain_min_cover=5, gain_alpha=0.05,   # GAIN tier (see decide_gain)
                gain_stability=0.5,   # the gain objective is flat (many rules tie), so its rule flips more across LOO folds than a 0.95 rule's
                gain_objective="cover")  # fit objective of the gain-tier rule: 'cover' (most readings at precision >= 0.5; the retrospective's
                                         # 'gain' objective bought 22 more acceptances only through threshold-hugging atoms) or 'gain'


def _hard_ok(atom) -> bool:
    """Hard witnesses (`require`) may only be boolean or string atoms: a numeric threshold declared as 'hard' is a
    fitted threshold that would bypass the leave-one-out check (seen: symbolic `n_valid >= 12`)."""
    return atom[1] in ("==", "!=") and isinstance(atom[2], (bool, str))


def rows_from_raw(raw: list[dict], tasks: dict, convention: str = "identity") -> list[dict]:
    """instrument_eval-style rows (value, answer, correct, diag, gt) from runner rows; answers via the adapters."""
    rows = []
    for r in raw:
        t = tasks[r["task_id"]]
        ans = A.answer_from_value(r.get("value"), t, convention) if not r.get("error") else None
        rows.append({"task_id": t.task_id, "source": t.source, "family": t.family, "sample": 0, "value": r.get("value"),
                     "answer": ans, "correct": A.is_correct(ans, t), "in_range": None, "diag": r.get("diag") or {},
                     "n_calls": r.get("n_calls", 0), "gt_key": A.gt_key(t), "gt_value": I.gt_value(t),
                     "error": r.get("error"), "elapsed": r.get("elapsed"), "contents": r.get("contents", []),
                     "debug": r.get("debug") or [], "checks": r.get("checks") or {}, "tokens": r.get("tokens") or {}})
    return rows


def records(rows: list[dict], tasks: dict, eye: dict) -> list[dict]:
    recs = []
    for r in rows:
        t = tasks[r["task_id"]]
        cs = eye.get(r["task_id"])
        recs.append({"task_id": r["task_id"], "sample": 0, "feats": F.features_from_row(r, t), "correct": r["correct"],
                     "has_answer": r["answer"] is not None, "stored_in_range": False,
                     "eye": (sum(c["correct"] for c in cs) / len(cs)) if cs else None})
    return recs


def hints(recs: list[dict], min_precision: float = 0.95, min_cover: int = 3, top: int = 6) -> list[str]:
    """Single atoms (over all features incl. value / opt_*) that keep >= min_cover readings at >= min_precision:
    what a witness would have to look like.  Feedback only; the gate uses the declared features."""
    n = len(recs)
    if not n:
        return []
    has = [bool(r["has_answer"]) for r in recs]
    right = [r["correct"] == 1 for r in recs]
    out = []
    for a in F._atoms(recs, None, True, True):
        sel = [has[i] and F._atom_true(a, recs[i]["feats"]) for i in range(n)]
        c = sum(sel)
        if c < min_cover:
            continue
        rr = sum(1 for i in range(n) if sel[i] and right[i])
        if rr / c >= min_precision:
            out.append((c, rr / c, a))
    out.sort(key=lambda x: (-x[0], -x[1]))
    seen, res = set(), []
    for c, p, a in out:
        if a[0] in seen:
            continue
        seen.add(a[0])
        res.append(f"{F._atoms_str([a])} -> {c} in range, precision {p:.2f}")
        if len(res) >= top:
            break
    return res


def decide(fit: dict, lo: dict, cfg: dict) -> tuple[bool, str]:
    if fit["rule"] is None:
        return False, f"no rule keeps >= {cfg['min_cover']} readings at precision >= {cfg['min_precision']}"
    if lo["precision"] is None:
        if fit["train"]["cover"] <= cfg["min_cover"] and fit["train"]["wrong"] == 0:
            # a route with only min_cover train tasks can never be reproduced under leave-one-out (2 < 3 left);
            # that is a lack of evidence, not a wrong measurement (symbolic rings route: 3/3 train, 4/4 val)
            return False, (f"insufficient evidence: the rule covers only {fit['train']['cover']} train readings (= min_cover), "
                           "leave-one-out cannot test it; not a rejection of the measurement - more train tasks on this "
                           "route, or coverage of another template, would be needed")
        return False, "leave-one-out selected nothing"
    if lo["precision"] < cfg["loo_precision"]:
        return False, f"leave-one-out precision {lo['precision']:.2f} < {cfg['loo_precision']}"
    if lo["rule_stability"] < cfg["stability"]:
        return False, f"rule stability {lo['rule_stability']:.2f} < {cfg['stability']} (thresholds hug single tasks)"
    return True, "ok"


def gain_bound(pairs: list, alpha: float = 0.05, n_boot: int = 2000, seed: int = 0) -> dict:
    """One-sided (1 - alpha) lower confidence bound of the mean paired gain (instrument correct - executor pass@1) over
    the leave-one-out in-range readings that have an executor reference; bootstrap over readings, deterministic."""
    d = np.array([c - e for c, e in pairs if e is not None], dtype=float)
    if len(d) == 0:
        return {"n": 0, "mean": None, "lower": None}
    rng = np.random.RandomState(seed)
    means = d[rng.randint(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return {"n": int(len(d)), "mean": float(d.mean()), "lower": float(np.quantile(means, alpha))}


def decide_gain(fit: dict, lo: dict, cfg: dict) -> tuple[bool, str, dict]:
    """GAIN tier: a weaker, explicitly labelled claim used only when no certified rule exists - in range the
    instrument is right >= gain_precision of the time (leave-one-out) AND beats the executor's pass@1 on the same
    train tasks (one-sided 95 % bootstrap lower bound of the paired LOO gain > 0, >= gain_min_cover readings).  It is
    a selection claim ('better than the fallback'), not a certificate ('right'); the loop prefers a certified rule."""
    gs = gain_bound(lo.get("pairs") or [], cfg["gain_alpha"])
    if fit["rule"] is None:
        return False, f"no rule keeps >= {cfg['gain_min_cover']} readings at precision >= {cfg['gain_precision']}", gs
    if lo["precision"] is None or lo["cover"] < cfg["gain_min_cover"]:
        return False, f"leave-one-out keeps {lo['cover']} readings (< {cfg['gain_min_cover']})", gs
    if lo["precision"] < cfg["gain_precision"]:
        return False, f"leave-one-out precision {lo['precision']:.2f} < {cfg['gain_precision']}", gs
    if lo["rule_stability"] < cfg["gain_stability"]:
        return False, f"rule stability {lo['rule_stability']:.2f} < {cfg['gain_stability']}", gs
    if gs["n"] < cfg["gain_min_cover"]:
        return False, f"only {gs['n']} in-range readings have an executor reference", gs
    if gs["lower"] <= 0:
        return False, (f"gain over the executor {gs['mean']:+.2f} per in-range reading (LOO), one-sided 95 % lower bound "
                       f"{gs['lower']:+.2f} <= 0"), gs
    return True, (f"ok - GAIN tier: LOO precision {lo['precision']:.2f} on {lo['cover']} readings, gain over the executor "
                  f"{gs['mean']:+.2f} per reading (lower bound {gs['lower']:+.2f})"), gs


def evaluate(rows: list[dict], tasks: dict, eye: dict, declared: dict | None = None, hypothesis: dict | None = None,
             **over) -> dict:
    """`hypothesis` = hypothesis.check() result; when its verdict is False, stage B (rule fit + LOO) is skipped and the
    candidate is rejected with that reason (playbook fix the measurement before any threshold).  Routes that failed
    stage A on their own (hypothesis['routes_fail']) are excluded from stage B.  Stage B has two tiers: certified
    (decide) and, when that fails and cfg['gain_tier'], GAIN (decide_gain); `tier` names the one that accepted."""
    cfg = {**DEFAULTS, **over}
    declared = declared or {}
    require, dropped = [], []
    for a in declared.get("require", []) or []:
        try:
            atom = F.parse_atom(a) if isinstance(a, str) else tuple(a)
        except ValueError:
            dropped.append(str(a))
            continue
        (require if _hard_ok(atom) else dropped).append(atom if _hard_ok(atom) else str(a))
    group_by = declared.get("route_key") or ("route" if any(isinstance((r["diag"] or {}).get("route"), str) for r in rows) else None)
    features = set(declared.get("features") or []) or None
    level = (hypothesis or {}).get("level") or ("certified" if hypothesis is None or hypothesis.get("verdict") else None)
    excl = set((hypothesis or {}).get("routes_fail_gain" if level == "gain" else "routes_fail") or {})
    rkey = (hypothesis or {}).get("route_key") or group_by or "route"
    rows_b = [r for r in rows if not (excl and (r.get("diag") or {}).get(rkey) in excl)]
    recs = records(rows_b, tasks, eye)
    kw = dict(min_precision=cfg["min_precision"], min_cover=cfg["min_cover"], max_atoms=cfg["max_atoms"],
              group_by=group_by, require=require, features=features, min_margin=cfg["min_margin"])
    tier, cert_why, gain_why, gs, gain_eval = None, None, None, None, None
    empty_fit = {"rule": None, "train": {"cover": 0, "right": 0, "wrong": 0, "precision": None, "gain": 0.0}, "searched": 0}
    empty_lo = {"cover": 0, "right": 0, "wrong": 0, "precision": None, "gain": 0.0, "n_tasks": len({r["task_id"] for r in rows}),
                "rule_stability": 0.0, "pairs": []}

    def _gain_tier():
        kwg = dict(kw, min_precision=cfg["gain_precision"], min_cover=cfg["gain_min_cover"], objective=cfg["gain_objective"])
        fit_g, lo_g = F.fit(recs, **kwg), F.loo(recs, **kwg)
        okg, why_g, gs_g = decide_gain(fit_g, lo_g, cfg)
        ge = {"rule_str": F.rule_str(fit_g["rule"]), "train": fit_g["train"], "searched": fit_g["searched"],
              "loo": {k: v for k, v in lo_g.items() if k != "pairs"}, "stats": gs_g, "why": why_g, "accepted": okg}
        return okg, why_g, gs_g, ge, fit_g, lo_g

    if level is None:
        fit, lo = empty_fit, empty_lo
        why = f"stage A failed ({hypothesis.get('why')}): stage B skipped"
    elif level == "gain":
        # below the 0.6 bar but at the gain level -> only the GAIN tier is tried (the certified tier needs stage A)
        fit, lo = empty_fit, empty_lo
        cert_why = f"stage A below {hypothesis.get('h_min')} ({hypothesis.get('why', '').split(' - ')[0]}): certified tier unavailable"
        if cfg["gain_tier"]:
            okg, gain_why, gs, gain_eval, fit_g, lo_g = _gain_tier()
            if okg:
                tier, fit, lo, why = "gain", fit_g, lo_g, f"{cert_why}; {gain_why}"
            else:
                why = f"{cert_why}; gain tier: {gain_why}"
        else:
            why = f"stage A failed ({hypothesis.get('why')}): stage B skipped"
    else:
        fit = F.fit(recs, **kw)
        lo = F.loo(recs, **kw)
        ok, cert_why = decide(fit, lo, cfg)
        if ok:
            tier, why = "certified", cert_why
        elif cfg["gain_tier"]:
            okg, gain_why, gs, gain_eval, fit_g, lo_g = _gain_tier()
            if okg:
                tier, fit, lo, why = "gain", fit_g, lo_g, f"certified tier failed ({cert_why}); {gain_why}"
            else:
                why = f"certified: {cert_why}; gain tier: {gain_why}"
        else:
            why = cert_why
    accepted = tier is not None
    parsed = [r for r in rows if r["answer"] is not None]
    n_right = sum(1 for r in parsed if r["correct"] == 1)
    errors = [r for r in rows if r.get("error")]
    eyes = [r["eye"] for r in recs if r["eye"] is not None]
    return {"n_tasks": len(rows), "parsed": len(parsed), "parsed_right": n_right,
            "parsed_acc": (n_right / len(parsed)) if parsed else None,
            "errors": len(errors), "timeouts": sum(1 for r in errors if "timeout" in str(r["error"])),
            "n_calls": sum(r.get("n_calls", 0) for r in rows),
            "rule": fit["rule"], "rule_str": F.rule_str(fit["rule"]), "train": fit["train"], "searched": fit["searched"],
            "loo": lo, "accepted": accepted, "why": why, "group_by": group_by, "require": [list(a) for a in require],
            "require_dropped": dropped, "gate_version": 4, "min_margin": cfg["min_margin"], "hypothesis": hypothesis,
            "insufficient": (cert_why or "").startswith("insufficient evidence"),
            "tier": tier, "certified": tier == "certified", "cert_why": cert_why, "gain_why": gain_why, "gain_stats": gs,
            "gain_tier": bool(cfg["gain_tier"]), "excluded_routes": sorted(excl), "gain_eval": gain_eval, "level": level,
            "features": sorted(features) if features else None, "eye_mean": (sum(eyes) / len(eyes)) if eyes else None,
            "hints": hints(recs, cfg["min_precision"], cfg["min_cover"])}


def score(ev: dict) -> tuple:
    """Candidate ranking: accepted first - certified above GAIN tier, then LOO gain; among the rest the
    HYPOTHESIS decides (stage A verdict, number of parsed readings that agree, agreement rate) before any rule-based
    number (train gain, coverage)."""
    h = ev.get("hypothesis") or {}
    rank = {"certified": 2, "gain": 1}.get(ev.get("tier"), 2 if ev["accepted"] else 0)
    return (rank, ev["loo"].get("gain", 0.0) if ev["accepted"] else 0.0,
            int(bool(h.get("verdict", ev["parsed_acc"] is not None and ev["parsed_acc"] >= 0.6))),
            h.get("right", ev["parsed_right"]), h.get("agreement", ev["parsed_acc"] or 0.0),
            ev["train"].get("gain", 0.0), ev["train"]["cover"])
