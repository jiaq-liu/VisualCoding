#!/usr/bin/env python
"""GEPA baseline — reflective prompt evolution, tool-free arm.

GEPA itself is used unmodified (its clone needs no install: `dependencies = []`, and the
LiteLLM path is bypassed because `reflection_lm` accepts any callable).  What we supply is

  * a GEPAAdapter that evaluates a candidate prompt with one tool-free VLM call per rollout
    and scores it with our judge (common.vlm -> common/vlm_cli.py under the main environment's interpreter),
  * a reflective dataset carrying the model's answer, the gold answer and the judge's note,
  * the reflection LM (gemini-3.1-pro through google-genai).

Protocol: per family, no held-out val, 4 rollouts per task while training, and SINGLE PASS
by default (`--passes 0`): the seed prompt is evaluated once on the family's train set --
that pass is the shared, content-addressed tool-free seed pass, so it is paid for once
across GEPA and TextGrad -- and GEPA's own reflective proposer
(`InstructionProposalSignature`, the same prompt template the search uses) is called once on
those traces.  The proposal is accepted; there is no candidate re-evaluation and no search.

`--passes N` restores the published evolutionary search: budget = (N + 1) x |train| metric
calls, pareto candidate selection, accept-if-better.  That is the faithful GEPA and it costs
roughly N extra passes over the train set at 4 rollouts each; single pass is what makes this
baseline comparable in budget to the one-shot skill-induction methods.

  python drivers/run_gepa.py train  --families all                      # single pass
  python drivers/run_gepa.py train  --families all --passes 8           # the search
  python drivers/run_gepa.py deploy --families all --split test          # 1 rollout, frozen prompt
  python drivers/run_gepa.py control --families all --split test         # shared no-skill control
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

BASELINES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASELINES))
sys.path.insert(0, str(BASELINES / "gepa" / "src"))

from common import data, driver, gemini, vlm                      # noqa: E402
from common.paths import MODEL                                    # noqa: E402

METHOD = "gepa"
ARM = "no_tool"
COMPONENT = "task_instruction"
SEED_INSTRUCTION = ("Answer the question about the image. Look at the image carefully before "
                    "deciding, and put the final answer inside <answer></answer> tags.")
MAX_OUTPUT_CHARS = 1500


class VQAAdapter:
    """GEPAAdapter: candidate {COMPONENT: text} -> tool-free rollouts -> scores + reflective records."""

    # GEPA reads this attribute to decide who owns instruction proposal; None = GEPA's own
    # reflection path (the reflection_lm below).  It must exist, not merely be absent.
    propose_new_texts = None

    def __init__(self, rollouts: int = 4, model: str = MODEL, workers: int = 8, log=print):
        self.rollouts, self.model, self.workers, self.log = rollouts, model, workers, log
        self.calls_new = 0          # rollouts that actually hit the API
        self.calls_total = 0

    def evaluate(self, batch, candidate, capture_traces: bool = False):
        from gepa.core.adapter import EvaluationBatch
        prompt = candidate[COMPONENT]
        rep = vlm.run(batch, prompt, rollouts=self.rollouts, model=self.model,
                      workers=self.workers, log=self.log)
        self.calls_new += int(rep.get("to_run", 0))
        self.calls_total += len(batch) * self.rollouts
        rows = vlm.read(rep["out_dir"], batch, self.rollouts)
        outputs, scores, trajs = [], [], []
        for rec in batch:
            rs = rows.get(rec["doc_id"], [])
            score = sum(float(r["accuracy_score"]) for r in rs) / len(rs) if rs else 0.0
            answers = [str(r.get("final_answer") or "") for r in rs]
            outputs.append({"answers": answers, "text": (rs[0].get("trajectory_text") if rs else "")})
            scores.append(score)
            if capture_traces:
                trajs.append({"question": rec.get("problem", ""), "gold": rec.get("solution", ""),
                              "answers": answers, "score": score, "family": rec.get("family", ""),
                              "text": (rs[0].get("trajectory_text") if rs else ""),
                              "judge": (rs[0].get("trajectory_analysis") if rs else "")})
        return EvaluationBatch(outputs=outputs, scores=scores,
                               trajectories=trajs if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = []
        for t in (eval_batch.trajectories or []):
            verdict = "correct" if t["score"] >= 1.0 else ("partly correct" if t["score"] > 0 else "wrong")
            fb = (f"The answer is {verdict} (score {t['score']:.2f}). "
                  f"The correct answer is {t['gold']!r}; the model answered {t['answers']}.")
            if t.get("judge"):
                fb += f" Judge: {str(t['judge'])[:400]}"
            records.append({
                "Inputs": {"question": str(t["question"])[:1200], "image": "(one image, shown to the model)"},
                "Generated Outputs": str(t["text"])[:MAX_OUTPUT_CHARS],
                "Feedback": fb,
            })
        return {c: records for c in components_to_update}


def _reflection_batch(records: list[dict], k: int, seed: int) -> list[dict]:
    """Up to `k` reflective records, failures first: the examples that carry signal.

    GEPA always reflects on a minibatch rather than the whole set, so this is the shape of
    its own input; the ordering is the one deviation, and it is the one that matters when a
    single call has to stand in for a search.
    """
    if k <= 0 or len(records) <= k:
        return records
    bad = [r for r in records if "wrong" in r["Feedback"] or "partly" in r["Feedback"]]
    good = [r for r in records if r not in bad]
    rng = random.Random(seed)
    rng.shuffle(bad)
    rng.shuffle(good)
    out = bad[:k]
    return out + good[: k - len(out)]


def train_single_pass(fam: data.Family, args, log=print) -> dict:
    """One seed evaluation (reused) + one call to GEPA's reflective proposer."""
    from gepa.strategies.instruction_proposal import InstructionProposalSignature

    d = driver.family_dir(METHOD, fam)
    recs = fam.records("train")
    adapter = VQAAdapter(rollouts=args.rollouts, model=args.model, workers=args.workers, log=log)
    t0 = time.time()
    eb = adapter.evaluate(recs, {COMPONENT: SEED_INSTRUCTION}, capture_traces=True)
    seed_acc = sum(eb.scores) / len(eb.scores) if eb.scores else 0.0
    records = adapter.make_reflective_dataset({COMPONENT: SEED_INSTRUCTION}, eb, [COMPONENT])[COMPONENT]
    batch = _reflection_batch(records, args.reflect_k, args.seed)
    log(f"   seed train acc {seed_acc:.3f}; reflecting on {len(batch)}/{len(records)} records")

    lm = gemini.Gemini(model=args.reflection_model, temperature=1.0, max_tokens=8192)
    out = InstructionProposalSignature.run(lm, {"current_instruction_doc": SEED_INSTRUCTION,
                                                "dataset_with_feedback": batch,
                                                "prompt_template": None})
    text = (out.get("new_instruction") or "").strip() or SEED_INSTRUCTION
    (d / "reflection.json").write_text(json.dumps(
        {"seed_train_acc": seed_acc, "n_records": len(records), "n_reflected": len(batch),
         "new_instruction": text}, indent=1, ensure_ascii=False))
    driver.Artifact(
        method=METHOD, family=fam.key, arm=ARM, text=text, model=args.model, seed=args.seed,
        train_tasks=len(recs), rollouts_used=adapter.calls_total, rollouts_new=adapter.calls_new,
        lm_calls=gemini.USAGE["n_calls"],
        notes={"seed_instruction": SEED_INSTRUCTION, "mode": "single_pass",
               "reflect_k": args.reflect_k, "reflection_model": args.reflection_model,
               "seed_train_acc": seed_acc, "wall_s": round(time.time() - t0)}).save()
    return {"family": fam.key, "n_train": len(recs), "rollouts_new": adapter.calls_new,
            "seed_acc": round(seed_acc, 4), "chars": len(text), "wall_s": round(time.time() - t0)}


def train_family(fam: data.Family, args, log=print) -> dict:
    import gepa
    d = driver.family_dir(METHOD, fam)
    d.mkdir(parents=True, exist_ok=True)
    if (d / "artifact.json").is_file() and not args.force:
        return {"family": fam.key, "skipped": "artifact exists"}
    recs = fam.records("train")
    if args.passes <= 0:
        if args.dry_run:
            return {"family": fam.key, "n_train": len(recs), "mode": "single_pass",
                    "metric_calls": len(recs), "rollouts": len(recs) * args.rollouts,
                    "reflection_calls": 1, "dry_run": True}
        return train_single_pass(fam, args, log=log)
    budget = (args.passes + 1) * len(recs)
    if args.dry_run:
        return {"family": fam.key, "n_train": len(recs), "metric_calls": budget,
                "rollouts": budget * args.rollouts, "dry_run": True}

    adapter = VQAAdapter(rollouts=args.rollouts, model=args.model, workers=args.workers, log=log)
    reflection_lm = gemini.Gemini(model=args.reflection_model, temperature=1.0, max_tokens=8192)
    t0 = time.time()
    result = gepa.optimize(
        seed_candidate={COMPONENT: SEED_INSTRUCTION},
        trainset=recs, valset=None,                     # no held-out val: selection signal = train
        adapter=adapter, reflection_lm=reflection_lm,
        max_metric_calls=budget, reflection_minibatch_size=args.minibatch,
        candidate_selection_strategy="pareto", seed=args.seed,
        run_dir=str(d / "gepa_state"), display_progress_bar=False,
        raise_on_exception=False,
    )
    best = (result.best_candidate or {}).get(COMPONENT, SEED_INSTRUCTION)
    art = driver.Artifact(
        method=METHOD, family=fam.key, arm=ARM, text=best, model=args.model, seed=args.seed,
        train_tasks=len(recs), rollouts_used=adapter.calls_total, rollouts_new=adapter.calls_new,
        lm_calls=gemini.USAGE["n_calls"],
        notes={"seed_instruction": SEED_INSTRUCTION, "max_metric_calls": budget,
               "minibatch": args.minibatch, "reflection_model": args.reflection_model,
               "best_score": getattr(result, "best_score", None), "wall_s": round(time.time() - t0)})
    art.save()
    (d / "gepa_result.json").write_text(json.dumps(
        {"best_score": getattr(result, "best_score", None),
         "num_candidates": len(getattr(result, "candidates", []) or []),
         "val_aggregate_scores": getattr(result, "val_aggregate_scores", None)}, indent=1, default=str))
    return {"family": fam.key, "n_train": len(recs), "rollouts_new": adapter.calls_new,
            "best_score": getattr(result, "best_score", None), "wall_s": round(time.time() - t0)}


def main(argv=None) -> int:
    ap = driver.parser(METHOD, ARM)
    ap.add_argument("--passes", type=int, default=0,
                    help="0 = single pass (one reflective proposal, no search); "
                         "N > 0 runs the published search on a budget of (N+1) x |train| "
                         "metric calls")
    ap.add_argument("--reflect-k", type=int, default=24,
                    help="single-pass only: reflective records per family, failures first")
    ap.add_argument("--minibatch", type=int, default=5, help="GEPA reflection minibatch size")
    ap.add_argument("--reflection-model", default=MODEL)
    ap.add_argument("--which", default="best", choices=["best", "seed"], help="deploy the optimized or the seed prompt")
    args = ap.parse_args(argv)
    fams = driver.select_families(args.families)[: args.limit]
    log = print

    if args.verb == "status":
        print(json.dumps(driver.status(METHOD, fams), indent=1))
        return 0
    rollouts = args.rollouts if args.rollouts is not None else (4 if args.verb == "train" else 1)
    args.rollouts = rollouts
    reports = []
    for i, fam in enumerate(fams, 1):
        log(f"[{i}/{len(fams)}] {args.verb} {fam.key}")
        if args.verb == "train":
            rep = train_family(fam, args, log=log)
        elif args.verb == "deploy":
            text = SEED_INSTRUCTION if args.which == "seed" else None
            rep = (vlm.run(fam.records(args.split), text, rollouts=rollouts, model=args.model,
                           workers=args.workers, dry_run=args.dry_run, log=log) if text else
                   driver.deploy_family(METHOD, fam, ARM, args.split, rollouts=rollouts,
                                        workers=args.workers, model=args.model, dry_run=args.dry_run, log=log))
        else:
            rep = driver.control_family(fam, ARM, args.split, rollouts=rollouts, workers=args.workers,
                                        model=args.model, dry_run=args.dry_run, log=log)
        rep["family"] = fam.key
        reports.append(rep)
        log("   " + json.dumps({k: v for k, v in rep.items() if k != "map"}))
    out = driver.method_root(METHOD) / (
        f"_{args.verb}_report.json" if args.verb == "train" else
        f"_{args.verb}_{args.split}_{driver.model_tag(args.model)}_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=1, default=str))
    tot_new = sum(int(r.get("rollouts_new") or r.get("to_run") or 0) for r in reports)
    log(f"\n{args.verb}: {len(reports)} families, new rollouts {tot_new}; report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
