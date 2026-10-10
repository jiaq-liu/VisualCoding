#!/usr/bin/env python
"""TextGrad baseline — textual-gradient prompt optimization, tool-free arm.

The system prompt of a vision-language model is the trainable Variable; each training
example runs MultimodalLLMCall(image, question) under the current prompt, an LLM evaluator
turns (question, gold, response) into a textual loss, and TextualGradientDescent edits the
prompt.  Everything runs on gemini-3.1-pro through the google-genai engine added to the
clone (`textgrad/engine/gemini_genai.py`).

SINGLE PASS by default: one epoch, no measurement, and the prompt that the pass produces is
the artifact.  Every example is seen exactly once (the optimizer still steps once per batch
inside the pass, which is TextGrad's own granularity), so the training budget matches the
one-shot skill-induction baselines.  No held-out val either -- the paper script's
`--run_validation` is off by default -- so this is default-accept end to end.

`--measure-rollouts K` (K > 0) restores the safety net: the prompt is measured on train with
the shared tool-free arm after each epoch and the best-scoring epoch is kept.  That is worth
a small train-side cost across the families at K=1 and guards against shipping a drifted prompt; it is off
because it is a validation signal by another name.

  python drivers/run_textgrad.py train  --families all                    # single pass
  python drivers/run_textgrad.py train  --families all --epochs 3 --measure-rollouts 4
  python drivers/run_textgrad.py deploy --families all --split test
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

BASELINES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASELINES))
sys.path.insert(0, str(BASELINES / "textgrad"))

from common import data, driver, vlm                                  # noqa: E402
from common.paths import IMAGE_ROOT, MODEL, gemini_api_key            # noqa: E402

METHOD = "textgrad"
ARM = "no_tool"
SEED_INSTRUCTION = ("Answer the question about the image. Look at the image carefully before "
                    "deciding, and put the final answer inside <answer></answer> tags.")
EVAL_INSTRUCTION = ("Below is a question about an image, the ground-truth answer, and a model's "
                    "response. Decide whether the response gives the correct final answer. "
                    "Explain briefly what the response got right or wrong, then output the "
                    "accuracy as a single token, 1 or 0, between <ACCURACY> and </ACCURACY>.")
ROLE_DESCRIPTIONS = ["question about an image", "ground-truth answer",
                     "response from the vision-language model"]


def _engine(model: str):
    import textgrad as tg
    os.environ.setdefault("GEMINI_API_KEY", gemini_api_key())
    eng = tg.get_engine(model)
    tg.set_backward_engine(eng, override=True)
    return eng


def _image_bytes(rec: dict) -> bytes:
    return (Path(IMAGE_ROOT) / rec["images"][0]).read_bytes()


def train_family(fam: data.Family, args, log=print) -> dict:
    import textgrad as tg
    from textgrad.autograd import MultimodalLLMCall
    from textgrad.loss import MultiFieldTokenParsedEvaluation

    d = driver.family_dir(METHOD, fam)
    d.mkdir(parents=True, exist_ok=True)
    if (d / "artifact.json").is_file() and not args.force:
        return {"family": fam.key, "skipped": "artifact exists"}
    recs = fam.records("train")
    if args.dry_run:
        return {"family": fam.key, "n_train": len(recs), "epochs": args.epochs,
                "forward_calls": args.epochs * len(recs),
                "measure_rollouts": (args.epochs + 1) * len(recs) * args.measure_rollouts,
                "dry_run": True}

    eng = _engine(args.model)
    prompt_var = tg.Variable(SEED_INSTRUCTION, requires_grad=True,
                             role_description="instructions appended to the system prompt of a "
                                              "vision-language model that answers questions about images")
    optimizer = tg.TextualGradientDescent(engine=eng, parameters=[prompt_var])
    eval_fn = MultiFieldTokenParsedEvaluation(
        tg.Variable(EVAL_INSTRUCTION, requires_grad=False, role_description="evaluation instruction"),
        role_descriptions=ROLE_DESCRIPTIONS, engine=eng, parse_tags=["<ACCURACY>", "</ACCURACY>"])

    def measure(text: str) -> float | None:
        """Train accuracy of a prompt, or None when measurement is off (the default)."""
        if args.measure_rollouts <= 0:
            return None
        rep = vlm.run(recs, text, rollouts=args.measure_rollouts, model=args.model,
                      workers=args.workers, log=log)
        s = vlm.mean_scores(rep["out_dir"], recs, args.measure_rollouts)
        return sum(s.values()) / len(s) if s else 0.0

    t0 = time.time()
    history = [{"epoch": 0, "prompt": prompt_var.value, "train_acc": measure(prompt_var.value)}]
    if history[0]["train_acc"] is not None:
        log(f"   seed train acc {history[0]['train_acc']:.3f}")
    rng_order = list(range(len(recs)))
    for epoch in range(1, args.epochs + 1):
        for start in range(0, len(recs), args.batch_size):
            batch = [recs[i] for i in rng_order[start:start + args.batch_size]]
            optimizer.zero_grad()
            losses = []
            for rec in batch:
                q = tg.Variable(rec["problem"].replace("<image>", "").strip(), requires_grad=False,
                                role_description=ROLE_DESCRIPTIONS[0])
                gold = tg.Variable(str(rec["solution"]), requires_grad=False,
                                   role_description=ROLE_DESCRIPTIONS[1])
                img = tg.Variable(_image_bytes(rec), requires_grad=False, role_description="image")
                try:
                    resp = MultimodalLLMCall(eng, system_prompt=prompt_var)([img, q])
                    losses.append(eval_fn([q, gold, resp]))
                except Exception as e:                                  # noqa: BLE001
                    log(f"   [{fam.key}] example {rec['doc_id']} failed: {type(e).__name__}: {e}")
            if not losses:
                continue
            tg.sum(losses).backward()
            optimizer.step()
        acc = measure(prompt_var.value)
        history.append({"epoch": epoch, "prompt": prompt_var.value, "train_acc": acc})
        log(f"   epoch {epoch}: train acc {acc:.3f}" if acc is not None
            else f"   epoch {epoch}: {len(prompt_var.value)} chars (not measured)")

    # Measurement off (single pass) => the artifact is what the pass produced, unmeasured.
    best = (history[-1] if args.measure_rollouts <= 0
            else max(history, key=lambda h: h["train_acc"]))
    (d / "history.json").write_text(json.dumps(history, indent=1, ensure_ascii=False))
    driver.Artifact(
        method=METHOD, family=fam.key, arm=ARM, text=best["prompt"], model=args.model,
        seed=args.seed, train_tasks=len(recs),
        rollouts_used=(args.epochs + 1) * len(recs) * args.measure_rollouts + args.epochs * len(recs),
        rollouts_new=-1, lm_calls=getattr(eng, "n_calls", -1),
        notes={"seed_instruction": SEED_INSTRUCTION, "epochs": args.epochs,
               "mode": "single_pass" if (args.epochs == 1 and args.measure_rollouts <= 0) else "search",
               "batch_size": args.batch_size, "measure_rollouts": args.measure_rollouts,
               "best_epoch": best["epoch"], "train_acc": best["train_acc"],
               "seed_train_acc": history[0]["train_acc"], "wall_s": round(time.time() - t0)}).save()
    return {"family": fam.key, "n_train": len(recs), "best_epoch": best["epoch"],
            "seed_acc": history[0]["train_acc"], "best_acc": best["train_acc"],
            "engine_calls": getattr(eng, "n_calls", -1), "wall_s": round(time.time() - t0)}


def main(argv=None) -> int:
    ap = driver.parser(METHOD, ARM)
    ap.add_argument("--epochs", type=int, default=1, help="1 = single pass over the train set")
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--measure-rollouts", type=int, default=0,
                    help="0 = never measure, keep the prompt the pass produced (default); "
                         "K > 0 scores each epoch on train at K rollouts and keeps the best")
    args = ap.parse_args(argv)
    fams = driver.select_families(args.families)[: args.limit]
    log = print

    if args.verb == "status":
        print(json.dumps(driver.status(METHOD, fams), indent=1))
        return 0
    rollouts = args.rollouts if args.rollouts is not None else (4 if args.verb == "train" else 1)
    reports = []
    for i, fam in enumerate(fams, 1):
        log(f"[{i}/{len(fams)}] {args.verb} {fam.key}")
        if args.verb == "train":
            rep = train_family(fam, args, log=log)
        elif args.verb == "deploy":
            rep = driver.deploy_family(METHOD, fam, ARM, args.split, rollouts=rollouts,
                                       workers=args.workers, model=args.model,
                                       dry_run=args.dry_run, log=log)
        else:
            rep = driver.control_family(fam, ARM, args.split, rollouts=rollouts,
                                        workers=args.workers, model=args.model,
                                        dry_run=args.dry_run, log=log)
        rep["family"] = fam.key
        reports.append(rep)
        log("   " + json.dumps({k: v for k, v in rep.items() if k != "map"}, default=str))
    out = driver.method_root(METHOD) / (
        f"_{args.verb}_report.json" if args.verb == "train" else
        f"_{args.verb}_{args.split}_{driver.model_tag(args.model)}_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=1, default=str))
    log(f"\n{args.verb}: {len(reports)} families; report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
