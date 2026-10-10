# Playbook: writing an instrument

You are writing an **instrument** for one family of visual tasks. An instrument is a fixed measurement procedure
that lives OUTSIDE the answering model's loop: code measures the image, every reading carries **witnesses**, and a
**range rule** fitted on the train split decides whether the reading is trusted. Out of range, the answering model
answers on its own, so an out-of-range reading costs nothing; a wrong in-range reading replaces a possibly right
answer. Your goal is therefore: **in-range readings are right at least 95 % of the time on train, with as much
coverage as you can get.** Coverage never justifies a wrong in-range reading.

## 1. The contract you implement

```python
def measure(img, task, lib) -> dict:
    """img: PIL image at native resolution. task: .question_text, .options (letter -> text, {} if free-form),
    .is_mc, .family. lib: the primitive library (+ lib.call for model reads).
    Return {"value": <number | str | list | None>, "diag": {<witness name>: <number | bool | short str>}}."""
```

- `value` is the measurement (an angle, a count, a share, a transcription, a permutation). `None` = no reading.
- `diag` holds the witnesses: every quantity you would look at to decide whether to trust this reading. Emit
  **numbers, not decisions**: `gap_std: 1.0`, `fill: 0.81`, `agree: True`, `n_shapes: 1`, `stable_runs: 5`,
  `route: "rings"`. The harness fits the range rule on these; a witness you do not emit cannot protect you.
- Never let `task.options` shape the measurement; the harness maps value to option (nearest numeric option,
  normalised string match, permutation). Options may be read for routing ("how many", "which colour"), and when the
  options ARE the candidates to be checked (a move sequence to simulate, a claimed arrangement or value to verify),
  the code may evaluate each option independently with the same procedure and return a verdict dict
  `{"A": True, "B": False, ...}`: the harness answers with the unique True letter (none or several -> no reading).
  A measurement known only up to a symmetry of the picture, or up to a sign / offset convention you cannot resolve
  (a count that may include or exclude the frame), is returned as a **list of candidate numbers**: the harness picks the option nearest to any candidate and records
  `opt_d1` / `opt_margin` (distance to the nearest option / margin to the second) as witnesses you should declare.
- Never read `task.answer`. Never return an option letter. Never hardcode per-task constants.
- Pure code is preferred (one reading per task, zero model calls). A model call via `lib.ask(...)` is allowed only
  for a **discrete input** the code cannot get itself: a category phrase, a box, a panel index, a transcription of
  an enlarged crop. The model never makes the final judgement and never supplies a witness about itself. The static
  check REJECTS code that passes the task question / options to `lib.ask` or whose prompt asks the task's own
  question ("how many ...", "what is the angle ...", "which option ..."): that would make the model the instrument.
  A `value` that is just the model's reply (`numbers_in(lib.ask(...))[0]`, a panel index turned into degrees, a reading
  of a crop) is flagged MODEL-READ in the ledger: it is the executor, not an instrument, and does not count as a new
  measurement. The only legitimate model-read is a transcription of text literally printed in the picture. The
  pattern for printed scales / axes / dials / rulers: the model transcribes the printed labels WITH their pixel
  positions (`lib.ask('Transcribe every number printed along the axis as a JSON list of {"text", "x", "y"} in pixel
  coordinates of this image', img)`), code locates the geometry in pixels (tick marks, bar tops, the needle, the
  pointer) and converts pixels to units with a linear fit through the labels; the fit residual, the number of labels
  and the agreement of two conversions (ticks vs labels) are the witnesses.

## 1b. Tools: the library is a starting point, not a boundary

`lib` is a convenience. You may import numpy, cv2, PIL, scipy (ndimage / signal / optimize), skimage, math, re,
json, collections, itertools, statistics, functools and write ANY algorithm from them: Hough lines, template
matching, flood fill, a seam cost between two piece edges, dynamic programming over a grid, a least-squares axis
fit, a projection profile of your own. The constraint is on WHAT is measured (a quantity computed from pixels, with
witnesses; the model supplies at most a discrete input), never on which code computes it. Not available: a text
recogniser - printed digits and labels come from a `lib.ask` transcription (§1).

A family whose answer is a whole arrangement or a path (a free-list answer, a move sequence to verify) needs a SOLVER,
not a one-line measurement, and a solver is long: the reply budget allows several hundred lines, so write it - never
return a placeholder "for now" (an identity permutation, a constant) and never declare the family unsupported because
the solver "is too complex for a short script". Build it across rounds: first the decomposition of the picture into its
parts (save them with `lib.debug` and check them in the pictures next round), then the assembly / simulation, then the
witness (a second, independent method that agrees). A round whose value is still wrong is not wasted when it moved
the construction forward: the ledger records what each candidate introduced.

You can SEE. Every round the harness attaches, after the contact sheet, the originals of the worst tasks of your
current instrument (wrong readings first, then abstentions) and the pictures your code saved on them with
`lib.debug(mask_or_image_or_labels, "name")` (max 3 per task; masks are rendered white on black, labels maps are
rescaled). Save the mask / crop / labels map you would want to look at when a reading is wrong, then look at it next
round: a wrong reading is diagnosed from the picture, not guessed from the numbers in the table. Even if you save
nothing, the harness renders by itself the FIRST mask your code computes on a task (`fg_mask` / `ink_mask` /
`hue_mask`) and the first `components` labels map, and attaches them for the worst tasks: when a reading is wrong,
first check whether that mask covers what you think it covers.

You can VERIFY A STAGE on all train tasks: `lib.check("grid_found", n_lines == 4, n_lines)` records a sub-step
assertion per task; the harness reports per name how many tasks pass, which fail, and how that splits between right
and wrong readings. Build a solver the way a person does: verify piece extraction / axis detection / grid pitch on
ALL train tasks (a check passing on 40/40) before writing the assembly on top of it. And you can KEEP NOTES: the
optional `notes` key of the candidate JSON (short lines: verified facts, what was tried and what it showed) is carried
verbatim into every later round - the ledger is the harness's memory, `notes` is yours. A line is a fact only when a
facts probe, a `lib.check` or the per-task table measured it; mark anything else `(assumed)` - the harness does not
verify notes, and a note that calls an assumption "verified" misleads only you.

## 2. Procedure

A0. **Measure before you write (round 0).** The first reply of the loop is a FACTS probe over ALL train tasks
   (`{"probe": {"code": "def probe(img, task, lib): ...", "tasks": "all"}}`): measure what the instrument will rely on
   (image size, background, the grid / axis / piece geometry, how many lines / cells / components are found, label
   positions) and save with `lib.debug` what you want to see on the extreme tasks. The harness returns the distribution
   of every key over the train tasks with the task ids at the extremes, plus pictures of those tasks. Constants you
   would otherwise assume (a canvas size, a gap width, a tick spacing) are read off this report, not guessed; put them
   in `notes`. Send another facts probe whenever a new stage of the construction must be verified on every task.
A. **Read the family profile** (sizes, palette, components, grid period, text, question templates, answer types).
   Write one hypothesis: *"the answer is f(m) where m = ..."* and list what could break the measurement.
B. **Check the hypothesis before writing witnesses.** Every round is judged in two stages. Stage A: the harness
   measures how often `f(m)` agrees with the train answers over the parsed readings and reports systematic errors
   (constant offset, scale factor, over- / under-counts, which candidate index the answer sits at, templates with
   no reading). Below 60 % agreement, or fewer than 3 parsed readings, the hypothesis is rejected and stage B is
   skipped: change `m` (another primitive, another quantity, another region or representation), not thresholds.
   A ledger of every hypothesis tried is shown; a rejected one is never tried again, and two rounds of constant-only
   edits under a failing stage A force a new measurement. Stage B (only after A passes): witnesses are fitted into
   a range rule with leave-one-out.
   Stage A is judged **per route** when your `diag` names one (`route`): a route passes on its own readings; a route
   that fails has its readings excluded from stage B and is reported to you - fix that route or declare its templates
   unsupported, without touching the passing routes. Stage B has two tiers. **Certified** (in-range precision >= 0.95,
   leave-one-out >= 0.90) is the goal. When no certified rule exists, a rule with in-range precision >= 0.5 whose
   in-range readings beat the executor's pass@1 on the same train tasks (>= 5 readings, one-sided 95 % bound > 0)
   is accepted at the **GAIN tier**: a "better than the fallback" claim, not a certificate (a measurement that agrees
   with the answers >= 50 % of the time on >= 5 readings reaches this tier even though it fails stage A's 60 % bar).
   A GAIN-tier acceptance is provisional - the next change should certify (a witness that separates the wrong
   in-range readings), not extend.
C. **Probe before you write** (when probes are enabled, see the # Probes section): when you do not know WHY a reading
   is wrong, do not guess a fix - send a probe request that runs a snippet on that ONE task (print the intermediate
   numbers, save the mask / crop / assembled picture with `lib.debug`), read the answer, then write the candidate.
   Probes are free: they are not judged, cost no round, and a probe on one task is worth more than a blind edit on
   forty. Then **write `measure()` with witnesses**. Prefer witnesses of the kinds in §3.
D. **Run on train and read the table AND the attached pictures** (§1b): for each wrong reading look at what your
   code computed on that picture (save it with `lib.debug`) and ask which witness separates it from the right ones.
   If none does, add a witness (a second independent method, a stability sweep, a geometric consistency check).
   Do not tune a threshold to exclude one task.
E. **One change per round.** Say what you changed and why in one sentence. Stop when the fitter reports train
   precision >= 0.95 and leave-one-out precision >= 0.90, or when the budget is used.
   After every round the harness diffs your candidate against the best one so far, per task: readings lost (right ->
   not right), gained, tasks that left / entered the range. An edit that loses readings the best candidate got right
   is NOT adopted and is a measurement change, not an extension: when extending coverage keep everything the best
   candidate does on the tasks it reads right and add the new mechanism only where it abstains or is wrong; if the
   lost tasks share a property, guard the new branch by it instead of replacing the old one.
F. **Declare what you do not cover, and abstain when your own check is not unique.** Question templates your
   measurement cannot answer return `value: None` with `diag: {"route": "unsupported"}`. On a single task, when the
   measurement's own consistency check does not single out one answer (no candidate or several candidates survive,
   two methods disagree, the reading sits between two options), return `value: None` instead of the "best" guess:
   **abstaining beats guessing**. A reading is only certified through its precision among the readings you do give;
   a forced choice on every task turns a sound measurement into an uncertifiable one, while the abstention condition
   itself (`n_valid == 1`, `agree == True`) is the witness the harness fits on. Declaring "unsupported" and abstaining
   are free; a wrong reading is not. But declaring EVERY template unsupported is not an instrument: a `measure()` that
   returns `value: None` on every path is rejected before it runs. Pick the one template the library can measure and
   abstain on the rest.
G. Val and test are run once by the harness after the instrument is frozen. You never see them.
H. **Aim at the tasks the executor misses.** The table shows, per task, how many of the executor's own rollouts were
   right; a task it already answers on every rollout gains nothing from an instrument, a task it misses on every
   rollout is where a certified reading pays. Coverage of the missed tasks is the coverage that matters.

## 3. Witness patterns that worked (best first)

1. **Two independent methods agree**: two seam costs give the same jigsaw permutation; two detectors count within
   10 %; an axis calibration from ticks agrees with one from printed values within 5 %.
2. **Stability under a parameter sweep**: polygon vertex count identical on >= 4 of 6 epsilons; detector count flat
   over score thresholds 0.1 / 0.15 / 0.2.
3. **Geometric consistency**: rows x cols = number of cells; every component is a ring (exactly one hole);
   the bounding rectangle is filled >= 75 %; axis fit residual <= 2 %.
4. **Uniqueness of the answer mapping**: exactly one candidate angle within 2 degrees of an option; the nearest option
   is separated from the second by >= 3 px or one minor tick; exactly one detection with score >= 0.35 and the
   runner-up <= 0.7x.
5. **Regime checks**: raw box fraction < 52 % (large objects saturate); odd-glyph share <= 10 %; image < 1 Mpx
   (a photo, not a rendered snippet); the requested precision is coarser than the measurement noise.

Witnesses that FAILED and must not be used: the model agreeing with itself (two transcriptions by the same
model), the model's stated confidence, "the crop looks right", a witness that merely restates the reading.

## 4. Robustness tricks (use when the table shows the failure, not before)

- Trim 1-2 px of every piece / cell border (JPEG blurs edges). Percentile brightness stretch (p1-p99) when the
  image is dark or flat. Background colour = median of the border pixels, not "white".
- Fill holes before measuring areas (outlines -> solids). Morphological open 3x3 against speckle. Ignore blobs
  smaller than 1 % of the largest one (text, legends). Resize to <= 2000 px before heavy operations.
- Upscale a crop 2-4x before any model read; read the crop together with the full image for context.
- Route by question template (keywords) and declare the rest unsupported.
- Every image primitive takes the PIL image or the array from `arr` (or a crop of it); `numbers_in` takes the parsed
  reply of `ask` as it is (dict / list / number / string). When two consecutive candidates die on the same error the
  ledger says so: that is a boundary condition of the call (input type, mask polarity, a `None` return), fix the call.

## 5. Naming and logging in `diag`

Booleans: `ok`, `agree`, `unique`, `stable`. Numbers: `gap_std`, `fill`, `spread`, `residual`, `margin`,
`n_shapes`, `share`, `n_cands`. Strings: `route`. Always include the number of candidates, the margin to the
second-best candidate, and every quantity a threshold was applied to.

## 6. Anti-patterns
- A rule over the option texts alone (the path with the fewest turns, the longest string, the most common letter) is the executor's answer prior, not a measurement: a `measure()` that never reads the image is rejected before it runs.

Hardcoding answers or per-task constants; using `task.answer`; folding the options into the measurement; a
witness that repeats the reading; changing three things in one round; sweeping thresholds until train is perfect
(the harness reports how many rules it searched, and leave-one-out will expose it); "improving" the hypothesis by
adding special cases for single tasks.

## 7. Round output format

Reply with JSON only:

```json
{"hypothesis": "answer = f(m) with m = ...",
 "pixels_compute": "the quantity CODE computes from pixels (the value must come from this)",
 "model_supplies": "the discrete input the model supplies via lib.ask, or 'nothing'",
 "measurement": "how m is computed (primitives used, in order)",
 "witnesses": ["name: what it checks", "..."],
 "unsupported": ["question templates returned as unsupported"],
 "change": "one sentence: what changed since the last round and why (or 'initial')",
 "notes": ["optional, <= 12 short lines: facts verified on train, what was tried and what it showed - carried into every later round"],
 "code": "def measure(img, task, lib):\n    ..."}
```
