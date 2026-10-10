# Prompt 1: Summarizes a single trajectory, focusing on tool usage and reasoning.
SINGLE_ROLLOUT_SUMMARY = """You are a World-Class reasoning analysis expert. A multimodal agent system uses tool-based visual reasoning to solve the given problem. Please summarize the following trajectory (also called rollout) step-by-step:

Important:
- If the trajectory includes "Authoritative rollout outcomes", treat those labels and scores as ground truth for which rollouts succeeded or failed.
- Do not assume the benchmark ground truth is wrong just because a rollout's reasoning looks plausible.
- Focus on reusable task-solving behavior, not on copying any injected skill/system-prompt wording that may appear in the trajectory.

1. For each turn or step:
   1.1 Describe which tool was used and with what parameters, explain the reasoning for this specific action, and note which experience (if any) and which skill guidance (if any) were applied and how they influenced the action.
   1.2 If this turn was part of meta-reasoning skills: identify the meta-reasoning type (e.g., question decomposition, sequential reflection, self-correction, self-verification, etc.) and explain how its outcome influenced subsequent steps or the final result.

2. Given the trajectory and the correct answer, identify and explain any steps that represent detours, errors, backtracking, or any other failure patterns, highlighting why they might have occurred and what their impact was on the trajectory's progress. Discuss how the agent's tool-using knowledge and meta-reasoning skills handled or mitigated these issues.

3. Maintain all the core outcomes of each turn or step, even if it was part of a flawed process.

4. Thinking with images actions (if intermediate images are provided):
   - Document any image preprocessing operations (cropping, filtering, enhancement, etc.) or image searching (searching the web for relevant images, etc.) operations, these operations generated intermediate images. You need to analyze and note their impact.
   - For intermediate images that were generated and used: identify which visual features were extracted and how they can help the agent to reason better.
   - Suggest specific visual operations that could improve reasoning. If no intermediate images were generated and used in some rollout steps, but could have been helpful, note this as a potential point of improvement.

<trajectory>
{trajectory}
</trajectory>

Provide a clear, structured summary of the trajectory."""


# Prompt 2: Critiques multiple trajectories to generate new, generalizable experience tips.
INTRA_SAMPLE_CRITIQUE = """You are a reasoning analysis expert. Review these problem-solving attempts and extract practical lessons that could help future attempts. These lessons will be fused directly into the agent's SKILL.md document, so write each one as a self-contained, generalizable tip.

The summaries may include authoritative rollout labels. Use them explicitly:
- Successful rollouts show what should be repeated.
- Failed rollouts show what should be avoided or verified.
- Mixed outcomes are the most valuable: identify the decision or tool-use difference that explains the success/failure gap.
- If all rollouts failed, extract failure-avoidance rules rather than copying the failed solution path.

### Analysis Framework:

1. **Trajectory Review**:
   - What worked? What key decisions or insights led to correct answers?
   - What didn't work? Where did reasoning go wrong, and why?
   - How were different tools combined? What sequences were effective?
   - For visual operations: What preprocessing helped? What was missing?

2. **Experience Extraction**:
   Create experience tips in two categories:
   - **Execution Tips**: Practical advice on tool usage - when to use which tool, what parameters work best, how to interpret results effectively.
   - **Decision Rules**: Simple guidelines for reasoning choices - when to decompose a problem, when to search for information, when to double-check results.

   Emit at most {max_ops} tips for this case.

3. **Experience Quality**:
   - Keep each tip under 64 words.
   - Start with the situation or condition when the advice applies.
   - Make it general enough to apply to similar problems.
   - Focus on actionable guidance, not abstract principles.
   - Avoid specific examples - the tip should generalize.

Output ONLY a JSON array. Do not include markdown fences, preamble, or analysis text:
[
  {{
    "experience": "the new generalizable tip"
  }},
  ...
]
Quality over quantity.

<question>
{question}
</question>

<summaries>
{summaries}
</summaries>

<ground_truth>
{groundtruth}
</ground_truth>
"""


# Prompt 2a: Success-lens analyst. Focuses on what worked in successful
# rollouts. Always emits "do this / repeat this" tips, never avoidance rules.
SUCCESS_ANALYST_CRITIQUE = """You are a SUCCESS-PATTERN analyst. Your sole job is to look at successful rollouts on this task and extract repeatable patterns that future attempts should imitate. Failed rollouts are provided only as contrast — DO NOT write avoidance/anti-patterns; another analyst handles that.

The summaries include authoritative rollout labels. Use them:
- Successful rollouts (status=success, accuracy>=1.0) are your primary evidence.
- Failed rollouts are provided only so you can spot the *decisive* difference (a tool choice, an order, a verification step) that explains why the success worked.
- If NO rollout succeeded on this task, emit an empty list [].

### Analysis Framework:

1. **What worked?**
   - Which tool calls / parameter choices produced the correct answer?
   - Which decision or verification step was decisive (i.e., absent in failures, present in successes)?
   - Which sequence or ordering of operations led to the answer efficiently?
   - For visual operations: which preprocessing / inspection step extracted the key feature?

2. **Generalize, don't memorize**:
   - The tip must apply to *similar future tasks*, not just this exact one.
   - Strip task-specific entities (object names, file paths, exact numbers).
   - Keep the *condition* and the *action*: "When <situation>, do <action> because <effect>."

3. **Tip quality bar**:
   - Each tip MUST be a positive recommendation ("do X when Y", "prefer A over B").
   - At most {max_ops} tips. Quality over quantity — emit fewer if you only have one strong pattern.
   - Each tip under 64 words.
   - Skip tips that just restate the task statement or the ground truth.

Output ONLY a JSON array. No markdown fences, no preamble, no analysis text:
[
  {{
    "experience": "the success-pattern tip"
  }},
  ...
]

<question>
{question}
</question>

<summaries>
{summaries}
</summaries>

<ground_truth>
{groundtruth}
</ground_truth>
"""


# Prompt 2b: Failure-lens analyst. Focuses on what went wrong in failed
# rollouts. Always emits "avoid this / verify this" tips, never positive
# how-tos.
FAILURE_ANALYST_CRITIQUE = """You are a FAILURE-MODE analyst. Your sole job is to look at failed rollouts on this task and extract reusable avoidance rules and verification checks. Successful rollouts are provided only as contrast — DO NOT write positive how-tos; another analyst handles that.

The summaries include authoritative rollout labels. Use them:
- Failed rollouts (status=failure, accuracy<1.0) are your primary evidence.
- Successful rollouts are provided only so you can identify the *missing* step or the *wrong* choice that distinguishes failure from success.
- If NO rollout failed on this task, emit an empty list [].

### Analysis Framework:

1. **What went wrong?**
   - Which tool call / parameter / assumption produced the wrong answer?
   - Which verification step was *missing* in failures but *present* in successes?
   - Which detour, premature commit, or hallucinated fact derailed the trajectory?
   - For visual operations: what preprocessing was needed but skipped? What feature was misread?

2. **Generalize, don't memorize**:
   - The tip must apply to *similar future tasks*, not just this exact one.
   - Strip task-specific entities (object names, file paths, exact numbers).
   - Keep the *trigger* and the *guard*: "When <situation>, watch for <pitfall> — verify by <check>."

3. **Tip quality bar**:
   - Each tip MUST be an avoidance rule, a verification check, or a precondition reminder ("don't X without Y", "verify Z before committing", "watch out for ...").
   - At most {max_ops} tips. Quality over quantity — emit fewer if you only have one decisive failure mode.
   - Each tip under 64 words.
   - Skip tips that just blame the model or restate the failure outcome — the tip must be ACTIONABLE for a future agent.

Output ONLY a JSON array. No markdown fences, no preamble, no analysis text:
[
  {{
    "experience": "the failure-avoidance tip"
  }},
  ...
]

<question>
{question}
</question>

<summaries>
{summaries}
</summaries>

<ground_truth>
{groundtruth}
</ground_truth>
"""


# Prompt 2c: Agentic failure analyst — turn-1 init prompt. Same evidence as
# the single-shot failure analyst, plus a list of available image IDs the
# analyst can request to inspect, plus the action protocol.
FAILURE_ANALYST_AGENTIC_INIT_PROMPT = """You are a FAILURE-MODE analyst running in an iterative loop. Your goal is to extract reusable avoidance / verification rules from the failed rollouts on this task. You have a budget of {max_turns} total turns.

The summaries include authoritative rollout labels:
- Failed rollouts (status=failure, accuracy<1.0) are your primary evidence.
- Successful rollouts (if any) are contrast — they may show what the failures missed.
- If NO rollout failed, immediately call `done` with an empty list.

### Your action protocol

In every turn you respond with EXACTLY ONE JSON object — nothing else, no commentary, no markdown fences. One of two shapes:

A) Inspect a specific image to verify a hypothesis:
{{
  "action": "view_image",
  "image_id": "<one ID from the available-images list>",
  "reason": "<one short sentence on what you expect to verify>"
}}
The system will load that image and present it to you in your next turn. The text trajectory is text-only; images are needed when you suspect the agent's *visual perception* (not its reasoning) is what failed. You can view at most {max_views} images across all turns.

B) Emit your final tips and exit:
{{
  "action": "done",
  "tips": [
    {{"experience": "the failure-avoidance tip"}}
  ]
}}
At most {max_ops} tips. Each tip MUST be an avoidance rule, verification check, or precondition reminder ("don't X without Y", "verify Z before committing"). Each tip ≤ 64 words. Strip task-specific entities (object names, file paths, exact numbers) — the tip must apply to *similar future tasks*. Skip tips that just blame the model or restate the failure.

### When to call `done` vs. `view_image`
- If the text trajectory already makes the failure mode clear (e.g. agent never invoked a needed tool, agent looped on retries, agent hallucinated a value clearly contradicted by the trajectory), call `done` immediately. Don't burn turns.
- Use `view_image` ONLY when the failure is plausibly visual: the agent saw an image and reported the wrong content, OR the agent cropped/preprocessed and the result is suspect.
- DO NOT browse images out of curiosity. Each `view_image` costs a turn.

### Hard rules
- Output exactly ONE JSON object per turn. No markdown fences, no preamble.
- `view_image.image_id` MUST be a value from the available-images list below; unknown IDs will be rejected and you lose the turn.
- After {max_turns} turns OR {max_views} image views, the system forces a final tips emission whether you call `done` or not — so prefer to `done` early when you have enough evidence.

### Available images (request by ID)
{image_catalog}

<question>
{question}
</question>

<summaries>
{summaries}
</summaries>

<ground_truth>
{groundtruth}
</ground_truth>

Respond with the single JSON object for your turn 1 action."""


# Prompt 2d: Force-final prompt used when the agentic analyst hits its turn
# or view budget without calling `done`. Sent with all viewed images in
# context so the analyst can write tips grounded in everything it saw.
FAILURE_ANALYST_AGENTIC_FORCE_FINAL_PROMPT = """Your turn / view budget for this case is exhausted. Emit your final tips NOW.

Constraints:
- Respond with ONE JSON object: {{"action": "done", "tips": [{{"experience": "..."}}, ...]}}.
- At most {max_ops} tips. Each tip is an avoidance rule, verification check, or precondition reminder (NOT a positive how-to). Each ≤ 64 words.
- Generalize: strip task-specific entities. The tip must apply to similar future tasks.
- If you don't have enough evidence to write a meaningful tip, emit {{"action": "done", "tips": []}} — that is preferred to inventing one.

No markdown fences, no preamble, no commentary outside the JSON."""


# Prompt 2e: Continuation prompt for turn N (N>=2). Sent with whatever
# images the analyst has requested so far. Reminds it of the protocol AND
# of the two action shapes — empty-image_id `view_image` was observed in
# the wild and now counts as malformed.
FAILURE_ANALYST_AGENTIC_CONTINUE_PROMPT = """You have used {turns_used} of {max_turns} turns and {views_used} of {max_views} image views. {remaining_views} image views remain.

Available images you have NOT yet viewed:
{remaining_catalog}

Based on the trajectory text and the {views_used} image(s) you have seen, choose your next action. Respond with EXACTLY ONE JSON object — no commentary, no markdown fences. One of:

A) {{"action": "view_image", "image_id": "<EXACT id from the list above>", "reason": "<one sentence>"}}
B) {{"action": "done", "tips": [{{"experience": "..."}}, ...]}}

CRITICAL:
- If the "Available images you have NOT yet viewed" list is empty OR you already have enough evidence, you MUST call `done` now. Do NOT emit `view_image` with an empty / null / missing `image_id`; that is treated as a malformed response.
- The `image_id` must be the verbatim string from the list above (e.g. "rollout_0_tool_image_3"); not a description, not a path, not a number.
- Two consecutive malformed responses end the loop early."""


# ============================================================================
# Stage 1: structured visual-recipe extraction. The success and failure
# analysts emit a JSON list of structured entries (instead of a flat list of
# {"experience": str}). Each entry captures concrete visual operations,
# triggers, and verification rules. Stage 2 (not yet built) will use these
# fields to render conditional advisory bullets; for now the dispatcher
# saves the raw structured JSON for inspection AND renders one fallback
# advisory sentence per entry so the existing SKILL.md fuse path keeps
# working.
# ============================================================================

# Shared schema description used in both prompts so success + failure analyst
# outputs are interpretable and round-trippable. Each field is REQUIRED in the
# output schema; if a field doesn't apply (e.g. visual_failure on a success
# entry) emit "" — but never omit the key.
#
# Stage 2: dropped `evidence_coordinates` (Stage-1 audit showed 1/3 of
# entries hallucinated suffixes like "_enhanced_contrast" instead of citing
# real image_ids; the field had the lowest signal-to-noise of all 6).
_STRUCTURED_SCHEMA_BLOCK = """### Output schema (STRICT)
Return ONLY a JSON array. No markdown fences, no preamble. Each element is an object with EXACTLY these 5 keys (string values only — use "" for "not applicable"):

[
  {{
    "trigger_condition":      "<image-observable property at INITIAL inspection that says when this entry applies — e.g. 'target text region is small (<~30 pixels tall) or partially occluded'. MUST be something the agent can decide from the raw image alone, BEFORE running any tool. Trajectory-only properties (e.g. 'after the OCR failed') are FORBIDDEN here. Keep ≤ 20 words.>",
    "visual_failure":         "<concrete visual mistake the failed rollout(s) made — e.g. 'OCR returned garbage on the unrotated raw region'. Empty string for success-side entries when no contrasting failure exists. Keep ≤ 15 words.>",
    "successful_visual_ops":  "<the visual operations that actually worked (success-side) OR the hypothesized fix (failure-side). One short clause, e.g. 'rotate to horizontal then crop tight'. Avoid hedge words. Keep ≤ 15 words.>",
    "tool_sequence":          "<ordered tool/operation sequence, '->' separated, e.g. 'crop_tight -> upscale_2x -> ocr'. Use generic operation names (crop / zoom / rotate / ocr / enhance / segment / count / measure / annotate), NOT specific function signatures or parameter values. Keep ≤ 8 ops.>",
    "verification_rule":      "<the post-action check that catches the failure mode — e.g. 'read OCR output twice with different thresholds; mismatch means re-crop tighter'. Empty string only if no verification is meaningful. Keep ≤ 25 words.>"
  }}
]

### Hard rules on the schema
- All 5 keys MUST be present in every entry. Use "" (empty string) for not-applicable, never null and never omit.
- Each field is a single short string clause. Word caps above are budgets; staying under the budget is preferred over hitting it.
- `trigger_condition` MUST be checkable BEFORE the agent runs any tool — i.e. a property of the raw image (size, contrast, angle, occlusion, count of candidates, presence of glare/text/chart, etc.). If the only condition you can articulate is "after-the-fact" (e.g. "after OCR failed"), do not write that — instead leave the entry out entirely. We'd rather have fewer high-quality entries than padded ones.
- `tool_sequence` describes operations the agent could execute via its tools or via `code_interpreter` for image processing. No code, no `(args)`, no specific pixel values from this case. Avoid library-specific function names like `cv2.adaptiveThreshold` — say `threshold` instead.
- DROP entries whose generalization value is zero (specific to this exact prompt / image / answer). The agent will see other tasks, not this one.
- Cap: emit at most {max_ops} entries. Prefer 1-2 strong entries over {max_ops} weak ones. If there is genuinely nothing structured to extract, emit []."""


# Success-lens structured analyst.
SUCCESS_ANALYST_STRUCTURED_PROMPT = """You are a SUCCESS-PATTERN visual analyst. Look at the successful rollout(s) on this task and extract reusable visual-operation recipes that future similar tasks should imitate. Failed rollouts are provided only as contrast.

The summaries include authoritative rollout labels:
- Successful rollouts (status=success, accuracy>=1.0) are your primary evidence.
- Failed rollouts are contrast — use them to identify the *decisive* visual difference (the operation, ordering, or verification step that the success had and the failure lacked).
- If NO rollout succeeded, emit [].

### What to extract
For each entry, identify ONE coherent visual recipe:
- a *trigger condition* (image-observable property where this recipe applies),
- the *visual ops* / *tool sequence* that worked,
- a *verification rule* the agent can apply to confirm the result.

Set `visual_failure` to "" (success-side: this recipe worked, no failure to record).

### Downstream rendering (read carefully — your fields are surfaced to the agent verbatim)
Each entry is rendered into a single bullet of the form:
    "When [trigger_condition], a path that has worked: [tool_sequence rendered as op1 → op2 → op3] (verify by [verification_rule])."

This means:
- `trigger_condition` is the ONLY thing that prevents the procedure from firing on unrelated tasks — make it precise and image-observable. An entry with a vague trigger ("when the task is hard") is worse than no entry at all; emit fewer entries rather than pad.
- `tool_sequence` is surfaced as the bullet's body. Use 2-5 generic ops, '->' separated (e.g. "locate_target -> estimate_angle -> crop_tight -> flip_vertical -> read_text"). Underscores become spaces on render. The reader treats this as the canonical successful procedure for cases where the trigger fires, so order it the way it actually worked.
- `verification_rule` is kept as a brief tail check ONLY when ≤ 12 words. Anything longer is dropped on render — write a compact one-clause check ("re-read with horizontal orientation; mismatch means re-crop") or leave the field empty.
- `successful_visual_ops` is a fallback used only when `tool_sequence` is empty. Prefer to fill `tool_sequence` instead.

Avoid:
- Generic advice ("be careful", "verify the answer"). The output goes into a structured field where vague text has zero downstream value.
- Recipes that just say "look at the image" without specifying any visual operation.
- Memorization (object names, exact answers, specific pixel values, sample IDs).
- Redundancy across entries. When the success rollouts cluster on one recipe, emit ONE entry; do not pad to {max_ops} by re-phrasing it. Multiple entries are only worth emitting when their `trigger_condition` fields are mutually exclusive (different image-observable conditions invoke different recipes).

{schema}

<question>
{question}
</question>

<summaries>
{summaries}
</summaries>

<ground_truth>
{groundtruth}
</ground_truth>
"""


# Failure-lens structured analyst.
FAILURE_ANALYST_STRUCTURED_PROMPT = """You are a FAILURE-MODE visual analyst. Look at the failed rollout(s) on this task and extract structured failure-avoidance recipes.

The summaries include authoritative rollout labels:
- Failed rollouts (status=failure, accuracy<1.0) are your primary evidence.
- Successful rollouts are contrast — use them to identify the *missing* visual operation or *wrong* choice.
- If NO rollout failed, emit [].

### What to extract
For each entry, identify ONE coherent failure mode:
- a *trigger condition* (image-observable property where this failure happens),
- the *visual_failure* — what specifically went wrong visually (be concrete: "OCR misread '0' as 'o' on small italic text" beats "agent got the wrong answer"),
- the *successful_visual_ops* — the recipe that WOULD have worked (extract from any successful rollout, OR hypothesize from contrast),
- the *tool_sequence* implied by that recipe,
- the *verification_rule* that would have caught the failure pre-commit.

Avoid:
- Tips that just blame the model or say "the agent failed" — extract WHAT visually failed and HOW to avoid it.
- "Anti-tips" with no positive alternative — every entry must offer a concrete `successful_visual_ops` and `tool_sequence`, even if hypothesized.
- Memorization.

{schema}

<question>
{question}
</question>

<summaries>
{summaries}
</summaries>

<ground_truth>
{groundtruth}
</ground_truth>
"""


# Agentic-mode `done` action that returns structured entries instead of
# the bare `tips` list. Used when the agentic failure analyst is run with
# structured extraction enabled. The init prompt mentions both shapes; the
# system accepts either.
FAILURE_ANALYST_AGENTIC_STRUCTURED_DONE_HINT = """When you call `done`, the `tips` array MUST contain structured entries with the 5-field schema (trigger_condition, visual_failure, successful_visual_ops, tool_sequence, verification_rule). Use "" for not-applicable fields, never null."""


# ============================================================================
# TEXT-KNOWLEDGE extraction (MCE text tasks). Opt-in via --text-knowledge-skills.
#
# Produces ADDITIVE domain knowledge / disambiguation rules instead of the
# visual "watch out for / verify by" recipes. Designed for closed-set text
# classification/generation where the base model already has strong priors:
# only emit knowledge that corrects a GENUINE error, never blanket
# "re-examine everything" nudges that make a competent model second-guess
# its (correct) first instinct.
# ============================================================================

_TEXT_KNOWLEDGE_SCHEMA_BLOCK = """### Output schema (STRICT)
Return ONLY a JSON array. Each entry has exactly these 3 keys:
[
  {{
    "applies_when": "<text-observable property of the QUESTION where TWO OR MORE candidate labels are genuinely confusable — e.g. 'a scaly skin rash that could be psoriasis or fungal infection' or 'a percentage stated on a debt instrument'. Frame the CONFUSION to resolve, not a one-way trigger for a single answer. Decidable from the question text alone. <= 25 words.>",
    "knowledge": "<the hallmark feature(s) of EACH competing candidate, given SYMMETRICALLY so the reader decides by which features are actually present — e.g. 'psoriasis: silvery scaly plaques on extensor surfaces + nail pitting; fungal infection: ring-shaped lesion with central clearing and an active scaly border'. Give BOTH/ALL sides; do NOT pre-select a winner. <= 40 words.>",
    "distinguish_from": "<the decision rule that picks among them by FEATURE-MATCHING, including the NEGATIVE check — e.g. 'pick whichever hallmark is actually present; if silvery plaques / nail pitting are absent, it is NOT psoriasis'. <= 30 words.>"
  }}
]

### Hard rules
- All 3 keys present in every entry; use "" for not-applicable, never null.
- BIDIRECTIONAL & FEATURE-BASED ONLY. For any confusable pair/group, `knowledge` MUST give the distinguishing features of EVERY candidate, not just the one that happened to be correct here. The decision is made by matching features to the case — NEVER by a baked-in default winner.
- FORBIDDEN: a one-directional rule of the shape "When <cue>, suspect/answer <label A>" that names only the winner and gives the competing label(s) no distinguishing features. That overfits the winner and FLIPS cases that are actually the other label (the exact failure this schema exists to prevent). If you cannot articulate the OTHER side's features, drop the entry.
- Emit ADDITIVE knowledge ONLY: concrete, non-obvious features that separate genuinely confusable labels. DO NOT emit generic process advice ("read carefully", "consider all options", "verify the answer", "evaluate the whole cluster", "watch for confirmation bias") — a competent base model already reasons; vague nudges make it second-guess CORRECT answers and are net-harmful. If you have no concrete contrastive feature, emit [].
- No memorization of THIS question's answer; the rule must generalize to other questions — including ones where the OTHER candidate is the correct label.
- CLOSED-SET LABELS ONLY: every label you name MUST be one of the candidate labels offered in the question. NEVER introduce a near-synonym outside that set (e.g. write the offered "varicose veins"/"drug reaction", not "chronic venous insufficiency"/"anaphylaxis") — an off-list label can never be the graded answer.
- GENERALIZE, DON'T MEMORIZE THE BATCH: you see a tiny sample of rollouts. Only emit a feature that is CAUSAL or DEFINITIONAL (true because of what the label means), not one that merely correlated with the right answer in these few cases. A "this label was usually right" base-rate does NOT transfer to the test distribution — if you cannot name WHY a feature implies a label, drop it.
- Prefer 1-2 high-confidence SYMMETRIC discriminators over {max_ops} weak ones. Cap: at most {max_ops} entries."""


SUCCESS_ANALYST_TEXT_PROMPT = """You are a DOMAIN-KNOWLEDGE analyst for a text task. From the rollout(s) that answered CORRECTLY, extract the reusable domain knowledge or decision rule that produced the right answer, so future similar questions benefit.

Rollout labels:
- Successful rollouts (accuracy>=1.0) are your evidence for what knowledge worked.
- Failed rollouts (if any) are contrast — what knowledge they lacked.
- If NO rollout succeeded, emit [].

Extract SYMMETRIC, FEATURE-BASED discriminators between confusable labels: what features distinguish each candidate from the other(s). The base model is already competent, so only record contrastive features specific and non-obvious enough to change a borderline decision. Never record "be careful" nudges, and never record a one-directional "cue -> winner" rule that omits the competing label's features (it flips cases that are actually the other label).

{schema}

<question>
{question}
</question>

<summaries>
{summaries}
</summaries>

<ground_truth>
{groundtruth}
</ground_truth>
"""


FAILURE_ANALYST_TEXT_PROMPT = """You are a DOMAIN-KNOWLEDGE analyst for a text task. From the rollout(s) that answered INCORRECTLY, identify the SPECIFIC missing fact or wrong mapping that caused the error, and state the correct rule.

Rollout labels:
- Failed rollouts (accuracy<1.0) are your primary evidence.
- Successful rollouts (if any) show the knowledge the failures lacked.
- If NO rollout failed, emit [].

The model confused two labels: it picked the wrong one when the right one was correct. Capture the SYMMETRIC discriminator for that confusable pair — the hallmark features of BOTH the right label and the wrong label the model picked — so future cases are decided by feature-matching in EITHER direction. Do NOT write a one-directional "cue -> right label" rule (it would simply flip the bias the other way and break cases where the model's pick is actually correct). Do NOT write blanket warnings or "re-examine everything" advice. If the failure was not due to a missing contrastive feature (e.g. just a formatting slip), emit [].

{schema}

<question>
{question}
</question>

<summaries>
{summaries}
</summaries>

<ground_truth>
{groundtruth}
</ground_truth>
"""


# ============================================================================
# EXCEL / SPREADSHEET-OPERATION extraction (SpreadsheetBench & similar agentic
# workbook tasks). Selected when skill modality == 'excel'.
#
# Excel tasks are PROCEDURAL and CODE-DRIVEN: the agent reads a workbook,
# applies a transformation (formulas / openpyxl edits), and saves an .xlsx
# whose cell VALUES are graded against a golden range. So the reusable asset
# is neither a visual "watch out / verify by" recipe nor a closed-set domain
# fact — it is a concrete OPERATION RECIPE plus the verifier-aligned pitfall it
# avoids. The schema reuses the three text-knowledge keys (applies_when /
# knowledge / distinguish_from) so the downstream render + patch pipeline is
# unchanged; only the semantics differ:
#   applies_when     -> task condition decidable from the instruction/data shape
#   knowledge        -> the concrete operation/code recipe that produces correct
#                       cell values
#   distinguish_from -> the common WRONG approach this rules out
# ============================================================================

_EXCEL_OPERATION_SCHEMA_BLOCK = """### Output schema (STRICT)
Return ONLY a JSON array. Each entry has exactly these 3 keys:
[
  {{
    "applies_when": "<task/data condition where this recipe applies, decidable from the instruction or visible workbook shape — e.g. 'instruction asks to sum over the last N workdays relative to a reference cell' or 'date column stores ISO strings with a T...Z time component'. <= 20 words.>",
    "knowledge": "<the concrete operation/code recipe that yields the correct CELL VALUES, stated POSITIVELY — e.g. 'write the computed numeric result into the target cell, not just a formula string, because the grader reads cached values' or 'use WORKDAY/NETWORKDAYS for workday offsets and anchor the range on the reference cell'. <= 35 words.>",
    "distinguish_from": "<the common WRONG approach this rules out + why it fails the range/value check — e.g. 'leaving a bare formula whose cached value is missing, so openpyxl data_only reads None'. Empty string if none. <= 30 words.>"
  }}
]

### Hard rules
- All 3 keys present in every entry; use "" for not-applicable, never null.
- Emit ADDITIVE, ACTIONABLE recipes ONLY: a concrete operation, formula choice, type/format handling, or write/save step that changes the produced values. The recipe must generalize beyond THIS workbook.
- DO NOT emit generic process advice ("inspect the file carefully", "double-check the output", "verify the structure", "be careful with edge cases"). A competent agent already does this; vague nudges make it over-engineer correct solutions and are net-harmful. If you have no concrete recipe, emit [].
- Anchor recipes to how the answer is GRADED: a cell range compared by VALUE (cached/evaluated), exact target sheet + range, number_format for dates, and saving to the expected output path.
- No memorization of THIS task's literal answer; the recipe must transfer to other workbooks. Prefer 1-2 high-confidence recipes over {max_ops} weak ones. Cap: at most {max_ops} entries."""


SUCCESS_ANALYST_EXCEL_PROMPT = """You are a SPREADSHEET-OPERATION analyst for an agentic Excel task. From the rollout(s) that produced the CORRECT workbook, extract the reusable operation recipe(s) that yielded the right cell values, so future similar tasks imitate them.

Rollout labels:
- Successful rollouts (accuracy>=1.0) are your evidence for what worked.
- Failed rollouts (if any) are contrast — what operation they got wrong.
- If NO rollout succeeded, emit [].

Extract POSITIVE, concrete operation recipes: which formula/function to use, how to anchor ranges, how to handle data types / number_format / dates, and how to write+save so the GRADED cell values are correct. The base agent already knows generic spreadsheet basics, so only record recipes specific and non-obvious enough to change the produced values. Never record "be careful / verify" nudges.

{schema}

<question>
{question}
</question>

<summaries>
{summaries}
</summaries>

<ground_truth>
{groundtruth}
</ground_truth>
"""


FAILURE_ANALYST_EXCEL_PROMPT = """You are a SPREADSHEET-OPERATION analyst for an agentic Excel task. From the rollout(s) that produced an INCORRECT workbook, identify the SPECIFIC wrong operation that caused the value/range mismatch, and state the correct recipe.

Rollout labels:
- Failed rollouts (accuracy<1.0) are your primary evidence.
- Successful rollouts (if any) show the operation the failures got wrong.
- If NO rollout failed, emit [].

For each error, state the POSITIVE corrective recipe (the operation/formula/format/save step that would have produced the correct graded values) and the wrong approach it rules out. Common failure modes worth capturing: leaving bare formulas with no cached value, writing to the wrong sheet/range, losing date number_format, off-by-one range anchoring, or not saving to the expected output path. Do NOT write blanket "re-check everything" advice — write the concrete operation that fixes the specific error. If the failure was not an operation issue, emit [].

{schema}

<question>
{question}
</question>

<summaries>
{summaries}
</summaries>

<ground_truth>
{groundtruth}
</ground_truth>
"""


# ---------------------------------------------------------------------------
# Router-context "examples" sidecar (<slug>_examples.txt)
# ---------------------------------------------------------------------------
# The two prompts below back the LLM-summarize + LLM-merge consolidation
# pipeline for the per-skill examples sidecar. The sidecar is rendered into
# the router prompt (format_skills_list) and into the two-step strict judge
# (JUDGE_SKILL_STRICT_PROMPT), so its content must stay specific enough to
# disambiguate sibling buckets — DO NOT collapse summaries into category
# names.

SUMMARIZE_TASK_FOR_SKILL_PROMPT = """You are condensing a single task description into ONE short line that captures the *kind of problem* this skill was asked to solve. The line is shown to a router that picks among similar skill buckets, so it must let the router recognize "another task like this" later.

Skill bucket: {skill_name}
Skill description: {skill_description}

Rules:
- Output ONE line, <= 25 words, no trailing period.
- Keep concrete cues that distinguish this task from sibling skills: input form (e.g., "ruler image", "stacked bar chart", "SMILES string"), key operation (e.g., "conditional date arithmetic", "XBRL tag selection"), and any salient constraint.
- Drop instance-specific noise: file names, sheet names, exact numbers, attached-image paths, "please help" filler.
- DO NOT restate the skill description — describe the *task*, not the skill.
- DO NOT add a preamble. Output the line directly, nothing else.

<task>
{task}
</task>"""


MERGE_EXAMPLES_PROMPT = """You are consolidating a per-skill examples list used as router context. The list currently has {count} entries and exceeds the cap of {max_count}. Merge near-duplicates so the final list has AT MOST {max_count} entries and still spans the distinct task patterns the skill has seen.

Skill bucket: {skill_name}
Skill description: {skill_description}

Rules:
- Output AT MOST {max_count} lines. Fewer is fine if the inputs genuinely collapse.
- Each line <= 25 words, no trailing period, no numbering or bullets.
- Each line must still let a router say "this new task looks like that one" — keep input-form and operation cues, do NOT collapse into pure category labels.
- Merge only entries that share both input-form AND operation. Different patterns must stay as separate lines.
- Order: most distinct / most informative pattern first.
- Output the lines directly, one per line, nothing else (no JSON, no fences, no commentary).

<examples>
{examples}
</examples>"""
