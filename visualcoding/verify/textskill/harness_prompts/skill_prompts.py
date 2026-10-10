# -*- coding: utf-8 -*-

GENERATE_SKILL_PROMPT = """You are an AI agent architect. From the trajectory below, produce an ADVISORY SKILL.md — a brief reference the agent will read alongside (not instead of) its own judgment on similar future tasks.

### What advisory means
- The body is NEVER a procedure or a script. The agent already knows how to call tools and read images. Your job is to leave it alerts, not orders.
- Strong base models on simple tasks (e.g. "the value is printed on the chart, just read it") get worse when an SOP forces them through rotation / contrast / pixel-ratio / multi-step pipelines they did not need. Do NOT write such pipelines.

### MANDATORY output structure (no other sections allowed)
```
---
name: [SkillSlug]
description: [≤ 25 words, single sentence, names ONLY this skill's procedural signature: the action + target + (modifier if it defines the bucket)]
version: 1.0.0
---

# [Title]

## When to use
[1–2 sentences. What kind of question triggers this skill. No examples, no sub-bullets.]

## Strategy
[ONE sentence, ≤ 20 words. The single most important decision rule. NOT a procedure.
Good: "Prefer the value printed on the chart over any estimation."
Bad:  "Step 1: Identify target. Step 2: Normalize view. Step 3: ..."]

## Watch out for
- When [image-observable trigger condition], watch for [specific failure mode] — verify by [check].
- [unconditional bullet — a universal failure mode that applies regardless of image properties]
- [up to 3 bullets total; less is fine. Each ≤ 20 words.]
```

### Watch-out bullet style — WARNING-SHAPED, NOT PRESCRIPTIVE (HARD RULE)

**Each bullet describes a FAILURE TO AVOID, not an action to take.** This is the most important rule on this page — the agent uses Watch-outs as a checklist of pitfalls to be aware of, NOT as a recipe to execute. Prescriptive bullets ("when X, apply Y") cause strong models to over-process simple cases that loosely match the trigger; warning bullets ("when X, watch for failure mode Z") leave the action choice to the agent.

**FORBIDDEN bullet shapes** (rewrite or drop if you find yourself producing these):
- "When [trigger], **apply / run / use / do / consider** [operation]" ← prescriptive recipe
- "[trigger] requires / needs [operation]" ← prescription disguised as requirement
- Any bullet whose body after the trigger names a tool, preprocessing op, or sequence of operations to perform

**REQUIRED bullet shape** when a trigger condition exists:
- "When [trigger], watch for [specific failure mode]." (verify clause optional)
- "When [trigger], watch for [specific failure mode] — verify by [check]." (only when failure is hard to detect without an explicit check)

The verb stem after "When ..." MUST be `watch / be / check / verify` (passive observation), NOT `apply / run / use / do / consider / prefer` (active prescription). When a corrective action is the only way to phrase the warning, frame it as the *check* not the *action*: "verify by re-reading after [op]" beats "apply [op] then re-read".

**Verify clause is OPTIONAL — drop it when the failure is self-evident.** A bullet that says "When the image is rotated 90°, watch for incorrect line parsing" is complete on its own; tacking on "— verify by inspecting the text baseline" doubles the bullet's length without adding signal. Only include "verify by ..." when (a) the failure could plausibly slip past the agent's first read, AND (b) the verification step is non-trivial enough that the agent might not think of it. Each bullet ≤ 20 words.

The trigger MUST be checkable from the raw image at INITIAL inspection (no tool call needed). Examples of valid triggers: "When target text is < ~30px tall", "If the image is rotated 90/180°", "When multiple objects overlap in a single cluster". Examples of INVALID triggers (post-hoc / trajectory-only): "When OCR fails", "After cropping returns empty", "If the first count was wrong".

Bullets WITHOUT a trigger clause are allowed ONLY when the failure is genuinely universal (e.g. "Confusing punctuation with characters" applies to any text). If you can articulate a trigger, you MUST use it.

Aim for **≥ 50% of Watch-out bullets to start with "When ..." or "If ..."**. Two consecutive unconditional bullets in the same skill is a strong signal you're missing the trigger.

### Hard rules (REJECT yourself and rewrite if you violate any of these)
1. **NO code blocks**, no ```python, no shell, no pseudocode, NOTHING in triple backticks. The agent does not need scaffolding.
2. **NO numbered Workflow / Procedure / Steps section.** No "Step 1 / Step 2 / Step 3" anywhere in the body.
3. **NO Tool Templates section.** No template strings like "find_chart(image, ...)" or "estimate_from_axis(...)".
4. **NO Task Variants / Reading Patterns / Variant Workflows / Common Failure Modes / Final Check / Decision Principle / Core Principles** subsections. Only the three sections listed above.
5. **NO multi-step image pipeline** in prose either ("first rotate, then crop, then enhance contrast"). Each Watch out for bullet must be a SINGLE failure mode, stated independently. Do NOT name a tool or operation as the action of the bullet (see Warning-shaped rule above).
6. **No "Always" / "Must" / "Never deviate" / "Run X first" / "Apply X" / "Use X" / "Consider X" wording in conditional bullets** — these turn the bullet into a recipe. Use "watch for X" / "verify by X" / "[failure mode] is common" framings instead.
7. **Body ≤ 200 words total.** If you exceed, cut Watch-out bullets — start by removing the ones that are mostly common sense.
8. **No memorization of the case**: no exact answers, option letters, object names, crop coordinates, dates, numbers from the prompt, file paths, sample IDs, or benchmark-specific wording. Use placeholders like `[TARGET]`, `[QUERY]`.
9. **Description ≤ 25 words**, mentions ONLY this skill's own procedural signature. Do NOT mention adjacent buckets, alternative task framings, or topical categories that this skill does NOT actually cover.

### Input:
<trajectory>
{trajectory}
</trajectory>

<ground_truth>
{ground_truth}
</ground_truth>

Output ONLY the SKILL.md content starting with `---`. No preamble, no explanation."""


# ============================================================================
# Description-update rule blocks (interpolated into FUSE/MERGE/REFINE prompts).
# Two variants:
#   _DESCRIPTION_RULE_LOCKED:    multi-bucket default — description is bucket
#                                identity, frozen at create-time.
#   _DESCRIPTION_RULE_EXPANDABLE: single-skill-per-category mode — description
#                                may grow to reflect procedural families that
#                                have entered the body, so the test-time
#                                selector can correctly match cross-category
#                                queries onto the right category skill.
# ============================================================================

_DESCRIPTION_RULE_LOCKED = (
    "The `description` MUST stay ≤ 25 words, single sentence, and describe ONLY this bucket's own procedural signature. "
    "Do NOT broaden the description to absorb adjacent topics just because some tips brushed them. "
    "Preserve the existing `description` text byte-for-byte unless the existing text violates a hard format rule."
)

_DESCRIPTION_RULE_EXPANDABLE = (
    "The `description` MAY be expanded ONLY to cover procedural families already represented by Watch-out bullets in the (post-update) body. "
    "Do NOT add to the description any procedural family that is not represented by at least one bullet. "
    "Hard caps that always apply: ≤ 25 words, single sentence, no mention of adjacent buckets / sibling categories. "
    "If you do not have a substantive reason to expand, KEEP the existing `description` text byte-for-byte. "
    "When expanding, list procedural families separated by commas: 'Counts cars in fog' → 'Counts, identifies text, and identifies color in fog/haze degraded images'. "
    "Never write 'Handles all X tasks' or any phrase implying global coverage — be specific about which procedural families this skill actually covers."
)


MERGE_SKILL_PROMPT = """You are a knowledge architect. Your job is to fold one or more new per-sample skill drafts into a single unified ADVISORY skill document.

### What advisory means
- The merged skill is a brief reference the agent reads alongside its own judgment, NOT a procedure.
- The merged skill body has EXACTLY three sections: When to use, Strategy, Watch out for. NOTHING else.
- Each new draft brings candidate failure modes (Watch-outs). The valuable signal in a draft is its Watch-out bullets — not its Workflow / Tool Templates / Code (those should be discarded outright).

### Frontmatter Discipline (HARD RULES)
- Preserve the existing frontmatter `name`, `version`, (if present) `prototype_signature`, and (if present) `image_conditions` fields verbatim. Do NOT alter `prototype_signature`; do NOT drop or rewrite `image_conditions` (the latter is a Stage-2 routing anchor populated from training-image evidence).
- {description_rule}
- If a draft's content does not procedurally fit this bucket's `prototype_signature`, DROP that draft rather than absorbing it.

### Body integration rules
1. **ADDITIVE-FIRST.** Every Watch-out bullet currently in the existing skill is presumed correct. Do NOT remove existing bullets unless (a) a new tip directly contradicts that bullet with stronger evidence, OR (b) two existing bullets are exact duplicates and you are merging them into one. Wholesale rewriting that drops bullets without justification is forbidden.
2. **PRESERVE CONDITIONAL TRIGGERS.** Bullets that begin with "When ..." or "If ..." encode an explicit image-observable trigger that gates the advice. When merging or rewriting such a bullet, KEEP the trigger clause at the front. Do NOT collapse a conditional bullet into an unconditional rule.
3. **PROMOTE TO CONDITIONAL — WARNING-SHAPED, NOT PRESCRIPTIVE.** If a draft's Watch-out bullet is unconditional but the failure mode is actually gated by an image-observable property (small target, low contrast, rotation, occlusion, dense layout, glare, ...), rewrite it into a conditional WARNING — "When [trigger], watch for [specific failure mode] — verify by [check]." — NOT a prescriptive recipe like "When [trigger], apply [op]". The verb stem after "When ..." MUST be `watch / be / check / verify` (passive observation), NOT `apply / run / use / do / consider / prefer` (active prescription). Trigger MUST be checkable from the raw image at initial inspection (no tool call needed); post-hoc / trajectory-only triggers ("when OCR fails") are forbidden. Aim for ≥ 50% of merged Watch-out bullets to start with "When ..." or "If ...".
4. **REWRITE INCOMING PRESCRIPTIVE BULLETS.** If a draft contains a bullet shaped like "When X, apply / run / use Y" (prescription gated by trigger), do NOT preserve that shape. Re-shape it into the warning form by extracting the failure mode that the prescription was implicitly trying to avoid: e.g. "When dim lighting, apply contrast enhancement" → "When dim lighting, watch for OCR misreads on faint character strokes — verify by reading twice and reconciling."
5. Discard everything in each draft EXCEPT the Watch-out bullets and any single-sentence Strategy hint that is genuinely a decision rule (not a procedure).
6. From each draft's Watch-outs, keep bullets that describe a *failure mode the agent might actually fall into* on similar future tasks. Drop bullets that are common sense or restate the strategy.
7. Merge near-duplicate Watch-outs into one cleaner bullet (this is allowed loss-of-bullet-count, but no loss of substantive failure mode). When merging two conditional bullets that share a trigger, keep the trigger; when merging two with different triggers, keep both as alternatives ("When X or Y, ...").
8. **TRIGGER-FAMILY MERGE.** If two bullets' triggers describe the same image-observable property family — e.g. both about lighting/contrast ("dim lighting" / "low contrast" / "intense glare"), both about rotation/orientation ("rotated 90°" / "skewed" / "sideways text"), both about size/density ("small target" / "distant" / "small text region") — merge them into ONE bullet with the broader trigger and the union of failure modes. Do NOT keep parallel bullets that differ only in trigger granularity. v9 had 4 lighting-related bullets in one skill — that is exactly the pattern this rule eliminates.
8. Keep the existing skill's When-to-use and Strategy unless the new drafts collectively reveal a clearer formulation. If you do rewrite, stay within the strict body format below AND keep all the same intent.

### MANDATORY body format (the merged skill MUST conform)
```
# [Title]

## When to use
[1–2 sentences max.]

## Strategy
[ONE sentence, ≤ 20 words. Anchor-naming decision rule (WHAT to ground in), NOT method-prescriptive ("prefer X over Y"). Method-prescriptive Strategy lines forbid the alternative path and propagate model-specific bias; rewrite as an anchor rule or move into a trigger-gated Watch-out.]

## Watch out for
- [advisory bullet 1]
- [advisory bullet 2]
- [...up to 3 bullets total, fewer is fine. Each ≤ 20 words.]
```

### Hard rules (REJECT yourself and rewrite if you violate any)
- **NO code blocks**, no ```python / ```shell / triple backticks of any kind.
- **NO numbered Workflow / Procedure / Steps section.**
- **NO Tool Templates / Reading Patterns / Task Variants / Variant Workflows / Final Check / Decision Principle / Core Principles** subsections.
- **NO multi-step pipeline** in prose ("first X, then Y").
- **No "Always" / "Must" / "Run X first"** wording.
- **Body ≤ 200 words.**
- **No memorization** of any specific case.

### Input:
<existing_skill>
{existing_skill}
</existing_skill>

<new_skills>
{new_skills}
</new_skills>

Output ONLY the merged SKILL.md starting with `---`. No preamble."""


FUSE_TIPS_INTO_SKILL_PROMPT = """You are a knowledge architect. Fold a batch of fresh experience tips into the existing ADVISORY SKILL.md.

### What advisory means
- The skill is a short reference the agent reads alongside its own judgment, NOT a procedure or script.
- Strong base models on simple tasks get worse when an SOP forces them through preprocessing pipelines they did not need (rotate / crop / contrast / pixel-ratio / multi-step chains). The skill body must NEVER prescribe such pipelines.
- The valuable signal in tips is the *failure modes the agent fell into* — those go under Watch out for. Procedure-flavored tips get either rewritten as a single decision rule or dropped.

### Inputs
- `base_skill`: an existing advisory SOP in markdown (may start with `---` frontmatter), possibly empty on the first batch.
- `tips`: {tip_count} short tips. Each is ≤ 64 words. Some may look procedural; do not preserve their procedural framing.

### Frontmatter Discipline
1. Preserve the existing frontmatter `name`, `version`, (if present) `prototype_signature`, and (if present) `image_conditions` verbatim. Do NOT alter `prototype_signature`; do NOT drop or rewrite `image_conditions` (Stage-2 routing anchor — its tags are populated from direct training-image evidence and survive across fuses).
2. {description_rule}
3. **Drop misaligned tips.** If a tip describes a procedure incompatible with this bucket's `prototype_signature`, DROP it — do not rewrite the body to absorb it.

### MANDATORY body format
The fused body has EXACTLY three sections, in this order:
```
# [Title]

## When to use
[1–2 sentences. Names the trigger condition for this skill, no examples, no sub-bullets.]

## Strategy
[ONE sentence, ≤ 20 words. Names WHAT to anchor on (the observation, feature, or quantity the agent should ground its answer in). It MUST NOT name WHICH METHOD to use or rank methods against each other.

Rationale: the test-time model may differ from this run's model (e.g. a more visual model at deployment time). A method-prescriptive Strategy ("prefer X over Y") propagates a model-specific bias and FORBIDS the alternative path even when that path is actually better for the deployment model — empirically this costs 5-15pp pass@2 on the affected bucket.

  GOOD (anchor-naming — leaves method choice open):
    - "Map the indicator position to labeled tick subdivisions to interpolate the value."
    - "Identify the slice by label and color, then quantify its share of the chart."
    - "Trace a continuous path through corridor centers from start to goal."

  BAD (method-prescriptive — forbids one path):
    - "Prioritize geometric analysis over subjective visual estimation."      ← bans visual
    - "Prefer printed values over visual estimates."                          ← bans estimation
    - "Use character-level OCR rather than direct visual counting."           ← bans direct count
    - "Rely on code-based pixel analysis instead of visual inspection."       ← bans visual

If you find yourself writing "prioritize / prefer / use X over Y", "instead of Z", "rather than Z", or "do not / never use Z" — rewrite the Strategy to name only the anchor. NOT a procedure either.]

## Watch out for
- [advisory bullet 1]
- [advisory bullet 2]
- [...up to 3 bullets total, fewer is fine. Each ≤ 20 words.]
```

### Integration rules
1. **ADDITIVE-FIRST.** Every Watch-out bullet that already exists in the base skill is presumed correct guidance. Do NOT remove existing bullets unless (a) a new tip directly contradicts that bullet with concrete failure evidence, OR (b) two existing bullets are near-duplicates being merged into one cleaner bullet. Adding new bullets is the default; deleting needs a reason.
2. **PRESERVE CONDITIONAL TRIGGERS, BUT REWRITE PRESCRIPTIVE BODIES.** Tips that begin with "When ..." or "If ..." encode an explicit trigger that gates the advice — KEEP the trigger clause at the front. HOWEVER, if the tip's body after the trigger is prescriptive ("apply / run / use / consider / do [op]"), REWRITE the body into a warning shape: "watch for [failure mode] — verify by [check]". Prescription gated by a trigger is still prescription; loose triggers cause the recipe to fire on tasks that don't need it. The body verb stem MUST be `watch / be / check / verify` (passive observation), NOT `apply / run / use / do / consider / prefer` (active prescription).
3. Route each tip:
   - tip describes a *failure mode* / *common mistake* / *trap* → distill as a Watch-out bullet in warning shape ("[trigger,] watch for [failure] — verify by [check]").
   - tip describes a *single anchor-naming decision rule* (e.g. "ground the count in tick-mark subdivisions", "identify slice by its label color") → fold into Strategy if absent, else drop. Method-prescriptive rules ("prefer / prioritize / use X over Y") MUST be rewritten as either an anchor rule (drop the comparison) or as a Watch-out warning ("When the alternative path looks tempting, watch for [its specific failure mode] — verify by ...") before adopting.
   - tip describes a *procedure / pipeline / multi-step recipe* WITHOUT a trigger condition → DROP entirely. Unconditional procedural framing is the failure mode the agent must NOT inherit.
   - tip describes a *procedure WITH a trigger condition* (e.g. "When X, consider sequence Y") → KEEP the trigger but REWRITE the body as the failure mode that procedure was implicitly trying to avoid (see rule 2). DO NOT preserve "consider sequence Y" verbatim.
   - tip describes a *tool template / code snippet* → DROP entirely.
4. Rewrite tips in the document's advisory voice. Do NOT paste them as raw "[E12] ..." entries. Do NOT keep experience IDs.
5. Merge near-duplicate Watch-outs into one clearer bullet (loss of bullet count OK; loss of substantive failure mode NOT OK). When merging two conditional bullets that share a trigger, keep the trigger; when merging two with different triggers, keep both as alternatives ("When X or Y, ...").
6. **TRIGGER-FAMILY MERGE.** If two bullets' triggers describe the same image-observable property family (lighting/contrast, rotation/orientation, size/density, occlusion/overlap, etc.), merge them into ONE bullet with the broader trigger and the union of failure modes — do NOT keep parallel bullets that differ only in trigger granularity. Aim for a final Watch-out section with ≤ 3 distinct trigger families.
7. Stay generalizable. Convert any concrete answer / option letter / object name / coordinate / date / file path / sample ID into placeholders.

### Output discipline
- **Body ≤ 200 words.** If you exceed, cut Watch-out bullets — start with common-sense bullets, never specific failure modes.
- The body has EXACTLY the three sections shown above. No `## Workflow` / `## Steps` / `## Phase` / `## Tool Templates` / `## Reading Patterns` / `## Variant Workflows` / `## Final Check` / `## Decision Principle` / `## Core Principles` subsections. No code blocks (no triple backticks of any kind). No multi-step pipeline in prose ("first X, then Y, then Z"). No "Always / Must / Run X first / Step 1" wording — body is permission-to-skip, not obligation.

Output ONLY the fused SKILL.md content, starting with `---` (or with `#` if the base skill had no frontmatter). No preamble, no closing commentary.

<base_skill>
{base_skill}
</base_skill>

<tips>
{tips}
</tips>
"""


# Structured-patch variant of the fuse prompt. The LLM does NOT rewrite the
# document; it emits a JSON patch that the system applies mechanically. This
# is the default fuse path (the free-form prompt above is kept as a fallback
# if patch JSON fails to parse).
FUSE_TIPS_INTO_SKILL_PATCH_PROMPT = """You are a knowledge architect. The agent has an existing ADVISORY SKILL.md (the FROZEN BASE) and a batch of fresh experience tips. Your job is to produce a STRUCTURED PATCH that the system will apply mechanically — you do NOT rewrite the document yourself.

### Inputs
- `base_skill`: the frozen base SKILL.md (its three-section advisory body and frontmatter). May be empty on the first batch.
- `tips`: {tip_count} short tips. Each ≤ 64 words. Some may look procedural; do not preserve their procedural framing.

### What advisory means
- The skill is a brief reference the agent reads alongside its own judgment, NOT a procedure.
- Strong base models on simple tasks get worse when an SOP forces them through preprocessing pipelines they did not need (rotate / crop / contrast / pixel-ratio / multi-step chains). The skill must NEVER prescribe such pipelines.
- The valuable signal in tips is the *failure modes the agent fell into* — those become Watch-out bullets. Procedure-flavored tips get either distilled into a Strategy decision rule or DROPPED.

### Output: a single JSON object, nothing else
```
{{
  "description": null | "new frontmatter description (≤ 25 words, single sentence)",
  "image_conditions": null | ["tag1", "tag2", ...],
  "title": null | "new H1 title (no leading #)",
  "when_to_use": null | "new When to use text (1–2 sentences)",
  "strategy": null | "new Strategy text (ONE sentence, ≤ 20 words; ANCHOR-NAMING, not method-prescriptive — see Strategy rule below)",
  "modify_bullets": [
    {{"from": "exact verbatim text of the existing Watch-out bullet (without leading - )",
      "to":   "the rewritten bullet text (without leading - )"}}
  ],
  "remove_bullets": [
    "exact verbatim text of an existing Watch-out bullet to remove (without leading - )"
  ],
  "add_bullets": [
    "new Watch-out bullet 1 (without leading - )"
  ]
}}
```

### Field semantics (READ CAREFULLY)
- `null` for `description` / `image_conditions` / `title` / `when_to_use` / `strategy` means "keep the existing text unchanged". Only set them to a string / list when you actually have a substantive replacement; do NOT cosmetically rephrase.
- For `description` specifically: {description_rule}
- For `image_conditions` specifically: this is a CLOSED-VOCABULARY tag list naming visible properties of the *kind of image* this skill is for — a Stage-2 routing anchor the test-time vision selector matches against. The full vocabulary is:
  `low_light, dim, haze, fog, overexposed, low_contrast, blur, rotated, upside_down, dense_clutter, small_text, text_heavy, scrambled_grid, side_by_side, color_swatch, line_drawing, geometry_diagram, chart_canvas, maze_grid, natural_scene, instrument_dial, occlusion`.
  Set this field to a JSON list ONLY when you have direct visual evidence (from the attached training images, if any) that this skill's intended scope IS one or more of these conditions. When the image attached to a tip clearly shows e.g. low_light, that is direct evidence for `low_light`. Do NOT infer the tags from tip text alone — they describe the IMAGE, not the prose. Leave the existing tag set alone (`null`) unless you have evidence to add. When set, the list REPLACES the existing tag list entirely (so include all tags you still want kept). Keep the list short (≤ 4 tags); prefer the most specific salient tags.
- `modify_bullets` / `remove_bullets` REFERENCE existing bullets by their exact verbatim text (the ASCII text after the leading `- `, no leading dash, no surrounding whitespace). If you cannot quote a bullet exactly, do not reference it — emit a fresh `add_bullets` entry instead.
- `add_bullets` are new Watch-out bullets to append. Each ≤ 20 words. The "verify by ..." clause is OPTIONAL — include it only when the failure mode is hard to detect without an explicit check; drop it when the failure is self-evident from observation.
- Apply order is fixed: modify → remove → add. So if a bullet is in both `modify_bullets.from` and `remove_bullets`, it ends up removed.

### Hard rules — ADDITIVE-FIRST (the system enforces this; violations get patches rejected)
1. Each entry in `remove_bullets` MUST cite either (a) a specific tip in this batch that DIRECTLY contradicts that bullet with concrete failure evidence, OR (b) a near-duplicate bullet you are merging (the merge target must appear in `modify_bullets` or `add_bullets`). The default for an existing bullet is to KEEP it.
2. `modify_bullets` is ONLY for substantive rewrites (sharper failure mode, factual correction, merging two near-duplicates into one). Do NOT use it to cosmetically reword a bullet.
3. Bound your patch: `add_bullets` ≤ 3 per batch, `remove_bullets` ≤ 2 per batch, `modify_bullets` ≤ 3 per batch. If you have more candidate changes, keep the highest-signal ones. **The total Watch-out section after the patch MUST stay ≤ 3 trigger families** (lighting/contrast, rotation/orientation, size/density, occlusion/overlap, etc.); if a new add_bullet's trigger family is already covered by an existing bullet, prefer `modify_bullets` (merge into the existing bullet) over `add_bullets` (new parallel bullet).
4. **PRESERVE TRIGGERS, REWRITE PRESCRIPTIVE BODIES.** If a tip starts with "When ..." or "If ...", KEEP the trigger clause when adding the bullet — but if the body after the trigger is prescriptive ("apply / run / use / consider / do [op]"), REWRITE the body into a warning shape before putting it into `add_bullets`. The verb stem after the trigger MUST be `watch / be / check / verify` (passive observation), NOT `apply / run / use / do / consider / prefer` (active prescription). Example: tip "When dim lighting, apply contrast enhancement" → add_bullet "When dim lighting, watch for OCR misreads on faint character strokes — verify by reading twice and reconciling." When `modify`-ing an existing bullet, do NOT strip its trigger AND do NOT introduce a prescriptive body; the `to:` text must be warning-shaped.

### Tip routing
- A tip describing a *failure mode* / *common mistake* / *trap* → propose an `add_bullets` entry shaped as "[trigger,] watch for [failure] — verify by [check]".
- A tip describing a *single anchor-naming decision rule* (e.g. "ground the count in distinct object boundaries", "identify the slice by its label and color") → fold into `strategy` (replace the whole Strategy text). If the existing Strategy already contains an equivalent rule, drop the tip. **Method-prescriptive tips ("prefer / prioritize / use X over Y", "instead of Z", "rather than Z") MUST NOT be written into `strategy`** — propagate a model-specific bias that breaks under deployment-time model changes. Either (a) rewrite the rule to name only the anchor and drop the comparison before folding, or (b) convert it into a Watch-out add_bullet ("When [alternative path] looks tempting, watch for [its specific failure mode] — verify by [check]") so it stays trigger-gated.
- A tip with a trigger AND prescriptive body (e.g. "When [trigger], consider [action]; verify by [rule]") → propose an `add_bullets` entry that PRESERVES the trigger but REWRITES the body as the failure mode the action was implicitly avoiding. DO NOT include "consider [action]" verbatim.
- A tip describing an unconditional *procedure / pipeline / multi-step recipe* → DROP entirely.
- A tip describing a *tool template / code snippet* → DROP entirely.

### Body discipline (apply to any text you write into title / when_to_use / strategy / bullets)
- NO code blocks, NO triple backticks.
- NO numbered Workflow / Steps / Phase wording inside any field.
- NO Tool Templates / Reading Patterns / Variant Workflows / Final Check / Decision Principle / Core Principles framing.
- NO multi-step pipeline in prose ("first X, then Y, then Z").
- NO "Always" / "Must" / "Run X first" / "Step 1" wording.
- Each bullet ≤ 20 words. Strategy ≤ 1 sentence (≤ 20 words). When-to-use ≤ 2 sentences.

### Empty-base case
If `base_skill` is "(empty — first batch)", set `title`, `when_to_use`, `strategy` to substantive strings (not null), populate `add_bullets` with the initial Watch-out bullets, and leave `modify_bullets` / `remove_bullets` as `[]`.

### Output rules
- Output ONLY the JSON object. No markdown fences, no commentary, no preamble, no closing remarks.
- Use double-quoted strings. Use `null` (not `"null"`) for keep-unchanged fields.

<base_skill>
{base_skill}
</base_skill>

<tips>
{tips}
</tips>
"""


# Appended to the routing prompt only when skill modality == 'excel'. Spreadsheet
# buckets must be NARROW and procedurally specific: a broad bucket (e.g.
# action=extract, target=data-aggregation, modifier=*) gets selected for many
# unrelated tasks at test time and its grab-bag tips mislead them.
ROUTE_SAMPLE_FOR_FUSION_EXCEL_ADDENDUM = """### EXCEL-SPECIFIC ROUTING OVERRIDES (highest priority)

**Aim for MID-GRAINED buckets — a recurring procedural family, NOT one task.** The test-time selector can only inject a bucket when a new task's signature matches it. A bucket scoped to a single quirky task will never match again and is dead weight; a bucket scoped to a whole task family gets reused. Target a granularity where you'd expect ROUGHLY 5–15% of spreadsheet tasks to fit.

- **Modifier = a reusable procedural condition**, at the granularity of these GOOD examples: `cross-row-lookup`, `workday-offset`, `multi-criteria-filter`, `date-parsing`, `multi-sheet-consolidation`, `running-total`, `dedup`, `group-aggregate`. Each names a *recurring* technique.
- **TOO NARROW — do NOT use** (one-off task descriptions disguised as modifiers; they never recur): `recursive-compounding`, `first-two-occurrences`, `conditional-formatting-fill`, `nth-workday-of-month`, `python-static-write`. If your modifier reads like a description of THIS one task, generalize it to the family it belongs to (e.g. `recursive-compounding` → `running-total`; `first-two-occurrences` → `positional-lookup`).
- **TOO BROAD — do NOT use** (umbrella tokens that match almost everything): modifier `conditional-logic` / `logic` / `calculation` / `computation` / `processing` / `criteria`; target `cell-value` / `value(s)` / `data` / `data-aggregation` / `result(s)`. Never `*`/`none` for a modifier. `(calculate, cell-value, conditional-logic)` is BANNED.
- A `target` may be a mild umbrella (e.g. `cell-value`) **only if** the `modifier` is a specific mid-grained family. The modifier carries the discrimination.
- **Prefer `use` over `create` when a family bucket already fits.** Consolidate procedurally-similar tasks into ONE family bucket rather than minting a near-duplicate per task — broader coverage from fewer, reusable buckets beats many dead one-off buckets."""


ROUTE_SAMPLE_FOR_FUSION_PROMPT = """You are a skill-library router. Decide where a training sample's experience tips should live in the agent's skill library. The library is partitioned by **procedural signature** — the steps an agent must run — NOT by topical similarity.

### Step 1 — Infer the sample's procedural signature
Read BOTH the task wording AND the experience tips. Extract a 3-tuple:

- `action`   = the primary verb the agent must execute
- `target`   = what kind of thing the agent produces / extracts
- `modifier` = the contextual condition that materially changes the procedure (or "none" if no condition meaningfully changes the procedure)

**Vocabulary** — these slot values have appeared in the current library; treat them as a soft prior. You MAY introduce a new value when none of these fit, but do NOT invent a value just to differentiate from an existing bucket if the procedure is actually the same:
{vocabulary_hint}

**Inference principles** (no fixed task taxonomy — judge from this sample's content):
- A `modifier` is worth setting only when it changes WHICH steps the agent runs (e.g. it forces a unique preprocessing chain, a different verification routine, or a distinct decoding strategy). Surface features like "the image is colorful" or "there is text in the scene" are NOT modifiers unless they change the procedure.
- The modifier often surfaces in the tips rather than the task wording — read both.
- If two candidate signatures both fit, pick the one whose slots most directly describe the steps that *separate* this sample from samples that need a different procedure.

### Step 2 — Match against existing buckets
Each bucket has a `prototype_signature` where some slots may be wildcard (`*`). The new sample MATCHES a bucket iff every NON-WILDCARD slot of the prototype equals the sample's slot at that position.

Decision policy:
- **0 matches** → `create` a new bucket ONLY IF this sample's procedure is genuinely different from every existing bucket's. If it's a near-fit to an existing bucket and the differences are surface-level (topic / vocabulary / domain words rather than procedural steps), prefer to RELAX one slot of an existing bucket later via merge — for now, route to the closest existing bucket via `use`. Bias toward FEWER buckets.
- **1 match** → `use` that bucket.
- **≥2 matches** → `use` the most specific bucket (fewest `*`). Tie-break lexicographically.

### Hard constraints
- A new bucket's prototype MUST have **at least one** non-wildcard slot. `(*, *, *)` is never allowed.
- When creating, set slots to specific values only when those slots genuinely identify the bucket's procedure. Use `*` for any slot that varies arbitrarily across procedurally-similar tasks.
- **Test-time reachability.** The test-time skill selector sees ONLY the task wording — not the trajectory, not any intermediate observations. A bucket whose `modifier` value can be inferred ONLY from the trajectory / tool outputs / preprocessing-tips (and not from the task wording itself) will be unreachable at test time. When choosing a `modifier` value for a new bucket's prototype, prefer values the test-time selector can infer from the question alone; if the modifier is purely a property of the *trajectory* (e.g. a preprocessing step the agent ran), set the prototype's modifier to `*` and absorb the tip as a workflow branch inside the body.
- Library cap is a HARD STOP at {cap} buckets ({at_cap} = at_cap). When at cap, you MUST `use` an existing bucket; do not create. Do NOT pre-emptively merge to stay below cap.
- Slug format for new skills: lowercase, hyphen-separated, ≤24 chars, no punctuation other than hyphens. The slug should name the procedure (e.g. action+target compound), NOT a topic / domain / specific input type.
- New-skill descriptions: one sentence, ≤25 words, names ONLY this bucket's signature. No mention of adjacent buckets.

### Current library
{skills_list}

### Sample
Task: {task}

Experience tips extracted from this sample:
{tips}

### Output
Respond with a single JSON object, nothing else. Do not wrap in markdown fences.

For "use":
{{"signature": {{"action": "...", "target": "...", "modifier": "..."}},
  "action": "use",
  "name": "<existing-skill-slug>"}}

For "create":
{{"signature": {{"action": "...", "target": "...", "modifier": "..."}},
  "action": "create",
  "name": "<short-slug>",
  "prototype": {{"action": "...|*", "target": "...|*", "modifier": "...|*"}},
  "description": "<≤25 words, names only this bucket's signature>"}}
"""


MERGE_TWO_SKILLS_PROMPT = """You are a knowledge architect. The skill library has grown past its cap and you must consolidate the two most-similar skills into ONE ADVISORY skill.

### What advisory means
- The merged skill body has EXACTLY three sections: When to use, Strategy, Watch out for. NOTHING else.
- Procedure / Workflow / Tool Templates / Code Blocks from either input are DISCARDED — they are not preserved by merging. Only Watch-outs and (≤2 sentences) Strategy survive.

### Frontmatter rules
1. Pick the clearer of the two `name` slugs (or propose a new one).
2. Write a fresh `description`: ≤ 25 words, single sentence, names ONLY the merged bucket's procedural signature. Do NOT mention adjacent buckets.
3. Bump `version` by 0.1.
4. Compute a unified `prototype_signature`: for each slot, if both inputs agree set to that value, otherwise set to `*`. The merged prototype MUST keep at least one non-wildcard slot.

### Body merge rules
1. Take the When-to-use from whichever input is clearer (or write a new one). 1–2 sentences total.
2. Take the Strategy from whichever input states the cleanest **anchor-naming** decision rule (or merge into one). **ONE sentence, ≤ 20 words.** Method-prescriptive Strategies ("prefer / prioritize / use X over Y", "instead of Z", "rather than Z") MUST be rewritten as anchor rules before merging — the comparison clause is dropped because it forbids a path the deployment model may actually need.
3. **ADDITIVE-FIRST.** Union the Watch-out bullets from both inputs. Merge near-duplicates into one cleaner bullet (no loss of substantive failure mode). Drop only: (a) exact duplicates after merging, (b) common-sense bullets ("be careful", "verify result"), (c) bullets that contradict the unified strategy with no supporting evidence. **Preserve conditional triggers.** Bullets starting with "When ..." or "If ..." carry an explicit trigger that gates the advice; keep the trigger at the front when you rewrite or merge such a bullet. Do NOT collapse a conditional bullet into an unconditional rule.
4. Cap Watch-outs at 8 bullets total ONLY by consolidating near-duplicates further; DO NOT drop a unique specific failure-mode bullet just to fit the cap.

### Hard rules (REJECT yourself and rewrite if you violate any)
- **NO code blocks**, **NO numbered Workflow**, **NO Tool Templates**, **NO Task Variants / Reading Patterns / Variant Workflows / Common Failure Modes / Final Check / Decision Principle / Core Principles** subsections.
- **NO multi-step pipeline** in prose.
- **No "Always" / "Must" / "Run X first" / "Step 1" wording.**
- **Body ≤ 200 words.**

### Skill A (name: {name_a})
{skill_a}

### Skill B (name: {name_b})
{skill_b}

Output ONLY the merged SKILL.md starting with `---`. No preamble."""


PICK_MERGE_PAIR_PROMPT = """Given the following list of skill names, prototype signatures, and descriptions, identify the TWO that are most procedurally similar — the pair most worth merging to reduce library redundancy.

### Definition of procedural similarity
Two skills are similar iff their `prototype_signature` triples match on at least 2 of 3 slots when wildcards are treated as compatible with anything. Topical overlap in the description does NOT count.

### Skills
{skills_list}

### Output
Respond with a single JSON object, nothing else:
{{"pair": ["<name_a>", "<name_b>"], "reason": "<one-sentence justification grounded in the prototype_signatures>"}}"""


SKILL_REFINE_PROMPT = """You are a skill document architect. Aggressively refactor SKILL.md into ADVISORY format. The body must shrink and become a brief reference, NOT a procedure.

### Current Stats
- Word count: {word_count}

### Why advisory matters
Strong base models on simple tasks get worse when an SOP forces them through preprocessing pipelines they did not need. The body's job is to let the agent skip mistakes, not to choreograph its tools — never prescribe such pipelines.

### Frontmatter Discipline (HARD RULES — DO NOT VIOLATE)
1. Preserve the frontmatter block exactly: `name`, `version`, (if present) `prototype_signature`, and (if present) `image_conditions` MUST stay byte-for-byte identical. Do NOT alter `prototype_signature`; do NOT drop, reorder, or rewrite `image_conditions` (Stage-2 routing anchor — populated from direct training-image evidence).
2. {description_rule}

### MANDATORY body format (output MUST conform)
The refactored body has EXACTLY three sections, in this order, and nothing else:
```
# [Title]

## When to use
[1–2 sentences. Trigger condition. No examples, no sub-bullets.]

## Strategy
[ONE sentence, ≤ 20 words. Anchor-naming decision rule. MUST NOT be method-prescriptive ("prefer / prioritize / use X over Y", "instead of Z") — such phrasing forbids the alternative path even when it would have worked for the deployment model. Rewrite into an anchor rule (drop the comparison) or move the warning into a trigger-gated Watch-out. NOT a procedure.]

## Watch out for
- [advisory bullet 1 — a specific failure mode]
- [advisory bullet 2]
- [...up to 3 bullets total, fewer is fine. Each ≤ 20 words.]
```

### Refactor rules
1. **Delete forbidden content.** Strip all code blocks (triple backticks), numbered Workflow / Procedure / Steps / Phase sections, Tool Templates / Reading Patterns / Variant Workflows / Task Variants / Final Check / Decision Principle / Core Principles / Common Failure Modes subsections, multi-step pipelines in prose, and "Always / Must / Run X first / Step 1" wording. When a forbidden section contains a specific failure-mode signal, EXTRACT that signal as a Watch-out bullet — do NOT silently drop the underlying point.
2. **ADDITIVE-FIRST.** Every Watch-out bullet in the source is presumed correct. Removing one is allowed ONLY when it (a) duplicates another bullet (merge them), OR (b) is a generic platitude with no specific failure mode (e.g. "be careful", "verify the result"). Specific, non-obvious failure modes MUST be preserved.
3. **PRESERVE CONDITIONAL TRIGGERS.** Bullets starting with "When ..." or "If ..." encode an image-observable trigger that gates the advice. When rewriting such a bullet, keep the trigger clause at the front. Stripping the trigger turns conditional advice into unconditional procedure — the failure mode this format exists to prevent.
4. **Length budget: body ≤ 200 words.** If you must trim, drop generic / common-sense bullets first; never drop a specific conditional bullet to satisfy the cap.

**Aim for ≤ 3 Watch-out bullets after consolidation.** Each bullet ≤ 20 words. Use TRIGGER-FAMILY MERGE: if multiple bullets share an image-observable property family (lighting/contrast, rotation/orientation, size/density, occlusion/overlap), merge them into ONE bullet with the broader trigger and the union of failure modes. Final state should have ≤ 3 distinct trigger families.

Output ONLY the refined SKILL.md starting with `---`. No preamble, no commentary on what you removed.

<current_skill>
{skill_content}
</current_skill>
"""


SKILL_FORMAT_FIX_PROMPT = """The SKILL.md below has format violations (already enumerated below by a static checker). Repair each listed violation while PRESERVING ALL substantive content.

### Format violations detected
{errors}

### Repair rules
1. **Frontmatter stays byte-identical** — `name`, `description`, `version`, `prototype_signature` (if present), and `image_conditions` (if present) cannot change. The description text may only shorten if the violation requires it.
2. **Body has exactly three sections in order**: `## When to use`, `## Strategy`, `## Watch out for`. Move content from forbidden sections into the appropriate required section (most often `Watch out for`); do NOT silently drop the underlying failure-mode signal.
3. **Preserve every substantive Watch-out bullet** — especially those starting with "When ..." or "If ..." (the trigger gates that prevent over-application). Trim only duplicates or generic platitudes ("be careful", "verify result"). Body ≤ {max_body_words} words; if you must trim, start with common-sense bullets, never specific failure modes.

Output ONLY the repaired SKILL.md starting with `---`. No preamble, no explanation, no commentary on what you changed.

### Current content (broken)
{current_content}
"""


# ============================================================================
# TEXT-KNOWLEDGE variants (MCE text tasks; selected when args.text_knowledge_skills).
#
# Same mechanical contract as the visual prompts (patch-JSON schema for fuse,
# three-section body for refine, so the static format checker and mechanical
# patch-apply keep working) — but the "Watch out for" bullets carry ADDITIVE
# DOMAIN KNOWLEDGE (concrete facts / confusable-pair discriminators) instead
# of generic process warnings. The whole point: on closed-set text tasks the
# base model already has strong priors, so vague "be careful / re-examine"
# advice makes it second-guess CORRECT answers. Only concrete, non-obvious
# domain facts that flip a borderline case belong in the skill.
# ============================================================================

FUSE_TIPS_INTO_SKILL_PATCH_PROMPT_TEXT = """You are a knowledge architect for a TEXT reasoning task. The agent has an existing ADVISORY SKILL.md (the FROZEN BASE) and a batch of fresh experience tips. Produce a STRUCTURED PATCH the system applies mechanically — you do NOT rewrite the document yourself.

### Inputs
- `base_skill`: the frozen base SKILL.md (three-section advisory body + frontmatter). May be empty on the first batch.
- `tips`: {tip_count} short tips, each a concrete domain fact or a confusable-pair discriminator.

### What this skill is for
- The skill is a brief reference the agent reads ALONGSIDE its own judgment, NOT a procedure.
- The base model is already a competent domain reasoner. Generic advice ("read carefully", "consider all options", "verify the answer", "evaluate the whole cluster", "watch for bias") makes it SECOND-GUESS correct answers — it is net-harmful. NEVER add such bullets.
- The only content worth keeping is ADDITIVE, NON-OBVIOUS DOMAIN KNOWLEDGE: a specific fact, a feature->label mapping, or a discriminator between two genuinely confusable labels (A vs B + the distinguishing cue).

### Output: a single JSON object, nothing else
```
{{
  "description": null | "new frontmatter description (<= 25 words, single sentence)",
  "image_conditions": null,
  "title": null | "new H1 title (no leading #)",
  "when_to_use": null | "new When to use text (1-2 sentences)",
  "strategy": null | "new Strategy text (ONE sentence, <= 20 words; names the domain anchor to ground decisions in, NOT method-prescriptive)",
  "modify_bullets": [
    {{"from": "exact verbatim existing bullet (without leading - )",
      "to":   "rewritten bullet (without leading - )"}}
  ],
  "remove_bullets": [
    "exact verbatim existing bullet to remove (without leading - )"
  ],
  "add_bullets": [
    "new knowledge bullet 1 (without leading - )"
  ]
}}
```

### Field semantics
- `null` for description / title / when_to_use / strategy means "keep unchanged". `image_conditions` is ALWAYS null for text tasks (no images).
- For `description` specifically: {description_rule}
- `modify_bullets` / `remove_bullets` cite existing bullets by exact verbatim text (after the leading `- `). If you cannot quote exactly, emit a fresh `add_bullets` entry instead.
- Apply order is fixed: modify -> remove -> add.

### Bullet content rules (the substance of this variant)
- Every `add_bullets` entry MUST be a BIDIRECTIONAL, FEATURE-BASED discriminator: it gives the hallmark features of EACH confusable label and lets the case decide, rather than naming a single winner.
  - Good: "Scaly rash — psoriasis: silvery plaques on extensor surfaces + nail pitting; fungal infection: ring-shaped lesion with central clearing and active border. Pick by which is present."
  - REJECT (one-directional, overfits the winner): "When the rash is scaly, answer psoriasis." / "Flaky plaques → psoriasis, not fungal."
  - REJECT (generic nudge): "Watch for confusing similar diseases — verify the full symptom cluster."
- A bullet may name only ONE label ONLY when it is a standalone, non-confusable fact (no competing label exists). Whenever two labels are confusable, the bullet MUST describe BOTH sides' features.
- Do NOT default: if a label's hallmark features are absent in a case, the rule must allow NOT choosing it. Phrase discriminators so they fire in either direction.
- DROP any tip that is generic process advice, restates the task, or just says "be careful / verify", and DROP one-directional "cue -> winner" tips whose competing label has no stated features.
- CLOSED-SET LABELS ONLY: only use label names from the task's own answer set; never introduce an off-list near-synonym (it can never be the graded answer).
- Each bullet <= 35 words.

### Patch bounds (enforced)
1. `remove_bullets` only for (a) a bullet a tip directly contradicts with evidence, or (b) a near-duplicate being merged. Default is KEEP.
2. `modify_bullets` only for substantive rewrites (sharper fact, correction, merge), not cosmetic rewording.
3. `add_bullets` <= 3, `remove_bullets` <= 2, `modify_bullets` <= 3 per batch. Keep the Watch-out section <= 6 bullets total; merge same-topic discriminators rather than adding parallel ones.

### Generalization filter (commit only what transfers)
These tips come from a SMALL batch of training rollouts. A pattern that merely held in this batch is NOT yet a rule — committing it as one biases every future case and flips examples that are actually the other label (the dominant failure mode of this skill).
- COMMIT a bullet only when its distinguishing feature is CAUSAL or DEFINITIONAL — true because of what the label MEANS, not because it happened to correlate with the right answer in a few cases. (e.g. "fungal infection = ring-shaped lesion with central clearing" is definitional; "scaly rash usually means psoriasis here" is a batch frequency.)
- A feature supported by only 1-2 rollouts, or that names a winner without a defining reason, is WEAK EVIDENCE — drop it or keep it strictly bidirectional (state both labels' defining features, let the case decide).
- Prefer fewer, high-transfer discriminators over covering every observed case. Do NOT encode batch base-rates ("X is usually the answer"); they do not generalize to the test distribution.

### Body discipline
- NO code blocks, NO numbered Workflow/Steps/Phase, NO "Always / Must / Run X first" wording.
- Keep the three section headers exactly: When to use / Strategy / Watch out for.

### Empty-base case
If `base_skill` is "(empty - first batch)", set `title`, `when_to_use`, `strategy` to substantive strings, populate `add_bullets` with the initial knowledge bullets, and leave `modify_bullets` / `remove_bullets` as [].

### Output rules
- Output ONLY the JSON object. No fences, no commentary. Double-quoted strings; `null` (not "null") for keep-unchanged fields.

<base_skill>
{base_skill}
</base_skill>

<tips>
{tips}
</tips>
"""


SKILL_REFINE_PROMPT_TEXT = """You are a skill document architect for a TEXT reasoning task. Refactor SKILL.md into ADVISORY format: a brief reference of ADDITIVE DOMAIN KNOWLEDGE, NOT a procedure.

### Current Stats
- Word count: {word_count}

### Why this matters
The base model is already a competent domain reasoner. The skill's job is to supply specific, non-obvious facts and confusable-pair discriminators it might miss — NOT to make it re-examine cases it would have gotten right. Generic "be careful / verify / consider all options" bullets make a strong model second-guess correct answers; they MUST be deleted.

### Frontmatter Discipline (HARD RULES)
1. Preserve the frontmatter block exactly: `name`, `version`, and (if present) `prototype_signature` / `image_conditions` stay byte-for-byte identical.
2. {description_rule}

### MANDATORY body format (output MUST conform)
Exactly three sections, in this order, nothing else:
```
# [Title]

## When to use
[1-2 sentences. Trigger condition. No examples, no sub-bullets.]

## Strategy
[ONE sentence, <= 20 words. Names the domain anchor to ground decisions in. NOT method-prescriptive, NOT a procedure.]

## Watch out for
- [bidirectional, feature-based discriminator — gives the hallmark features of EACH confusable label and decides by which is present, <= 35 words]
- [...up to 6 bullets total, fewer is better.]
```

### Refactor rules
1. **Delete forbidden content.** Strip code blocks, numbered Workflow/Procedure/Steps/Phase, Tool Templates, multi-step pipelines, "Always / Must / Run X first" wording.
2. **Delete generic nudges.** Any bullet that is process advice ("read carefully", "verify the answer", "consider all options", "evaluate the whole cluster", "watch for bias") has zero domain content — DELETE it. Keep ONLY bullets carrying a specific, reusable contrastive feature.
3. **Rewrite one-directional rules into bidirectional ones.** A bullet of the form "When <cue>, answer <label A>" (names a winner, gives the competing label no features) OVERFITS the winner and flips cases that are actually the other label — the main failure mode of this skill. Rewrite it to state the hallmark features of BOTH labels and decide by feature-matching; if the competing label's features cannot be recovered, DELETE the bullet. A single-label bullet survives ONLY for a standalone fact with no confusable counterpart.
4. **Preserve & sharpen real discriminators.** Bullets that already contrast both sides' features are the asset — keep them, merge near-duplicates.
5. **Delete batch-overfit bullets.** This body accreted from small training batches. DELETE any bullet that asserts a batch base-rate ("X is usually the answer", "scaly rash → psoriasis") rather than a CAUSAL/DEFINITIONAL feature (true because of what the label means). When in doubt about whether a feature transfers beyond the training cases, drop it — a wrong-but-confident discriminator is worse than none.
6. **Length budget: body <= 250 words.** Trim by deleting generic / one-directional / batch-overfit bullets first; never drop a genuine causal bidirectional discriminator to satisfy the cap.

Output ONLY the refined SKILL.md starting with `---`. No preamble, no commentary.

<current_skill>
{skill_content}
</current_skill>
"""


# ============================================================================
# EXCEL / SPREADSHEET-OPERATION variants (selected when skill modality=='excel').
#
# Same mechanical contract as the visual/text prompts (patch-JSON schema for
# fuse, three-section body for refine), but the "Watch out for" bullets carry
# concrete OPERATION RECIPES tied to how the workbook is GRADED (cell values in
# a target range), NOT visual watch-outs and NOT closed-set domain facts.
# ============================================================================

FUSE_TIPS_INTO_SKILL_PATCH_PROMPT_EXCEL = """You are a knowledge architect for an agentic SPREADSHEET (Excel) task. The agent has an existing ADVISORY SKILL.md (the FROZEN BASE) and a batch of fresh experience tips. Produce a STRUCTURED PATCH the system applies mechanically — you do NOT rewrite the document yourself.

### Inputs
- `base_skill`: the frozen base SKILL.md (three-section advisory body + frontmatter). May be empty on the first batch.
- `tips`: {tip_count} short tips, each a concrete spreadsheet operation recipe or a wrong-approach discriminator.

### What this skill is for
- The skill is a brief reference the agent reads ALONGSIDE its own judgment, NOT a procedure to follow step-by-step.
- The base agent already knows generic spreadsheet basics. Generic advice ("inspect the file", "verify the output", "be careful with edge cases", "double-check the structure") makes it OVER-ENGINEER tasks it would have solved directly — it is net-harmful. NEVER add such bullets.
- The only content worth keeping is an ADDITIVE, NON-OBVIOUS OPERATION RECIPE: a specific formula/function choice, a range-anchoring rule, a data-type / number_format handling rule, or a write/save step — each tied to producing the correct GRADED cell values.

### SINGLE-PURPOSE BUCKET (critical — prevents over-broad skills)
- This bucket covers ONE narrow procedural family (its `description` + `prototype_signature` define it). Keep it that way.
- A tip that belongs to a DIFFERENT family (e.g. cell-border styling when this bucket is about aggregation, or date-parsing when this bucket is about sorting) MUST NOT be added here — DROP it; it belongs to (or will form) another bucket. A broad grab-bag bucket gets mis-selected for unrelated tasks and hurts accuracy.
- Do NOT broaden `description` to absorb a new family. If a tip's recipe is on-family, add it; otherwise drop it.

### Output: a single JSON object, nothing else
```
{{
  "description": null | "new frontmatter description (<= 25 words, single sentence)",
  "image_conditions": null,
  "title": null | "new H1 title (no leading #)",
  "when_to_use": null | "new When to use text (1-2 sentences)",
  "strategy": null | "new Strategy text (ONE sentence, <= 20 words; names the operation anchor to ground decisions in, NOT method-prescriptive)",
  "modify_bullets": [
    {{"from": "exact verbatim existing bullet (without leading - )",
      "to":   "rewritten bullet (without leading - )"}}
  ],
  "remove_bullets": [
    "exact verbatim existing bullet to remove (without leading - )"
  ],
  "add_bullets": [
    "new operation-recipe bullet 1 (without leading - )"
  ]
}}
```

### Field semantics
- `null` for description / title / when_to_use / strategy means "keep unchanged". `image_conditions` is ALWAYS null for spreadsheet tasks (no images).
- For `description` specifically: {description_rule}
- `modify_bullets` / `remove_bullets` cite existing bullets by exact verbatim text (after the leading `- `). If you cannot quote exactly, emit a fresh `add_bullets` entry instead.
- Apply order is fixed: modify -> remove -> add.

### Bullet content rules (the substance of this variant — CONCRETE OR DISCARD)
- **HARD RULE: every `add_bullets` entry MUST contain at least one inline backticked code/expression** like `cell.value = sum(vals)`, `WORKDAY(ref, -N)`, `iter_rows(values_only=True)`, `SUMPRODUCT(...)`. A bullet with prose-only "watch for X — verify by Y" wording and ZERO backticks will be **mechanically filtered out** before commit by the downstream `filter_vague_excel_bullets` pass. If you can't name a specific code token, **drop the tip** — emit fewer bullets rather than vague ones.
- The backticked token must be the actual operation, not a decorative word ("`group`" doesn't count — use the function call: "`df.groupby(col).sum()`").
- **REQUIRED shape:** "When [task/data condition], do `[concrete op/code]` — not `[wrong op]`, which [why it fails the value/range check]." Include the wrong→right contrast whenever the failure was a specific wrong approach.
  - Good: "When the grader reads cell VALUES, write the computed result: `cell.value = sum(vals)` — not `cell.value = '=SUM(...)'`, whose cached value is None under openpyxl `data_only=True`."
  - Good: "When summing the last N workdays from a ref cell, anchor with `WORKDAY(ref, -N)` — not a fixed row offset, which breaks when rows shift."
  - REJECT (zero operation content): "Verify the output range carefully before saving." / "Be careful with date formats."
- **BAN the defensive form, even WITH a backtick.** Bullets shaped as "watch for X — verify `Y`" / "ensure `Z` is set" / "be careful of W — check `C`" are defensive noise: they slip a code token past the backtick filter but make the agent hunt for problems that usually are not there and OVER-ENGINEER a task it would have solved directly (the measured failure mode). Rewrite as an IMPERATIVE recipe ("When [trigger], do `op`") or a wrong→right contrast, or DROP it. The backtick must be the operation the agent PERFORMS, not a value it should "check".
- **Trigger-gate every recipe so it self-suppresses.** Start each bullet with a SPECIFIC, NARROW data/task condition ("When a date cell holds an ISO string with a 'T' suffix…", "When deleting rows by a numeric threshold…"). A recipe with no concrete trigger — or one that would fire on essentially every task — gets applied where it does not belong and lowers accuracy. If you cannot name a narrow trigger, DROP the tip.
- **Specificity ∝ fragility:** if the SAME mechanical failure recurs across tips, pin it with EXACT code, not general phrasing.
- DROP any tip that is generic process advice, restates the task, or just says "be careful / verify".
- Each bullet <= 45 words (a short inline code expression does not count against readability — keep prose tight).

### Patch bounds (enforced)
1. `remove_bullets` only for (a) a bullet a tip directly contradicts with evidence, (b) a near-duplicate being merged, (c) a vague non-operational nudge that slipped into the base, or (d) an off-family bullet that does not match this bucket's purpose. Default is KEEP for concrete on-family recipes.
2. `modify_bullets` only for substantive rewrites (sharper recipe, add the missing code, correction, merge), not cosmetic rewording.
3. `add_bullets` <= 2, `remove_bullets` <= 2, `modify_bullets` <= 3 per batch. **Keep the Watch-out section <= 5 bullets total** — a focused bucket beats a long one. When at the cap, MERGE same-topic recipes or REPLACE a weaker bullet rather than appending a parallel one.

### Body discipline
- Keep the three section headers exactly: When to use / Strategy / Watch out for. Bullets SHOULD carry inline code in backticks (e.g. `cell.value = ...`, `WORKDAY(ref,-N)`, `iter_rows(values_only=True)`). Keep each bullet to a SINGLE line (no multi-line fenced blocks here — richer code examples are added later during refine). NO numbered Workflow/Steps/Phase, NO "Always / Must / Run X first" wording.

### Empty-base case
If `base_skill` is "(empty - first batch)", set `title`, `when_to_use`, `strategy` to substantive strings, populate `add_bullets` with the initial operation recipes, and leave `modify_bullets` / `remove_bullets` as [].

### Output rules
- Output ONLY the JSON object. No fences, no commentary. Double-quoted strings; `null` (not "null") for keep-unchanged fields.

<base_skill>
{base_skill}
</base_skill>

<tips>
{tips}
</tips>
"""


SKILL_REFINE_PROMPT_EXCEL = """You are a skill document architect for an agentic SPREADSHEET (Excel) task. Refactor SKILL.md into ADVISORY format: a brief reference of ADDITIVE OPERATION RECIPES, NOT a procedure.

### Current Stats
- Word count: {word_count}

### Why this matters
The base agent already knows generic spreadsheet basics. The skill's job is to supply specific, non-obvious operation recipes it might miss (formula/function choices, range anchoring, data-type / number_format handling, write/save steps that fix the GRADED cell values) — NOT to make it re-inspect tasks it would have solved directly. Generic "inspect / verify / be careful" bullets make a competent agent over-engineer; they MUST be deleted.

### Frontmatter Discipline (HARD RULES)
1. Preserve the frontmatter block exactly: `name`, `version`, and (if present) `prototype_signature` / `image_conditions` stay byte-for-byte identical.
2. {description_rule}

### Body format (output MUST conform)
Keep the three core sections, and OPTIONALLY add a patterns section with concrete code. Unlike the visual skill format, fenced code blocks and a short numbered workflow ARE allowed here — concrete, copyable operations are the asset for spreadsheet tasks.
```
# [Title]

## When to use
[1-2 sentences. Trigger condition. No sub-bullets.]

## Strategy
[ONE sentence, <= 25 words. Names the operation anchor to ground decisions in.]

## Watch out for
- [operation-recipe bullet — a concrete op / formula / format / save rule with inline code, or a right-vs-wrong discriminator, <= 45 words]
- [...up to 5 bullets total, fewer is better — a focused bucket beats a long one.]

## Useful patterns        ← OPTIONAL; include only if you have tested, concrete code
```python
# short, generalized snippets (no task-specific values): e.g.
cell.value = sum(vals)            # write computed value, not a formula string
ws.iter_rows(values_only=True)    # raw python values for comparison
```
```

### Refactor rules
0. **HARD RULE: every Watch-out bullet MUST contain at least one inline backticked code/expression** (e.g. `cell.value = ...`, `WORKDAY(...)`, `iter_rows(values_only=True)`). Prose-only bullets without any backticks will be mechanically filtered out by the downstream `filter_vague_excel_bullets` pass — and if NONE of your bullets carry backticks, the entire refine output is rejected. Emit FEWER bullets rather than vague ones.
1. **Keep concrete recipes & code; delete vague nudges AND defensive bullets.** A bullet that is process advice ("inspect the file", "verify the output", "double-check the structure", "be careful with edge cases") has zero operation content — DELETE it. Also DELETE the defensive form even when it carries a backtick — "watch for X — verify `Y`" / "ensure `Z` is set" / "be careful of W — check `C`": it sneaks a code token past the filter but makes a competent agent hunt for non-problems and over-engineer. Keep ONLY a bullet that is an IMPERATIVE operation ("When [narrow trigger], do `op`") or a wrong→right contrast — where the backtick is the op the agent PERFORMS, not a value it should "check". Each kept bullet must open with a SPECIFIC, NARROW trigger so it self-suppresses on tasks that do not match; delete any recipe that would fire on essentially every task.
2. **Stay single-purpose (anti-bloat).** This bucket covers ONE narrow procedural family (its `description` / `prototype_signature`). DELETE bullets that belong to a different family (e.g. border-styling in an aggregation bucket) — an over-broad grab-bag gets mis-selected for unrelated tasks and hurts accuracy. Do NOT broaden the description to justify keeping them.
3. **Make recipes specific.** Prefer the wrong→right contrast with inline code. "Use `WORKDAY(ref,-N)` not a fixed offset" beats "be careful with date ranges". Promote repeated mechanical failures into the `## Useful patterns` code block with an exact snippet.
4. **Preserve working code examples** already in the document — they are the asset; merge near-duplicates, generalize away task-specific values, sharpen.
5. **Length budget: body <= 300 words** and **<= 5 Watch-out bullets** (code blocks excluded from the spirit of the cap, but keep them short). Trim by deleting vague/off-family bullets first; never drop a concrete on-family recipe or a tested snippet to satisfy the cap.

Output ONLY the refined SKILL.md starting with `---`. No preamble, no commentary.

<current_skill>
{skill_content}
</current_skill>
"""


GENERATE_SKILL_PROMPT_EXCEL = """You are an AI agent architect for agentic SPREADSHEET (Excel) tasks. From the trajectory below, produce a SKILL.md draft — a concrete, reusable reference the agent reads on similar future workbooks.

### What good means here (UNLIKE the visual skill format, concrete code is the asset)
- Spreadsheet tasks are procedural and code-driven. The most useful content is a SPECIFIC operation recipe: a formula/function choice, a range-anchoring rule, a data-type / number_format handling, a write/save step — each tied to producing the correct GRADED cell values.
- Put actual code/expressions inline in backticks or in a fenced block. "Be careful with formulas" is useless; "`cell.value = sum(vals)` not `cell.value = '=SUM(...)'`" is useful.
- The grader reads cell VALUES in the exact answer sheet/range (openpyxl `data_only=True` → cached values), so anchor recipes to that.

### Output structure
```
---
name: [skill-slug-kebab-case]
description: [<= 25 words, single sentence, names this skill's action + target + modifier]
version: 1.0.0
---

# [Title]

## When to use
[1-2 sentences. What kind of spreadsheet task triggers this skill.]

## Strategy
[ONE sentence, <= 25 words. The single most important operation anchor.]

## Watch out for
- [concrete operation-recipe bullet with inline code, or a wrong->right discriminator, <= 45 words]
- [...up to 5 bullets, fewer is better — a focused skill beats a long one.]

## Useful patterns        (OPTIONAL; include only if the trajectory shows a tested, concrete pattern)
```python
# short, generalized snippets (NO task-specific values, NO this case's literal answer)
cell.value = sum(vals)            # write computed value, not a formula string
ws.iter_rows(values_only=True)    # raw python values for comparison
```
```

### Hard rules
1. Code blocks and a short workflow ARE allowed (this is the spreadsheet variant). Keep snippets short and GENERALIZED.
2. NO memorization of THIS case: no literal answers, exact cell coordinates, file paths, sample IDs, or prompt-specific numbers. Use placeholders like `[TARGET_RANGE]`.
3. Every Watch-out bullet must carry a concrete operation / formula / code or a wrong->right contrast, and must OPEN with a specific, narrow trigger condition so it self-suppresses on tasks that do not match. Drop vague "be careful / verify" bullets.
3a. BAN the defensive form even when it has a backtick — "watch for X — verify `Y`" / "ensure `Z` is set" / "be careful of W — check `C`". It slips a code token past the filter but makes the agent hunt for non-problems and over-engineer a task it would have solved directly. Rewrite as an IMPERATIVE op ("When [trigger], do `op`") or wrong→right, or drop it; the backtick must be the op the agent PERFORMS, not a value it should "check".
4. Body (excluding code blocks) <= 400 words. Description <= 25 words.

### Input:
<trajectory>
{trajectory}
</trajectory>

<ground_truth>
{ground_truth}
</ground_truth>

Output ONLY the SKILL.md content starting with `---`. No preamble, no explanation."""
