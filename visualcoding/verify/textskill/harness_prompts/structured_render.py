"""Structured-critique parsing and rendering, extracted VERBATIM from the agent harness `skill/experience_critique.py`
(lines 68-102 _parse_experience_response; lines 519-820: _STRUCTURED_FIELDS, _coerce_structured_entry, _parse_structured_response, _strip_prescriptive_lead,
_lower_first, _render_procedure, _render_structured_to_text, _truncate).  Vendored for textskill/build.py."""
import json
import re
from typing import Any, Dict, List, Optional


# --- the agent harness skill/experience_critique.py lines 68-102 ---
def _parse_experience_response(resp: str) -> List[dict]:
    """Parse a JSON list from the critique response without relying on fences."""
    if not isinstance(resp, str) or not resp.strip():
        return []

    decoder = json.JSONDecoder()
    candidates = []

    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", resp, re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1))

    candidates.append(resp.strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    for idx, ch in enumerate(resp):
        if ch != "[":
            continue
        try:
            parsed, _ = decoder.raw_decode(resp[idx:])
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed

    return []




_STRUCTURED_FIELDS = (
    "trigger_condition",
    "visual_failure",
    "successful_visual_ops",
    "tool_sequence",
    "verification_rule",
)


def _coerce_structured_entry(raw: Any) -> Optional[Dict[str, str]]:
    """Force a single LLM-emitted entry into the canonical 6-field shape.

    - Every field becomes a string (use "" for missing / null).
    - Unknown extra keys are silently dropped.
    - Returns None if `raw` is not a dict or all fields end up empty
      (then there's no signal worth keeping).
    """
    if not isinstance(raw, dict):
        return None
    out: Dict[str, str] = {}
    for k in _STRUCTURED_FIELDS:
        v = raw.get(k)
        if v is None:
            out[k] = ""
        elif isinstance(v, str):
            out[k] = v.strip()
        else:
            # Coerce list/dict/number to a string so downstream rendering
            # never explodes; concrete schema violation but recoverable.
            out[k] = str(v).strip()
    if not any(out[k] for k in _STRUCTURED_FIELDS):
        return None
    return out


def _parse_structured_response(resp: str) -> List[Dict[str, str]]:
    """Parse a JSON array of structured entries from an LLM response.

    Tolerates ```fences```, leading commentary, and missing/extra keys.
    Returns canonicalized entries; never raises.
    """
    raw_list = _parse_experience_response(resp)  # reuse list-parser
    out: List[Dict[str, str]] = []
    for item in raw_list:
        entry = _coerce_structured_entry(item)
        if entry:
            out.append(entry)
    return out


# Verbs that turn a tip into a recipe. We strip these from the head of the
# `successful_visual_ops` / `tool_sequence` fields when rendering them as a
# parenthetical hint, so the bullet body stays warning-shaped end-to-end —
# even the tail evidence reads as a noun phrase ("brightness enhancement"
# rather than "apply brightness enhancement").
_PRESCRIPTIVE_LEAD_VERBS = (
    "apply", "applying", "use", "using", "run", "running",
    "consider", "considering", "do", "doing", "perform", "performing",
    "execute", "executing", "prefer", "try", "trying", "invoke", "invoking",
)


def _strip_prescriptive_lead(text: str) -> str:
    """Strip a leading prescriptive verb (and trailing 'a/an/the') from a phrase.

    'apply contrast enhancement' → 'contrast enhancement'
    'use a tighter crop' → 'tighter crop'
    'crop tight then upscale 2x' → unchanged (no leading prescriptive verb)
    """
    if not text:
        return text
    parts = text.split(maxsplit=2)
    if not parts:
        return text
    first = parts[0].lower().rstrip(",.;:")
    if first not in _PRESCRIPTIVE_LEAD_VERBS:
        return text
    rest = text.split(maxsplit=1)[1] if len(text.split(maxsplit=1)) > 1 else ""
    # Also strip a leading article on the remainder ("the / a / an").
    rest_parts = rest.split(maxsplit=1)
    if rest_parts and rest_parts[0].lower() in ("a", "an", "the"):
        rest = rest_parts[1] if len(rest_parts) > 1 else ""
    return rest.strip()


def _lower_first(s: str) -> str:
    """Lowercase the first character only when it's not part of an acronym.


    Used to chain clauses gracefully after a leading "When ..." or
    "verify by ..." connective. We can't naively `.lower()[0]` because
    sentences often begin with uppercase acronyms ("OCR misread ...",
    "CLAHE then scan", "TV bezel") and lowercasing the first letter
    produces "oCR" / "cLAHE" / "tV" — visibly broken.

    Heuristic: if the first two characters are both uppercase letters,
    treat the first word as an acronym and leave it alone. Otherwise
    lowercase only the first character.
    """
    if not s:
        return s
    if not s[0].isupper():
        return s
    if len(s) >= 2 and s[1].isupper():
        # Looks like an acronym (OCR, CLAHE, TV, ...). Leave it.
        return s
    return s[0].lower() + s[1:]


def _render_procedure(seq: str, ops: str) -> str:
    """Format a tool_sequence (arrow-separated) into a compact prose recipe.

    Used by the success-side rendering path. Falls back to `ops` (stripped
    of its prescriptive lead verb) when `seq` is empty. Underscores become
    spaces and any of `->`, `→`, `>` between ops are normalized to a
    single `→` for compact display.
    """
    if seq:
        normalized = seq.replace("->", "→").replace(" > ", " → ")
        parts = [p.strip() for p in normalized.split("→") if p.strip()]
        parts = [p.replace("_", " ") for p in parts]
        if parts:
            return " → ".join(parts)
    if ops:
        return _strip_prescriptive_lead(ops)
    return ""


def _render_structured_to_text(entry: Dict[str, str]) -> str:
    """Render one structured entry as a single conditional bullet.

    Two shapes depending on the entry's lens:

    1. SUCCESS-SIDE (visual_failure == "" AND a concrete tool_sequence /
       successful_visual_ops is present) → POSITIVE PROCEDURE bullet:

           "When [trigger], a path that has worked: <op1 → op2 → op3>
           (verify by <verification_rule>)."

       This revives a controlled subset of the v8 "When X, consider Y"
       pattern that was previously demoted (see history below). The
       safeguards vs. v8 are:
         - the bullet is hard-gated by `trigger_condition` (image-observable
           BEFORE any tool runs — schema enforces this);
         - the verify clause is only kept when ≤ 12 words, so it stays
           informative without doubling bullet length;
         - the success branch fires ONLY when `visual_failure == ""`, so
           the failure-side analyst's output is unaffected.

    2. FAILURE-SIDE (visual_failure non-empty) → WARNING bullet:

           "When [trigger], watch for [failure] — verify by [rule]."

       Identical to the post-v8 rendering: ops/sequence are NEVER the
       bullet's action on this path. Field priority for the body remains
       failure → verify → ops (parenthetical tail).

    History (pre-Stage-2-fix): earlier renderings used "When [trigger],
    consider [ops]" which the LLM (and downstream fuse) interpreted as a
    recipe — gated by trigger but still prescriptive. On v8 this surfaced
    as bullets like "When dim lighting, apply contrast enhancement before
    cropping" which fired the over-processing pathology (-20pp pass@1 on
    attention_focusing). The current success-side branch above narrows
    that pattern to entries that come with BOTH an image-observable
    trigger AND a verifiable post-procedure check — the two anchors v8
    lacked. Pieces are dropped (not just shortened) when the source field
    is empty.
    """
    trigger = entry.get("trigger_condition", "").strip().rstrip(".")
    failure = entry.get("visual_failure", "").strip().rstrip(".")
    ops_raw = entry.get("successful_visual_ops", "").strip().rstrip(".")
    seq = entry.get("tool_sequence", "").strip().rstrip(".")
    verify = entry.get("verification_rule", "").strip().rstrip(".")

    # ------------------------------------------------------------------
    # SUCCESS-SIDE branch: positive procedure bullet, gated by trigger.
    # Only fires when the entry has no failure to warn about AND carries
    # a concrete recipe (either an arrow-separated tool_sequence or a
    # successful_visual_ops phrase).
    # ------------------------------------------------------------------
    if not failure and (seq or ops_raw):
        procedure = _render_procedure(seq, ops_raw)
        if procedure:
            if trigger:
                sentence = f"When {_lower_first(trigger)}, a path that has worked: {procedure}"
            else:
                # No trigger means the bullet is ungated — keep but mark as
                # advisory rather than recommended, so the agent doesn't run
                # it unconditionally.
                sentence = f"On similar past cases, this path has worked: {procedure}"
            # Verify clause kept only when it adds genuine post-procedure
            # confirmation guidance AND the procedure itself is short. Long
            # procedures (4+ ops) are already self-contained; appending a
            # verify clause just pushes the bullet past the truncate cap
            # (we slightly raised the cap to 30 for the success-side path
            # vs. the warning-side 22 — the procedure body legitimately
            # needs the extra room).
            n_ops = len([p for p in procedure.split("→") if p.strip()])
            if verify and len(verify.split()) <= 12 and n_ops <= 3:
                sentence += f" (verify by {_lower_first(verify)})"
            sentence = sentence.rstrip(". ") + "."
            return _truncate(sentence, max_words=30)

    # ------------------------------------------------------------------
    # FAILURE-SIDE branch: warning-shaped bullet (unchanged from v8 fix).
    # ------------------------------------------------------------------
    # Strip prescriptive lead verbs from the ops field. The raw analyst
    # output usually starts with "apply / use / consider [op]" — that
    # phrasing bleeds prescription into the bullet's tail hint even when
    # the body is correctly warning-shaped. Reduce to a noun phrase.
    ops = _strip_prescriptive_lead(ops_raw)

    if trigger:
        lead = f"When {_lower_first(trigger)}"
    else:
        lead = ""

    # The body is a warning. We strongly prefer failure mode alone.
    # Ops/seq are demoted: they only appear as a trailing parenthetical hint,
    # and only when there is genuinely nothing else to say.
    body_segments: List[str] = []
    if failure:
        body_segments.append(f"watch for {_lower_first(failure)}")
    # Verify clause is OPTIONAL — only include it when:
    #   (a) we have no failure mode to anchor the bullet, OR
    #   (b) the verify rule adds substantive guidance beyond what the
    #       failure already implies.
    # Heuristic for (b): the verify text mentions a check that's not
    # already implicit in the failure description (e.g. "compare against
    # original" / "read twice"). When failure already says "watch for
    # OCR misreads on small text", an empty "verify by inspecting the
    # baseline" adds noise. We approximate (b) by skipping verify when
    # both failure and verify exist AND verify is short or starts with
    # a generic verb that the agent would default to anyway.
    _GENERIC_VERIFY_LEAD = ("check ", "verify ", "compare ", "inspect ", "review ", "look ")
    if verify and (
        not failure                                # case (a): no failure → keep verify
        or (
            len(verify.split()) > 5                # case (b): substantive guidance
            and not verify.lower().startswith(_GENERIC_VERIFY_LEAD)
        )
    ):
        body_segments.append(f"verify by {_lower_first(verify)}")

    # Trailing hint about ops — parenthetical, low-prominence, only when
    # we have nothing in the body OR when ops adds genuinely new info.
    # Skip the hint entirely if it would be the bullet's main content,
    # because that resurrects the prescriptive recipe pattern.
    ops_hint: str = ""
    if ops and not body_segments:
        # No failure / no verify — the only signal we have is the positive
        # ops evidence. Keep it factual (something "has helped"), NOT
        # prescriptive ("consider X").
        ops_hint = f"on similar cases, {_lower_first(ops)} has helped"
    elif ops and body_segments:
        # We have a warning body. Mention ops as a brief tail hint.
        ops_hint = f"observed remedy on similar cases: {_lower_first(ops)}"
    elif seq and not body_segments:
        ops_hint = f"on similar cases, the sequence {seq} has helped"

    # Compose the sentence.
    if not body_segments and not ops_hint:
        return ""

    if body_segments:
        body = " — ".join(body_segments)
    else:
        body = ""

    if lead and body:
        sentence = f"{lead}, {body}"
    elif lead:
        # Lead-only is awkward; promote ops hint to body if present.
        sentence = f"{lead}; {ops_hint}" if ops_hint else f"{lead}, watch for the gated failure mode"
        ops_hint = ""  # consumed
    elif body:
        sentence = body[0].upper() + body[1:]
    else:
        # body empty, lead empty → ops_hint must exist
        sentence = ops_hint[0].upper() + ops_hint[1:]
        ops_hint = ""

    if ops_hint:
        sentence += f" ({ops_hint})"

    sentence = sentence.rstrip(". ") + "."
    return _truncate(sentence)


def _truncate(sentence: str, max_words: int = 22) -> str:
    """Keep rendered bullets compact. v9 audit showed v9 bullets averaged 27
    words vs v4 at 14-16 words; the size doubling translated to ~2x
    attention cost per skill. The v9 disaster on AF (-20pp pass@1) had
    bullet length as a meaningful contributor. We cap at 22 words here
    to give a small headroom over the 20-word target enforced in the
    fuse / generate / merge prompts.
    """
    words = sentence.split()
    if len(words) <= max_words:
        return sentence
    return " ".join(words[:max_words]).rstrip(",;:") + "..."


