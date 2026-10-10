"""Program acquisition: a writer model (gemini-3.1-pro-preview by default) writes the instruments itself.

Design: the model makes discrete decisions (hypothesis, which primitives, one edit per round); the harness does
everything that needs judgement: the family profile, the agreement check of a hypothesis, the range-rule fit, the
acceptance gate, and the test run (which the model never sees).

Modules:
    loop.py       the acquisition loop: profile -> hypothesis JSON -> agreement check -> code -> sandboxed train run
                  -> feedback table -> one edit or accept; K candidates per round, train selects; freeze
    playbook.md   the model-facing "skill for writing instruments": contract, procedure, witness patterns,
                  robustness tricks, anti-patterns, round output format
    profile.py    automatic family profile (sizes, palette, components, grid period, text, question templates,
                  answer types) + thumbnails
    primitives.py `lib`: the fixed primitive library the model composes (masks, components, contours, minAreaRect,
                  projection profiles, detectors, orientation panels)
                  and one discrete model call (`lib.ask`)
    runner.py     sandboxed subprocess run of a candidate over the train tasks: import allow-list, static checks,
                  per-task timeout, model-read marking (a `value` derived from a lib.ask reply with no pixel-derived
                  quantity entering it = the executor, not an instrument; never counts as a new measurement)
    hypothesis.py stage A of a round: agreement of f(m) with the train answers + systematic-error diagnostics
                  (offset / scale / count bias / candidate index / unparsed templates) + edit-type classification
    fit_rule.py   range-rule fitter over the witnesses in `diag` (bounded rule space, leave-one-task-out estimate,
                  number of rules searched reported); drop-in `range_from_row` for fitted rules
    gate.py       acceptance tiers (certified / GAIN: train precision / coverage / LOO), candidate ranking, freeze
    adapters.py   reading value -> answer (nearest option, text match, permutation, candidate lists, option verdicts)
    deploy.py     wraps a frozen program as an `Instrument` for instrument_eval (test readings, report)
    prices.json   assumed per-token prices used only for the cost columns of the reports
"""
