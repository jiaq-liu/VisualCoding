"""Text-skill fallback of the method: when acquisition finds no instrument for a family, one advisory text skill is summarised from the
executor's own TRAIN rollout trajectories (the agent harness's skill-summarisation recipe; no validation split) and appended to the executor's system prompt at
deployment.  `build.py` = the summariser; `harness_prompts/` = verbatim copies of the agent harness prompt modules it uses."""
