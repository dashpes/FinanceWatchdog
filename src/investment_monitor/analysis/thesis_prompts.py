"""Prompt templates for the autonomous investor's thesis layer (Phase 3)."""

# Output schema (a single JSON object) shared by both prompts.
_THESIS_JSON_SPEC = """Respond with ONLY a single JSON object (no prose, no markdown fences):
{{"narrative": "<2-4 sentence core argument>",
  "conviction": <number 0.0-1.0>,
  "key_catalysts": ["<short>", ...],
  "risks": ["<short>", ...],
  "invalidation_conditions": {{"composite_drop": <points, optional>,
                              "price_drop_pct": <percent, optional>,
                              "keywords": ["fraud", "bankruptcy", ...]}}}}
conviction is your honest probability-weighted confidence the thesis plays out (0=no edge, 1=certain).
invalidation_conditions are the hard triggers that would END the thesis."""

THESIS_GENERATE_PROMPT = f"""You are a disciplined long-only equity analyst forming a fresh investment thesis \
for {{symbol}}. Be skeptical and concrete. You only produce a thesis; deterministic code sizes \
and risk-checks any resulting trade, so focus purely on the ARGUMENT and its INVALIDATION triggers.

EVIDENCE
Factor scores (0-100): {{score_block}}
Latest research recommendation: {{recommendation}}
Recent headlines:
{{news_block}}

{_THESIS_JSON_SPEC}

JSON object:"""

THESIS_UPDATE_PROMPT = f"""You maintain an existing investment thesis for {{symbol}} and must update it \
given fresh evidence. Revise the narrative and, crucially, your conviction (raise it only if evidence \
confirms; cut it if the case is weakening). You produce only the thesis JSON; deterministic code handles \
sizing and the guardrail gate.

CURRENT THESIS
Narrative: {{narrative}}
Current conviction: {{conviction}}

FRESH EVIDENCE
Factor scores now (0-100): {{score_block}}
Recent headlines:
{{news_block}}
Recent event signals: {{signals_block}}

{_THESIS_JSON_SPEC}

JSON object:"""


# Variant of THESIS_UPDATE_PROMPT used ONLY when the learning loop has a realized-
# outcome block to inject (Phase 6). The base template above is left byte-for-byte
# unchanged so a re-eval with no track record produces an identical prompt. The
# block is factual and non-leading (numbers only — no buy/sell verdict); the model
# still decides conviction.
THESIS_UPDATE_PROMPT_WITH_OUTCOME = f"""You maintain an existing investment thesis for {{symbol}} and must update it \
given fresh evidence. Revise the narrative and, crucially, your conviction (raise it only if evidence \
confirms; cut it if the case is weakening). Weigh how the thesis has ACTUALLY performed against your \
prior conviction — a confident thesis that keeps underperforming should lose conviction. You produce \
only the thesis JSON; deterministic code handles sizing and the guardrail gate.

CURRENT THESIS
Narrative: {{narrative}}
Current conviction: {{conviction}}

FRESH EVIDENCE
Factor scores now (0-100): {{score_block}}
Recent headlines:
{{news_block}}
Recent event signals: {{signals_block}}
Realized performance & track record: {{outcome_block}}

{_THESIS_JSON_SPEC}

JSON object:"""
