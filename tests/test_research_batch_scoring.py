"""Tests for the batched (single-call) 5-factor scoring parser.

Pure parsing only — no Ollama. The batched path replaces 5 LLM calls/stock with 1,
and on any parse failure falls back to the per-factor path, so the parser must be
strict (missing factor -> None) but tolerant of fences/clamping.
"""

from __future__ import annotations

from investment_monitor.analysis.research_scorer import ResearchScorer


def _scorer():
    return ResearchScorer(model="phi3:mini")  # no network until .generate()


def test_parse_batch_all_factors():
    s = _scorer()
    resp = (
        '{"value":{"score":60,"reasoning":"cheap"},'
        '"growth":{"score":70,"reasoning":"g"},'
        '"quality":{"score":80,"reasoning":"q"},'
        '"momentum":{"score":40,"reasoning":"m"},'
        '"sentiment":{"score":55,"reasoning":"s"}}'
    )
    out = s._parse_batch_response(resp)
    assert out is not None
    v, g, q, m, se = out
    assert (v.score, g.score, q.score, m.score, se.score) == (60, 70, 80, 40, 55)
    assert v.reasoning == "cheap"


def test_parse_batch_clamps_and_handles_fences():
    s = _scorer()
    resp = (
        '```json\n{"value":{"score":150},"growth":{"score":-10},'
        '"quality":{"score":50},"momentum":{"score":50},"sentiment":{"score":50}}\n```'
    )
    out = s._parse_batch_response(resp)
    assert out is not None
    assert out[0].score == 100.0  # clamped high
    assert out[1].score == 0.0    # clamped low


def test_parse_batch_missing_factor_falls_back():
    s = _scorer()
    # Missing quality/momentum/sentiment -> None so the caller uses per-factor scoring.
    resp = '{"value":{"score":60},"growth":{"score":70}}'
    assert s._parse_batch_response(resp) is None


def test_parse_batch_garbage_is_none():
    s = _scorer()
    assert s._parse_batch_response("not json at all") is None
    assert s._parse_batch_response(None) is None
    assert s._parse_batch_response('{"value":{"score":"abc"},"growth":{"score":1},'
                                   '"quality":{"score":1},"momentum":{"score":1},'
                                   '"sentiment":{"score":1}}') is None  # non-numeric score
