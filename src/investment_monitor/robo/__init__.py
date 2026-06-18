"""Robo advisor: locally-run autonomous portfolio rebalancing.

A cash-only, long-only robo advisor that uses a local LLM (Ollama) to *propose*
trades and a deterministic guardrail gate to *dispose* of them. The LLM can never
call the broker directly — every order passes through ``gate.validate`` and the
broker's preflight before being placed (and only when ``DRY_RUN`` is False).

Hard safety guarantees enforced in code (not left to the model):
  * cash account only (refuse to run if margin/borrowing is enabled)
  * no money movement (no deposit/withdraw/transfer ever)
  * long-only (no shorting; sells limited to held quantity)
  * no options writing, no margin, no leverage
  * spend never exceeds settled cash
  * order-type and symbol allowlists
  * per-run / per-day / per-order size caps
  * dry-run by default; full append-only audit log
"""

from __future__ import annotations

__all__ = [
    "AccountState",
    "GateDecision",
    "Position",
    "ProposedOrder",
]

from investment_monitor.robo.models import (
    AccountState,
    GateDecision,
    Position,
    ProposedOrder,
)
