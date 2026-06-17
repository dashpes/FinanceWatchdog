"""LLM proposal layer: ask a local Ollama model for candidate rebalance orders.

The model only *suggests* orders. Its output is parsed defensively into
:class:`ProposedOrder` objects (unrecognized keys are preserved in
``extra_fields`` so the gate can reject forbidden order shapes) and then every
order is re-checked by the deterministic gate. If Ollama is unavailable, the
config disables it, or the output can't be parsed, we fall back to the
deterministic allocator — the safety behavior is identical either way.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from loguru import logger

from investment_monitor.robo.allocation import compute_allocation, generate_candidate_orders
from investment_monitor.robo.config import RoboConfig
from investment_monitor.robo.models import (
    AccountState,
    OrderSide,
    OrderType,
    ProposedOrder,
)
from investment_monitor.robo.prompts import PROPOSAL_PROMPT

if TYPE_CHECKING:
    from investment_monitor.analysis.local_llm import LocalLLM

# Recognized keys and their aliases. Anything else lands in extra_fields.
_QUANTITY_KEYS = ("quantity", "qty", "shares")
_NOTIONAL_KEYS = ("notional", "amount", "dollars", "value")
_KNOWN_KEYS = {
    "symbol", "ticker", "side", "order_type", "ordertype", "type",
    "limit_price", "limitprice", "price", "reason", "rationale",
    *_QUANTITY_KEYS, *_NOTIONAL_KEYS,
}


def _to_decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value).replace("$", "").replace(",", "").strip())
    except (InvalidOperation, ValueError, TypeError):
        return None


def _extract_json_array(text: str) -> list[Any] | None:
    """Pull the first JSON array out of a possibly-noisy LLM response."""
    if not text:
        return None
    cleaned = text.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):  # single order object
            return [parsed]
    except json.JSONDecodeError:
        pass
    # Fall back to the first bracketed array in the text.
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def _first_key(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    lowered = {k.lower(): v for k, v in item.items()}
    for k in keys:
        if k in lowered and lowered[k] is not None:
            return lowered[k]
    return None


def _build_order(item: dict[str, Any]) -> ProposedOrder | None:
    """Convert one parsed dict into a ProposedOrder, or None if unusable.

    Unknown keys are preserved in ``extra_fields`` so the gate can reject forbidden
    order shapes (options/margin/etc.). Malformed entries are skipped.
    """
    if not isinstance(item, dict):
        return None

    symbol = _first_key(item, ("symbol", "ticker"))
    side_raw = _first_key(item, ("side",))
    if not symbol or not side_raw:
        return None
    symbol = str(symbol).upper().strip()
    side_raw = str(side_raw).lower().strip()
    if side_raw not in ("buy", "sell"):
        return None
    side = OrderSide.BUY if side_raw == "buy" else OrderSide.SELL

    type_raw = str(_first_key(item, ("order_type", "ordertype", "type")) or "market").lower().strip()
    order_type = OrderType.LIMIT if type_raw == "limit" else OrderType.MARKET

    quantity = _to_decimal(_first_key(item, _QUANTITY_KEYS))
    notional = _to_decimal(_first_key(item, _NOTIONAL_KEYS))
    # Exactly one of quantity/notional is required by ProposedOrder.
    if (quantity is None) == (notional is None):
        return None

    limit_price = _to_decimal(_first_key(item, ("limit_price", "limitprice", "price")))
    reason = str(_first_key(item, ("reason", "rationale")) or "")[:500]

    extra = {k: v for k, v in item.items() if k.lower() not in _KNOWN_KEYS}

    try:
        return ProposedOrder(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity if quantity and quantity > 0 else None,
            notional=notional if notional and notional > 0 else None,
            limit_price=limit_price,
            reason=reason,
            source="llm",
            extra_fields=extra,
        )
    except Exception as exc:  # noqa: BLE001 - malformed proposal, skip it
        logger.debug("Skipping unparsable LLM order {item}: {e}", item=item, e=exc)
        return None


def parse_orders(text: str) -> list[ProposedOrder]:
    """Parse an LLM response into a list of ProposedOrder (pure; well-tested)."""
    raw = _extract_json_array(text)
    if raw is None:
        return []
    orders: list[ProposedOrder] = []
    for item in raw:
        order = _build_order(item)
        if order is not None:
            orders.append(order)
    return orders


def _positions_block(account_state: AccountState, config: RoboConfig) -> str:
    rows = compute_allocation(account_state, config)
    lines = []
    for r in rows:
        pos = account_state.get_position(r.symbol)
        price = pos.price if pos else Decimal("0")
        lines.append(
            f"  {r.symbol}: {r.current_weight:.1%} -> {r.target_weight:.1%}, "
            f"value=${r.current_value}, last_price=${price}"
        )
    return "\n".join(lines)


class RoboProposer:
    """Produces candidate orders, preferring the LLM but always able to fall back."""

    def __init__(self, local_llm: "LocalLLM | None", config: RoboConfig) -> None:
        self._llm = local_llm
        self._config = config

    def propose(self, account_state: AccountState) -> tuple[list[ProposedOrder], str]:
        """Return (orders, source) where source is 'llm' or 'deterministic'."""
        deterministic = generate_candidate_orders(account_state, self._config)

        if not self._config.use_llm or self._llm is None or not self._llm.is_available():
            return deterministic, "deterministic"

        try:
            prompt = PROPOSAL_PROMPT.format(
                settled_cash=account_state.settled_cash,
                total_value=account_state.total_value,
                allowlist=", ".join(self._config.allowlist),
                max_order_pct=f"{self._config.caps.max_order_pct:.0%}",
                rebalance_threshold=f"{self._config.rebalance_threshold:.0%}",
                positions_block=_positions_block(account_state, self._config),
            )
            response = self._llm.client.generate(
                model=self._llm.model,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 512},
            )
            text = (response.get("response") or "").strip()
        except Exception as exc:  # noqa: BLE001 - any LLM failure -> deterministic fallback
            logger.warning("LLM proposal failed ({e}); using deterministic rebalance", e=exc)
            return deterministic, "deterministic"

        orders = parse_orders(text)
        if not orders:
            logger.info("LLM returned no usable orders; using deterministic rebalance")
            return deterministic, "deterministic"
        return orders, "llm"
