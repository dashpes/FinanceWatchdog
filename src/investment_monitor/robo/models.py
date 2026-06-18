"""Internal domain models for the robo advisor.

These models are intentionally decoupled from the Public.com SDK's raw response
shapes. ``broker.py`` translates raw API payloads into these types, and the
guardrail gate / allocation logic operate only on these — so the safety-critical
code never depends on broker-specific JSON.

Monetary amounts and share quantities use ``Decimal`` to avoid floating-point
drift in affordability math.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# A pseudo-symbol used in target allocations to mean "leave this fraction in cash".
CASH_SYMBOL = "CASH"


class OrderSide(str, Enum):
    """Side of an order. Long-only: only buy and sell of held positions."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Allowed order types. Anything else is rejected by the gate."""

    MARKET = "market"
    LIMIT = "limit"


class Position(BaseModel):
    """A single long position in the account.

    The ``unit_cost`` / ``unrealized_*`` fields are the broker's own cost-basis math
    (Public computes them and returns them on each position). They are all optional:
    the broker may omit a basis, and dry-run/paper snapshots have none. Treat ``None``
    as "unknown", never as zero.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    quantity: Decimal = Field(..., ge=0)
    price: Decimal = Field(..., ge=0, description="Latest/last price per share")
    unit_cost: Decimal | None = Field(
        default=None, ge=0, description="Average cost per share (broker cost basis)"
    )
    unrealized_gain: Decimal | None = Field(
        default=None, description="Unrealized P&L in dollars, broker-reported"
    )
    unrealized_gain_pct: Decimal | None = Field(
        default=None, description="Unrealized P&L as a percentage, broker-reported"
    )

    @property
    def market_value(self) -> Decimal:
        """Current market value of the position."""
        return self.quantity * self.price

    @property
    def cost_basis_value(self) -> Decimal | None:
        """Total cost basis (unit_cost * quantity); None when no basis is known."""
        if self.unit_cost is None:
            return None
        return self.unit_cost * self.quantity

    @property
    def unrealized_return(self) -> Decimal | None:
        """Unrealized return as a fraction (price / unit_cost - 1).

        Derived from the broker's unit cost and the last price so the scale is
        unambiguous (unlike the broker's percentage field). None when no basis.
        """
        if self.unit_cost is None or self.unit_cost <= 0:
            return None
        return self.price / self.unit_cost - 1


class AccountState(BaseModel):
    """A normalized snapshot of the brokerage account.

    ``is_cash_account`` / ``has_margin`` are the structural safety signals. The
    app refuses to run unless ``is_cash_account`` is True and ``has_margin`` is
    False (see ``gate`` and ``rebalance``).
    """

    model_config = ConfigDict(frozen=True)

    account_id: str
    account_type: str = Field(default="", description="Raw account type label from broker")
    is_cash_account: bool
    has_margin: bool
    settled_cash: Decimal = Field(..., description="Cash available to spend right now")
    positions: list[Position] = Field(default_factory=list)
    # Symbols with an in-flight (open/pending) order at the broker. The gate rejects
    # new orders for these to avoid duplicating queued trades across runs.
    open_order_symbols: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)

    @property
    def positions_value(self) -> Decimal:
        """Sum of market value across all positions."""
        return sum((p.market_value for p in self.positions), Decimal("0"))

    @property
    def total_value(self) -> Decimal:
        """Total portfolio value = settled cash + positions market value."""
        return self.settled_cash + self.positions_value

    @property
    def total_cost_basis(self) -> Decimal | None:
        """Sum of position cost bases; None if no position reports a basis."""
        bases = [p.cost_basis_value for p in self.positions if p.cost_basis_value is not None]
        return sum(bases, Decimal("0")) if bases else None

    @property
    def total_unrealized_gain(self) -> Decimal | None:
        """Total unrealized P&L across positions that report a cost basis.

        Prefers the broker's own ``unrealized_gain`` per position; falls back to
        (market value - cost basis) for any position that has a basis but no
        broker-reported gain. None when no position has a basis at all.
        """
        contributions: list[Decimal] = []
        for p in self.positions:
            if p.unrealized_gain is not None:
                contributions.append(p.unrealized_gain)
            elif p.cost_basis_value is not None:
                contributions.append(p.market_value - p.cost_basis_value)
        return sum(contributions, Decimal("0")) if contributions else None

    def get_position(self, symbol: str) -> Position | None:
        """Return the position for ``symbol`` if held, else None."""
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None

    def held_quantity(self, symbol: str) -> Decimal:
        """Quantity currently held for ``symbol`` (0 if not held)."""
        pos = self.get_position(symbol)
        return pos.quantity if pos else Decimal("0")


class Trade(BaseModel):
    """A single executed trade from the broker's transaction history.

    ``gross`` is the trade value before fees (price * qty); cost basis / proceeds
    accounting layers fees on top in :mod:`investment_monitor.robo.pnl`.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    side: OrderSide
    quantity: Decimal = Field(..., gt=0)
    gross: Decimal = Field(..., ge=0, description="Trade value (price * qty), fees excluded")
    fees: Decimal = Field(default=Decimal("0"), ge=0)
    timestamp: datetime | None = None

    @property
    def price(self) -> Decimal:
        """Per-share execution price (gross / quantity)."""
        return self.gross / self.quantity if self.quantity else Decimal("0")


class ProposedOrder(BaseModel):
    """A candidate order proposed by the allocation logic or the LLM.

    Exactly one of ``quantity`` or ``notional`` must be set. ``extra_fields``
    captures any unrecognized keys the LLM emitted (e.g. an ``option`` or
    ``leverage`` field) so the gate can reject forbidden order shapes outright.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: Decimal | None = Field(default=None, gt=0)
    notional: Decimal | None = Field(default=None, gt=0)
    limit_price: Decimal | None = Field(default=None, gt=0)
    reason: str = Field(default="", max_length=500)
    source: str = Field(default="deterministic", description="'deterministic' or 'llm'")
    extra_fields: dict[str, Any] = Field(default_factory=dict, repr=False)

    @model_validator(mode="after")
    def _check_quantity_xor_notional(self) -> "ProposedOrder":
        if (self.quantity is None) == (self.notional is None):
            raise ValueError("Exactly one of quantity or notional must be set")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit_price is required for limit orders")
        return self

    def estimated_cost(self, price: Decimal) -> Decimal:
        """Estimated gross cost/proceeds (qty * price) for affordability checks.

        For notional orders the notional *is* the cost. For quantity orders we
        multiply by the supplied reference price (limit price preferred when set).
        """
        if self.notional is not None:
            return self.notional
        ref = self.limit_price if self.limit_price is not None else price
        return (self.quantity or Decimal("0")) * ref


@dataclass
class GateDecision:
    """The deterministic gate's verdict on a single proposed order.

    A plain dataclass (not a pydantic model) so it can hold an order without
    re-validating it — the gate must be able to *report* on malformed orders.
    """

    accepted: bool
    reason: str
    order: ProposedOrder
    code: str = ""

    @classmethod
    def accept(cls, order: ProposedOrder, reason: str = "ok") -> "GateDecision":
        return cls(accepted=True, reason=reason, code="accepted", order=order)

    @classmethod
    def reject(cls, order: ProposedOrder, code: str, reason: str) -> "GateDecision":
        return cls(accepted=False, reason=reason, code=code, order=order)


class RunCounters(BaseModel):
    """Order counts the gate needs to enforce rate/size caps."""

    orders_this_run: int = 0
    orders_today: int = 0
