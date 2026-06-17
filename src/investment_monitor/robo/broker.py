"""Thin wrapper over the Public.com Trading API SDK (``publicdotcom-py``).

Design notes / safety:
  * The SDK (import root ``public_api_sdk``) is pre-1.0 and its exact response
    field names are not fully pinned. All raw-payload extraction is funneled
    through small ``_first(...)`` helpers that try several candidate keys, and
    the raw payloads are stashed on the returned models so ``robo check-safety``
    can print them for you to confirm the mapping on first run.
  * ``useMargin`` on Public's place-order request **defaults to true**. We always
    send it ``False`` and additionally assert preflight ``marginRequirement == 0``,
    on top of refusing to run on a non-cash account. Three independent guards.
  * No funding / ACH / transfer methods are wrapped here — by construction the
    robo advisor cannot move money.
  * There is no Public.com sandbox. ``dry_run`` makes the wrapper stop at
    preflight and never call ``place_order``.

This module isolates every SDK-specific detail. The rest of the robo advisor
depends only on the domain models in ``robo.models``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from loguru import logger

from investment_monitor.robo.models import (
    AccountState,
    OrderSide,
    OrderType,
    Position,
    ProposedOrder,
)


class BrokerError(Exception):
    """A recoverable error talking to the broker."""


class SafetyViolation(BrokerError):
    """A structural safety guarantee was violated (e.g. margin detected)."""


@dataclass
class PreflightResult:
    """Outcome of a broker preflight calculation for one order."""

    ok: bool
    estimated_cost: Decimal | None = None
    estimated_proceeds: Decimal | None = None
    total_fees: Decimal | None = None
    margin_requirement: Decimal | None = None
    message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlacedOrder:
    """Result of placing (or simulating) an order."""

    order_id: str
    status: str
    simulated: bool
    raw: dict[str, Any] = field(default_factory=dict)


def _to_decimal(value: Any) -> Decimal | None:
    """Best-effort conversion of a raw value to Decimal."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _as_dict(obj: Any) -> dict[str, Any]:
    """Coerce an SDK response object (or dict) into a plain JSON-able dict.

    Prefers pydantic ``model_dump(mode="json")`` so nested models become dicts and
    enums/Decimals/datetimes become strings — which the ``_first``/``_to_decimal``
    helpers handle uniformly. Falls back to ``.dict()``, dataclasses, then ``vars``.
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            result = dump(mode="json")
            if isinstance(result, dict):
                return result
        except Exception:  # noqa: BLE001 - defensive against SDK churn
            pass
    for attr in ("dict", "to_dict", "_asdict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                result = fn()
                if isinstance(result, dict):
                    return result
            except Exception:  # noqa: BLE001
                pass
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {}


def _first(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present, non-None value among ``keys`` (case-insensitive)."""
    lowered = {k.lower(): v for k, v in d.items()}
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
        lk = key.lower()
        if lk in lowered and lowered[lk] is not None:
            return lowered[lk]
    return default


def account_state_from_raw(
    raw_account: dict[str, Any],
    raw_portfolio: dict[str, Any],
    prices: dict[str, Decimal] | None = None,
) -> AccountState:
    """Map raw accounts + portfolio payloads into an :class:`AccountState`.

    Pure function (no SDK/network) so the cash-vs-margin guard is unit-testable.
    ``prices`` optionally supplies last prices for positions whose payload lacks one.
    """
    prices = prices or {}
    account = _as_dict(raw_account)
    portfolio = _as_dict(raw_portfolio)

    account_id = str(
        _first(account, "accountId", "account_id", "accountNumber", "id", default="")
    )
    brokerage_type = str(
        _first(account, "brokerageAccountType", "brokerage_account_type", default="")
    ).upper()
    account_type = str(_first(account, "accountType", "account_type", default=""))

    is_cash = brokerage_type == "CASH"
    has_margin = brokerage_type == "MARGIN"
    # Fail safe: if the field is missing/unknown, treat as NOT a cash account so
    # the startup guard refuses to run rather than assuming safety.
    if brokerage_type not in {"CASH", "MARGIN"}:
        logger.warning(
            "Unrecognized brokerageAccountType {bt!r}; treating account as non-cash for safety",
            bt=brokerage_type,
        )
        is_cash = False

    # Cash spendable in a cash account lives under buying_power.cash_only_buying_power.
    buying_power = _as_dict(_first(portfolio, "buying_power", "buyingPower", default={}))
    settled_cash = (
        _to_decimal(_first(buying_power, "cash_only_buying_power", "cashOnlyBuyingPower"))
        or _to_decimal(_first(buying_power, "buying_power", "buyingPower"))
        # Fall back to any flat cash field if the SDK shape changes.
        or _to_decimal(_first(portfolio, "settled_cash", "settledCash", "cash"))
        or Decimal("0")
    )

    positions: list[Position] = []
    raw_positions = _first(portfolio, "positions", "holdings", default=[]) or []
    if isinstance(raw_positions, dict):  # sometimes keyed by symbol
        raw_positions = list(raw_positions.values())
    for rp in raw_positions:
        pd = _as_dict(rp)
        instrument = _as_dict(_first(pd, "instrument", default={}))
        symbol = str(
            _first(instrument, "symbol", default=_first(pd, "symbol", "ticker", default=""))
        ).upper()
        if not symbol:
            continue
        quantity = _to_decimal(_first(pd, "quantity", "shares", "qty")) or Decimal("0")
        # last_price is a nested {last_price, timestamp} object.
        last_price_obj = _as_dict(_first(pd, "last_price", "lastPrice", default={}))
        current_value = _to_decimal(_first(pd, "current_value", "currentValue", "marketValue", "value"))
        price = (
            _to_decimal(_first(last_price_obj, "last_price", "lastPrice", "value"))
            or prices.get(symbol)
        )
        if price is None or price <= 0:
            if current_value is not None and quantity > 0:
                price = current_value / quantity
            else:
                price = Decimal("0")
        positions.append(Position(symbol=symbol, quantity=quantity, price=price))

    return AccountState(
        account_id=account_id,
        account_type=account_type,
        is_cash_account=is_cash,
        has_margin=has_margin,
        settled_cash=settled_cash,
        positions=positions,
        raw={"account": account, "portfolio": portfolio},
    )


class PublicBroker:
    """Wrapper over the Public.com SDK exposing only the calls the robo needs."""

    def __init__(
        self,
        api_token: str,
        account_id: str = "",
        base_url: str = "",
        dry_run: bool = True,
    ) -> None:
        self._api_token = api_token
        self._account_id = account_id
        self._base_url = base_url
        self.dry_run = dry_run
        self._client: Any = None

    # -- client / connectivity ------------------------------------------------

    @property
    def client(self) -> Any:
        """Lazily build the Public SDK client."""
        if self._client is None:
            if not self._api_token:
                raise BrokerError(
                    "PUBLIC_API_TOKEN is not set. Add it to .env before using the robo advisor."
                )
            try:
                from public_api_sdk import (  # type: ignore
                    ApiKeyAuthConfig,
                    PublicApiClient,
                    PublicApiClientConfiguration,
                )
            except ImportError as exc:  # pragma: no cover - import guard
                raise BrokerError(
                    "publicdotcom-py is not installed. Install with: pip install '.[robo]'"
                ) from exc

            auth = ApiKeyAuthConfig(api_secret_key=self._api_token, validity_minutes=60)
            config_kwargs: dict[str, Any] = {}
            if self._account_id:
                config_kwargs["default_account_number"] = self._account_id
            if self._base_url:
                config_kwargs["base_url"] = self._base_url
            client_config = PublicApiClientConfiguration(**config_kwargs)
            self._client = PublicApiClient(auth, config=client_config)
        return self._client

    def _resolve_account_id(self, accounts: list[dict[str, Any]]) -> dict[str, Any]:
        """Pick the configured account, or the sole account if only one exists."""
        if self._account_id:
            for acc in accounts:
                acc_d = _as_dict(acc)
                acc_id = str(_first(acc_d, "accountId", "account_id", "accountNumber", "id", default=""))
                if acc_id == self._account_id:
                    return acc_d
            raise BrokerError(f"Configured account_id {self._account_id!r} not found in accounts")
        if len(accounts) == 1:
            return _as_dict(accounts[0])
        raise BrokerError(
            f"{len(accounts)} accounts found; set account_id in config/robo.yaml to pick one"
        )

    # -- reads ----------------------------------------------------------------

    def list_accounts(self) -> list[dict[str, Any]]:
        """List all accounts the token can see (id, type, cash-vs-margin)."""
        accounts_resp = _as_dict(self.client.get_accounts())
        accounts = _first(accounts_resp, "accounts", default=[]) or []
        out: list[dict[str, Any]] = []
        for acc in accounts:
            d = _as_dict(acc)
            bt = str(
                _first(d, "brokerage_account_type", "brokerageAccountType", default="")
            ).upper()
            out.append({
                "account_id": str(
                    _first(d, "account_id", "accountId", "accountNumber", "id", default="")
                ),
                "account_type": str(_first(d, "account_type", "accountType", default="")),
                "brokerage_account_type": bt,
                "is_cash": bt == "CASH",
            })
        return out

    def get_account_state(self) -> AccountState:
        """Fetch accounts + portfolio and return a normalized :class:`AccountState`."""
        accounts_resp = _as_dict(self.client.get_accounts())
        accounts = _first(accounts_resp, "accounts", default=[]) or []
        if not accounts:
            raise BrokerError("No accounts returned by Public API")
        account = self._resolve_account_id(accounts)
        account_id = str(
            _first(account, "accountId", "account_id", "accountNumber", "id", default="")
        )

        try:
            portfolio = _as_dict(self.client.get_portfolio(account_id))
        except TypeError:
            # SDK may read the account from its default configuration instead.
            portfolio = _as_dict(self.client.get_portfolio())

        # Backfill any missing position prices from quotes.
        prelim = account_state_from_raw(account, portfolio)
        missing = [p.symbol for p in prelim.positions if p.price <= 0]
        prices: dict[str, Decimal] = {}
        if missing:
            try:
                prices = self.get_quotes(missing)
            except BrokerError as exc:
                logger.warning("Could not backfill prices for {syms}: {e}", syms=missing, e=exc)
        return account_state_from_raw(account, portfolio, prices=prices)

    def get_quotes(self, symbols: list[str]) -> dict[str, Decimal]:
        """Return last price per symbol. Takes/uses the SDK's instrument objects."""
        if not symbols:
            return {}
        from public_api_sdk import InstrumentType, OrderInstrument  # type: ignore

        instruments = [
            OrderInstrument(symbol=s.upper(), type=InstrumentType.EQUITY) for s in symbols
        ]
        resp = self.client.get_quotes(instruments)
        # get_quotes returns a List[Quote]; each Quote nests its symbol in `instrument`.
        items = resp if isinstance(resp, list) else _first(_as_dict(resp), "quotes", default=[])
        out: dict[str, Decimal] = {}
        for q in items or []:
            qd = _as_dict(q)
            instrument = _as_dict(_first(qd, "instrument", default={}))
            sym = str(
                _first(instrument, "symbol", default=_first(qd, "symbol", "ticker", default=""))
            ).upper()
            last = _to_decimal(_first(qd, "last", "lastPrice", "price", "previous_close", "previousClose"))
            if sym and last is not None:
                out[sym] = last
        return out

    def get_quote(self, symbol: str) -> Decimal | None:
        """Return the last price for a single symbol, or None."""
        return self.get_quotes([symbol]).get(symbol.upper())

    # -- order request construction -------------------------------------------

    def _build_order_request(self, order: ProposedOrder) -> Any:
        """Construct the SDK ``OrderRequest`` for a proposed order.

        Always sets ``use_margin=False`` (Public defaults it to True).
        """
        from public_api_sdk import (  # type: ignore
            EquityMarketSession,
            InstrumentType,
            OrderExpirationRequest,
            OrderInstrument,
            OrderRequest,
            OrderSide as SdkOrderSide,
            OrderType as SdkOrderType,
            TimeInForce,
        )

        side = SdkOrderSide.BUY if order.side is OrderSide.BUY else SdkOrderSide.SELL
        otype = SdkOrderType.MARKET if order.order_type is OrderType.MARKET else SdkOrderType.LIMIT
        common: dict[str, Any] = {
            "instrument": OrderInstrument(symbol=order.symbol, type=InstrumentType.EQUITY),
            "order_side": side,
            "order_type": otype,
            "expiration": OrderExpirationRequest(time_in_force=TimeInForce.DAY),
            "equity_market_session": EquityMarketSession.CORE,
        }
        if order.quantity is not None:
            common["quantity"] = order.quantity
        else:
            common["amount"] = order.notional  # notional / dollar order
        if order.order_type is OrderType.LIMIT:
            common["limit_price"] = order.limit_price
        return OrderRequest(order_id=str(uuid.uuid4()), use_margin=False, **common)

    def _build_preflight_request(self, order: ProposedOrder) -> Any:
        """Construct the SDK ``PreflightRequest`` (no order_id / use_margin)."""
        from public_api_sdk import (  # type: ignore
            EquityMarketSession,
            InstrumentType,
            OrderExpirationRequest,
            OrderInstrument,
            OrderSide as SdkOrderSide,
            OrderType as SdkOrderType,
            PreflightRequest,
            TimeInForce,
        )

        side = SdkOrderSide.BUY if order.side is OrderSide.BUY else SdkOrderSide.SELL
        otype = SdkOrderType.MARKET if order.order_type is OrderType.MARKET else SdkOrderType.LIMIT
        kwargs: dict[str, Any] = {
            "instrument": OrderInstrument(symbol=order.symbol, type=InstrumentType.EQUITY),
            "order_side": side,
            "order_type": otype,
            "expiration": OrderExpirationRequest(time_in_force=TimeInForce.DAY),
            "equity_market_session": EquityMarketSession.CORE,
            "validate_order": True,
        }
        if order.quantity is not None:
            kwargs["quantity"] = order.quantity
        else:
            kwargs["amount"] = order.notional
        if order.order_type is OrderType.LIMIT:
            kwargs["limit_price"] = order.limit_price
        return PreflightRequest(**kwargs)

    # -- preflight & place ----------------------------------------------------

    def preflight(self, order: ProposedOrder) -> PreflightResult:
        """Run Public's preflight calculation for an order (no order is placed)."""
        try:
            request = self._build_preflight_request(order)
            raw = _as_dict(self.client.perform_preflight_calculation(request))
        except BrokerError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface SDK errors as broker errors
            return PreflightResult(ok=False, message=f"preflight error: {exc}")

        # margin_requirement is a nested object; any positive component is disqualifying.
        margin_obj = _as_dict(_first(raw, "margin_requirement", "marginRequirement", default={}))
        margin_vals = [v for v in (_to_decimal(x) for x in margin_obj.values()) if v is not None]
        margin_req = max(margin_vals, default=Decimal("0"))

        exec_fee = _to_decimal(_first(raw, "estimated_execution_fee", "estimatedExecutionFee")) or Decimal("0")
        commission = _to_decimal(_first(raw, "estimated_commission", "estimatedCommission")) or Decimal("0")
        reg = _as_dict(_first(raw, "regulatory_fees", "regulatoryFees", default={}))
        reg_total = sum((d for d in (_to_decimal(v) for v in reg.values()) if d is not None), Decimal("0"))
        total_fees = exec_fee + commission + reg_total

        result = PreflightResult(
            ok=True,
            estimated_cost=_to_decimal(_first(raw, "estimated_cost", "estimatedCost")),
            estimated_proceeds=_to_decimal(_first(raw, "estimated_proceeds", "estimatedProceeds")),
            total_fees=total_fees,
            margin_requirement=margin_req,
            raw=raw,
        )
        # Cash-only guard #3: a cash-account order must never require margin.
        if margin_req and margin_req > 0:
            result.ok = False
            result.message = f"preflight reports marginRequirement={margin_req} (must be 0)"
        return result

    def place_order(self, order: ProposedOrder) -> PlacedOrder:
        """Place a real order. Refuses when the broker is in dry-run mode."""
        if self.dry_run:
            raise SafetyViolation(
                "place_order called while broker dry_run=True; this should never happen"
            )
        request = self._build_order_request(order)
        result = self.client.place_order(request)
        # place_order returns a NewOrder handle (not a plain model); read its id directly.
        order_id = str(getattr(result, "order_id", "") or "")
        status = str(getattr(result, "status", "") or "NEW")
        return PlacedOrder(order_id=order_id, status=status, simulated=False, raw={"order_id": order_id})

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Fetch the status of a placed order."""
        return _as_dict(self.client.get_order(order_id))

    def cancel_order(self, order_id: str) -> bool:
        """Attempt to cancel an order. Returns True on success."""
        try:
            self.client.cancel_order(order_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cancel order {oid} failed: {e}", oid=order_id, e=exc)
            return False
