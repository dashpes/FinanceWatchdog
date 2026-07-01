"""The rebalance orchestration — ``rebalance_run()``.

Pipeline (see the build brief, section 3):
  1. startup safety preflight — refuse to run unless it's a cash account with no margin
  2. fetch account + positions (normalized)
  3. fetch market data for the watchlist
  4. ask the local LLM for a proposal (or compute deterministically)
  5. proposal -> candidate orders
  6. GUARDRAIL GATE validates each order (deterministic code the LLM can't bypass)
  7. preflight each surviving order via the broker
  8. place orders, or simulate them when in dry-run
  9. write the audit log + persist the run

Live trading requires ALL of: ``ROBO_FORCE_DRY_RUN`` false (env kill-switch),
``dry_run: false`` in robo.yaml, and no per-run ``--dry-run`` override. Any one of
them keeps the run in simulation.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from decimal import Decimal

from loguru import logger
from sqlalchemy import func, select

from investment_monitor.config import Settings
from investment_monitor.robo.audit import AuditLogger
from investment_monitor.robo.blocklist import add_learned, is_unbuyable_message, load_learned
from investment_monitor.robo.broker import BrokerError, PublicBroker, fill_from_order_raw
from investment_monitor.robo.config import RoboConfig
from investment_monitor.robo.gate import validate_orders
from investment_monitor.robo.llm import RoboProposer
from investment_monitor.robo.market_hours import is_market_open
from investment_monitor.robo.models import AccountState, GateDecision, OrderSide
from investment_monitor.robo.sizing import is_averaging_up_without_support
from investment_monitor.robo.signals import fetch_signals
from investment_monitor.storage import (
    RoboOrder,
    RoboRun,
    count_placed_orders_today,
    finalize_robo_run,
    get_active_symbols,
    get_active_theses,
    get_thesis,
    get_unfilled_placed_orders,
    get_session,
    init_db,
    save_robo_order,
    save_robo_run,
)


def _order_rationale(thesis) -> str:
    """A concise human 'why' for an order, snapshotted from its thesis.

    Persisted on the order (rebalance's `reason` is the mechanical drift math; this is
    the investment justification) so it survives the thesis later mutating/invalidating,
    and is surfaced in trade emails. Empty when there is no owning thesis (rebalance mode
    or a manual position being trimmed).
    """
    if thesis is None:
        return ""
    narrative = " ".join((thesis.narrative or "").split())
    if len(narrative) > 200:
        narrative = narrative[:199].rstrip() + "…"
    conviction = thesis.conviction if thesis.conviction is not None else 0.0
    return f"{conviction:.0%} conviction — {narrative}" if narrative else f"{conviction:.0%} conviction"


def _entry_conviction(thesis) -> float | None:
    """Conviction recorded when the thesis was first established (history[0]), if any."""
    if thesis is None:
        return None
    for h in thesis.conviction_history or []:
        if isinstance(h, dict) and h.get("conviction") is not None:
            return float(h["conviction"])
    return None


def _drop_unsupported_adds(orders, account, prices, thesis_for, config):
    """Drop BUYs that would average UP a held position without a strengthened thesis.

    Opening a NEW position and averaging DOWN (buying at/below cost) always pass; an ADD
    above cost is kept only when the owning thesis has strengthened
    (sizing.is_averaging_up_without_support). Sells are untouched. Fail-open: any lookup
    error keeps the order rather than silently dropping a trade.
    """
    cash_etf = (config.cash_etf or "").upper()
    kept = []
    for o in orders:
        try:
            if o.side is OrderSide.BUY and o.symbol.upper() != cash_etf:
                pos = account.get_position(o.symbol)
                thesis = thesis_for(o.symbol) if pos is not None else None
                # Only gate an ADD to a held name that HAS a thesis. No thesis (rebalance
                # mode / manual holding) fails OPEN — the add-gate is a thesis-aware guard,
                # not a general no-average-up rule, so it must never block ordinary
                # rebalancing back to a fixed target.
                if pos is not None and pos.quantity and pos.quantity > 0 and thesis is not None:
                    ref_price = prices.get(o.symbol) or pos.price
                    cur = float(thesis.conviction) if thesis.conviction is not None else 0.0
                    if is_averaging_up_without_support(
                        avg_cost=float(pos.unit_cost) if pos.unit_cost is not None else None,
                        ref_price=float(ref_price) if ref_price is not None else None,
                        current_conviction=cur,
                        entry_conviction=_entry_conviction(thesis),
                        cfg=config.sizing,
                    ):
                        logger.info(
                            "add-gate: skipping add to {s} — would average up without a "
                            "strengthened thesis", s=o.symbol,
                        )
                        continue
        except Exception as exc:  # noqa: BLE001 - the add-gate must never break a run
            logger.warning("add-gate check failed for {s} (keeping order): {e}", s=o.symbol, e=exc)
        kept.append(o)
    return kept


@dataclass
class RebalanceResult:
    """Summary of one rebalance run, returned to the CLI."""

    run_id: str
    dry_run: bool
    source: str = "deterministic"
    status: str = "completed"
    account_id: str = ""
    total_value: Decimal = Decimal("0")
    settled_cash: Decimal = Decimal("0")
    num_proposed: int = 0
    num_accepted: int = 0
    num_rejected: int = 0
    num_placed: int = 0
    decisions: list[GateDecision] = field(default_factory=list)
    message: str = ""

    def summary_line(self) -> str:
        mode = "DRY-RUN" if self.dry_run else "LIVE"
        return (
            f"[{mode}] {self.status}: proposed={self.num_proposed} "
            f"accepted={self.num_accepted} rejected={self.num_rejected} "
            f"placed={self.num_placed} (source={self.source})"
        )


def _resolve_dry_run(
    config: RoboConfig, settings: Settings, override: bool | None
) -> bool:
    """Live only when the env kill-switch is off AND config/override permit it."""
    if settings.robo_force_dry_run:
        return True
    if override is not None:
        return override
    return config.dry_run


def _build_local_llm(config: RoboConfig, settings: Settings):
    """Construct the shared Ollama client, or None if the LLM is disabled."""
    if not config.use_llm:
        return None
    try:
        from investment_monitor.analysis.local_llm import LocalLLM
    except ImportError:
        logger.warning("LocalLLM unavailable; using deterministic rebalance")
        return None
    model = config.ollama_model or settings.ollama_model
    return LocalLLM(model=model, base_url=settings.ollama_host)


def _thesis_entry_fill_cost(session, thesis) -> float | None:
    """Fill price of the order that actually OPENED *this* thesis's position, or None.

    Returns the fill price of the earliest reconciled BUY for the thesis's symbol that
    was placed at/after the thesis was created — i.e. the robo's own entry order for
    this idea. Deliberately ignores:
      * the broker's *blended* unit cost (which folds in shares bought before/independent
        of the thesis, or later adds at different prices), and
      * any BUY recorded before the thesis existed (a pre-existing holding).
    Returns None when no entry-corresponding fill exists, so the caller keeps the
    idea-time ``entry_price`` as the basis rather than measuring return from the wrong
    price. Fail-open: any error yields None.
    """
    try:
        stmt = (
            select(RoboOrder)
            .where(RoboOrder.symbol == thesis.symbol)
            .where(RoboOrder.side == OrderSide.BUY.value)
            .where(RoboOrder.fill_price.is_not(None))
            .order_by(RoboOrder.created_at.asc())
        )
        thesis_created = getattr(thesis, "created_at", None)
        for order_row in session.scalars(stmt):
            # Only the robo's own opening order counts: skip BUYs that predate the
            # thesis (a position held before/independently of this idea).
            if thesis_created is not None and order_row.created_at is not None:
                if order_row.created_at < thesis_created:
                    continue
            if order_row.fill_price and order_row.fill_price > 0:
                return float(order_row.fill_price)
    except Exception as exc:  # noqa: BLE001 - matching is best-effort context only
        logger.warning("entry-fill match failed for {s}: {e}", s=thesis.symbol, e=exc)
    return None


def _reconcile_fill_costs(session, account: AccountState) -> None:
    """Persist the thesis's own entry fill cost onto each active thesis.

    Writes ``entry_conditions['fill_cost']`` = the fill price of the order that actually
    OPENED this thesis's position, so the feedback loop scores realized return against
    the *actual entry fill* — NOT the broker's blended unit cost. The blended average
    folds in shares held before/independent of the thesis (or later adds at a different
    price), so for such names it is not the thesis entry and would bias calibration.
    When there is no entry-corresponding fill we leave ``entry_price`` as the basis
    rather than overwrite it with the wrong price.

    Fully fail-open: a reconcile error must never abort a rebalance run, and it is a
    no-op in paper/dry-run (no live entry fills exist), keeping that path byte-identical.
    """
    try:
        for thesis in get_active_theses(session, account.account_id or None):
            # Only for names we genuinely hold (a fill could be a since-closed position).
            if account.get_position(thesis.symbol) is None:
                continue
            new_cost = _thesis_entry_fill_cost(session, thesis)
            if new_cost is None:
                continue
            entry = dict(thesis.entry_conditions or {})
            if entry.get("fill_cost") != new_cost:
                entry["fill_cost"] = new_cost
                thesis.entry_conditions = entry  # reassign so the JSON column is dirtied
    except Exception as exc:  # noqa: BLE001 - feedback bookkeeping must never break a run
        logger.warning("fill-cost reconcile failed: {e}", e=exc)


def _apply_fill(order_row, info: dict) -> bool:
    """Apply a parsed ``get_order`` fill onto a RoboOrder row. Returns True ONLY if the
    order reached a genuinely terminal STATUS (so it no longer needs polling).

    Terminality is driven by ``info['terminal']`` (the broker order status), NOT by the
    mere presence of an ``average_price``. A PARTIALLY_FILLED order carries an
    ``average_price`` / ``filled_quantity`` for the shares filled so far but is still
    working — we record that partial progress yet keep ``fill_status`` NULL (the "still
    polling" sentinel ``get_unfilled_placed_orders`` keys on) so the remainder is
    reconciled on a later run. Latching it terminal here would strand the unfilled
    shares forever.
    """
    terminal = bool(info.get("terminal"))
    if info.get("average_price") is not None:
        # Record progress whether partial or fully filled (shares filled so far).
        order_row.fill_price = float(info["average_price"])
        order_row.fill_quantity = float(info.get("filled_quantity") or 0)
        if terminal:
            order_row.fill_status = info.get("status") or "FILLED"
            return True
        # Still working (e.g. PARTIALLY_FILLED): leave fill_status NULL, poll again.
        return False
    if terminal:  # rejected / cancelled / expired — done, no fill
        order_row.fill_quantity = 0.0
        order_row.fill_status = info.get("status") or "TERMINAL"
        return True
    return False  # still working — leave fill_status NULL, retry next run


def _reconcile_order_fills(session, broker: PublicBroker) -> None:
    """Poll the broker for fills of previously-placed orders not yet reconciled.

    Read-only and fail-open: a poll error on one order must never abort the run.

    Skipped entirely in dry-run: a paper run must be fully read-isolated from the live
    broker. Unfilled rows are real LIVE placements left by an earlier live run; calling
    ``broker.get_order`` on them here would make an authenticated read against the real
    broker from a paper run. We leave them untouched so the next LIVE run reconciles
    them (fill_status stays NULL, so they remain in the unfilled-poll set).
    """
    if getattr(broker, "dry_run", False):
        return
    try:
        pending = get_unfilled_placed_orders(session)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not load unfilled orders: {e}", e=exc)
        return
    for order_row in pending:
        try:
            info = fill_from_order_raw(broker.get_order(order_row.broker_order_id))
            if _apply_fill(order_row, info):
                logger.info(
                    "order {oid} reconciled: {st} {q}@{p}", oid=order_row.broker_order_id,
                    st=order_row.fill_status, q=order_row.fill_quantity, p=order_row.fill_price,
                )
        except Exception as exc:  # noqa: BLE001 - one bad poll never breaks the run
            logger.warning("fill poll failed for {oid}: {e}", oid=order_row.broker_order_id, e=exc)


def rebalance_run(
    config: RoboConfig,
    settings: Settings,
    *,
    dry_run_override: bool | None = None,
    broker: PublicBroker | None = None,
) -> RebalanceResult:
    """Execute one full rebalance run. ``broker`` may be injected for testing."""
    init_db(settings.db_path)
    run_id = str(uuid.uuid4())
    dry_run = _resolve_dry_run(config, settings, dry_run_override)
    audit = AuditLogger(settings.log_dir, run_id)

    # Make the kill-switch state explicit on every LIVE run. A real OS/launchd env
    # var silently overrides the .env file, so surface where the value came from —
    # an operator who believes .env keeps them in paper would otherwise have no signal.
    if not dry_run:
        env_src = (
            "OS/launchd environment (OVERRIDES .env)"
            if "ROBO_FORCE_DRY_RUN" in os.environ
            else ".env/default"
        )
        logger.warning(
            "LIVE MODE — real orders ENABLED. ROBO_FORCE_DRY_RUN={v} (from {src}); "
            "config.dry_run={d}. Revert with ROBO_FORCE_DRY_RUN=true or dry_run: true.",
            v=settings.robo_force_dry_run, src=env_src, d=config.dry_run,
        )

    if broker is None:
        broker = PublicBroker(
            api_token=settings.public_api_token,
            account_id=config.account_id,
            base_url=settings.public_api_base_url,
            dry_run=dry_run,
        )
    else:
        broker.dry_run = dry_run

    # --- 1 & 2. Safety preflight + account snapshot ------------------------------
    try:
        account: AccountState = broker.get_account_state()
    except (BrokerError, Exception) as exc:  # noqa: BLE001
        logger.error("Failed to fetch account state: {e}", e=exc)
        audit.safety_check(passed=False, detail=f"account fetch failed: {exc}")
        _persist_refused(settings, run_id, dry_run, account_id="", reason=str(exc), status="failed")
        return RebalanceResult(run_id=run_id, dry_run=dry_run, status="failed", message=str(exc))

    if not account.is_cash_account or account.has_margin:
        detail = (
            f"account {account.account_id} is not cash-only "
            f"(is_cash={account.is_cash_account}, has_margin={account.has_margin})"
        )
        logger.error("REFUSING TO RUN: {d}", d=detail)
        audit.safety_check(passed=False, detail=detail)
        _persist_refused(settings, run_id, dry_run, account_id=account.account_id,
                         reason=detail, status="refused")
        return RebalanceResult(
            run_id=run_id, dry_run=dry_run, status="refused",
            account_id=account.account_id, message=detail,
        )

    audit.safety_check(passed=True, detail=f"cash account {account.account_id} confirmed")
    audit.run_start(
        dry_run=dry_run, account_id=account.account_id,
        total_value=account.total_value, settled_cash=account.settled_cash, source="pending",
    )

    # --- 3. Market data ----------------------------------------------------------
    symbols = sorted(set(config.allowlist) | {p.symbol for p in account.positions})
    try:
        prices = broker.get_quotes(symbols)
    except BrokerError as exc:
        logger.warning("Quote fetch failed ({e}); proceeding with position prices only", e=exc)
        prices = {}
    for p in account.positions:
        prices.setdefault(p.symbol, p.price)

    # --- 4 & 5. Proposal (optionally informed by event signals) ------------------
    proposer = RoboProposer(_build_local_llm(config, settings), config)

    # Open one session for the whole run (SQLite, single-user).
    with get_session() as session:
        # Reconcile fills of prior live placements (read-only; no-op in paper), then
        # record the broker's real cost basis onto held theses so the feedback loop
        # measures realized return from the actual fill, not the quote at idea time.
        _reconcile_order_fills(session, broker)
        _reconcile_fill_costs(session, account)

        # Event-driven signals are advisory only — never seen by the gate. A
        # failure here degrades gracefully to the baseline drift rebalance.
        snapshot = None
        if config.signals.enabled:
            try:
                snapshot = fetch_signals(session, config, account)
                if not snapshot.is_empty:
                    audit.signals(snapshot)
            except Exception as exc:  # noqa: BLE001 - signals must never break a run
                logger.warning("Signal fetch failed ({e}); proceeding without signals", e=exc)
                snapshot = None

        orders, source = proposer.propose(
            account, signals=snapshot, session=session, account_id=account.account_id or None
        )

        # Per-symbol thesis cache (most-recent non-EXITED thesis, any live/invalidated
        # status) — used to justify adds and to snapshot each order's rationale below.
        thesis_cache: dict[str, object] = {}

        def _thesis_for(symbol: str):
            if symbol not in thesis_cache:
                thesis_cache[symbol] = get_thesis(session, symbol, account.account_id or None)
            return thesis_cache[symbol]

        # Don't chase: drop BUYs that would average UP a held position without a
        # strengthened thesis (raising cost basis for no new reason).
        orders = _drop_unsupported_adds(orders, account, prices, _thesis_for, config)
        for order in orders:
            audit.proposal(order)

        # In autonomous mode the tradeable universe is the set of names with a live
        # thesis, anything currently held (so positions can always be exited), plus the
        # cash ETF (a valid buy target for parking idle cash). The gate enforces it via
        # its allowlist check, and the `no_active_thesis` guard restricts BUYs to this set.
        gate_config = config
        active_symbols: set[str] | None = None
        if config.mode == "autonomous":
            active_symbols = get_active_symbols(session, account.account_id or None)
            if config.cash_etf:
                active_symbols = active_symbols | {config.cash_etf.upper()}
            held = {p.symbol for p in account.positions}
            gate_config = config.model_copy(update={"allowlist": sorted(active_symbols | held)})

        # Drawdown circuit-breaker: halt new buys when the portfolio is down beyond
        # the configured limit vs its prior peak (sells still allowed). Off by default.
        halt_buys = _drawdown_halt(session, config, account.total_value, account.account_id)
        if halt_buys:
            logger.warning("Drawdown circuit-breaker active: halting new buys this run")

        # Live placement is gated to market hours (research/maintenance ran already).
        market_open = is_market_open()
        if not dry_run and config.require_market_hours and not market_open:
            logger.warning("Market closed: live order placement deferred this run")

        unrealized = account.total_unrealized_gain
        run_row = RoboRun(
            run_id=run_id, dry_run=dry_run, account_id=account.account_id, source=source,
            total_value=float(account.total_value), settled_cash=float(account.settled_cash),
            unrealized_pnl=float(unrealized) if unrealized is not None else None,
            num_proposed=len(orders), status="running",
        )
        save_robo_run(session, run_row)

        # --- 6. Guardrail gate ---------------------------------------------------
        # Blocklist = operator's static list ∪ the learned set of broker-refused,
        # un-buyable names. Blocklisted BUYs are rejected before the position-cap
        # check, so a perpetually-un-buyable pick can never strand the open slot.
        blocklist = {s.upper() for s in config.blocklist} | load_learned(settings.db_path)
        orders_today = count_placed_orders_today(session)
        decisions = validate_orders(
            orders, account, gate_config, prices, orders_today=orders_today,
            active_symbols=active_symbols, blocklist=blocklist, halt_buys=halt_buys,
        )

        num_accepted = num_rejected = num_placed = 0
        final_status = "completed"
        try:
            for decision in decisions:
                audit.gate_decision(decision)
                order = decision.order
                # Snapshot the owning thesis's "why" onto the order (survives later thesis
                # mutation; surfaced in trade emails). None for rebalance-mode/manual names.
                thesis = _thesis_for(order.symbol)
                order_row = RoboOrder(
                    run_id=run_id, symbol=order.symbol,
                    side=order.side.value, order_type=order.order_type.value,
                    quantity=float(order.quantity) if order.quantity is not None else None,
                    notional=float(order.notional) if order.notional is not None else None,
                    limit_price=float(order.limit_price) if order.limit_price is not None else None,
                    source=order.source, reason=order.reason,
                    gate_accepted=decision.accepted, gate_code=decision.code,
                    gate_reason=decision.reason,
                    thesis_id=thesis.id if thesis is not None else None,
                    rationale=_order_rationale(thesis),
                )
                if not decision.accepted:
                    num_rejected += 1
                    order_row.status = "rejected"
                    save_robo_order(session, order_row)
                    continue
                num_accepted += 1

                # --- 7. Preflight ------------------------------------------------
                preflight = broker.preflight(order)
                audit.preflight(order, preflight)
                order_row.preflight_ok = preflight.ok
                order_row.preflight_reason = preflight.message
                if not preflight.ok:
                    order_row.status = "preflight_failed"
                    audit.order_result(order, simulated=False, placed=False,
                                       detail=f"preflight failed: {preflight.message}")
                    save_robo_order(session, order_row)
                    # Learn un-buyable names (e.g. "only available when closing a
                    # position") so they stop winning the open slot every run.
                    if order.side is OrderSide.BUY and is_unbuyable_message(preflight.message):
                        add_learned(settings.db_path, order.symbol, preflight.message or "")
                    continue

                # --- 8. Place or simulate ---------------------------------------
                if dry_run:
                    order_row.simulated = True
                    order_row.status = "simulated"
                    num_placed += 1
                    audit.order_result(order, simulated=True, placed=False, status="simulated")
                    save_robo_order(session, order_row)
                elif config.require_market_hours and not market_open:
                    # Live, but the market is closed: defer placement (don't queue).
                    order_row.status = "deferred_market_closed"
                    audit.order_result(order, simulated=False, placed=False,
                                       detail="market closed; placement deferred")
                    save_robo_order(session, order_row)
                else:
                    # Real placement. Catch ANY broker/SDK/network error (the SDK's
                    # own exceptions are NOT BrokerError subclasses) so one bad order
                    # can never abort the run, and COMMIT each confirmed placement the
                    # instant the broker returns — a later failure must never roll back
                    # a live order (which would hide it and risk a double-buy next run).
                    try:
                        placed = broker.place_order(order)
                    except Exception as exc:  # noqa: BLE001 - unmapped SDK/network errors
                        order_row.status = "place_failed"
                        logger.error("Order placement failed for {s}: {e}", s=order.symbol, e=exc)
                        audit.order_result(order, simulated=False, placed=False, detail=str(exc))
                        save_robo_order(session, order_row)
                        if order.side is OrderSide.BUY and is_unbuyable_message(str(exc)):
                            add_learned(settings.db_path, order.symbol, str(exc))
                    else:
                        order_row.placed = True
                        order_row.broker_order_id = placed.order_id
                        order_row.status = placed.status or "placed"
                        num_placed += 1
                        audit.order_result(order, simulated=False, placed=True,
                                           broker_order_id=placed.order_id, status=placed.status)
                        # NB: we deliberately do NOT poll get_order here. Public placement
                        # is asynchronous — right after placing, the order is either not yet
                        # indexed (get_order 404s; eventual consistency, per the SDK) or not
                        # yet filled (status NEW). Either way there is no fill to capture. The
                        # fill price/qty is reconciled by _reconcile_order_fills at the start
                        # of the next run (fill_status stays NULL until then), and unrealized
                        # P&L / the learning fill_cost come from the position cost basis, not
                        # from this order — so neither depends on an immediate poll.
                        save_robo_order(session, order_row)
                        session.commit()  # durable before the next order is attempted
        except Exception as exc:  # noqa: BLE001 - record + return, never crash the loop
            final_status = "errored"
            logger.error("Placement loop aborted ({e}); finalizing run as errored", e=exc)
        finally:
            # --- 9. Finalize (always reflects what actually happened) ------------
            finalize_robo_run(
                session, run_id,
                finished_at=_utcnow(), status=final_status,
                num_accepted=num_accepted, num_rejected=num_rejected, num_placed=num_placed,
            )

    audit.run_summary(
        num_proposed=len(orders), num_accepted=num_accepted, num_rejected=num_rejected,
        num_placed=num_placed, dry_run=dry_run, status=final_status,
    )

    result = RebalanceResult(
        run_id=run_id, dry_run=dry_run, source=source, status=final_status,
        account_id=account.account_id, total_value=account.total_value,
        settled_cash=account.settled_cash, num_proposed=len(orders),
        num_accepted=num_accepted, num_rejected=num_rejected, num_placed=num_placed,
        decisions=decisions,
    )
    logger.info("Rebalance complete: {s}", s=result.summary_line())
    return result


def _drawdown_halt(
    session, config: RoboConfig, current_total: Decimal, account_id: str = ""
) -> bool:
    """True if the portfolio is down beyond ``max_drawdown_pct`` vs its prior peak.

    Peak total value is read from recorded run history (a self-contained high-water
    mark), scoped to this account so one account can't pin another's drawdown.
    Disabled (returns False) when ``max_drawdown_pct`` is 0. Fail-open by design
    (never blocks sells; buys proceed on a read error rather than stall).
    """
    if config.caps.max_drawdown_pct <= 0 or current_total <= 0:
        return False
    try:
        # True ALL-TIME high-water mark via a DB aggregate (MAX over every run for this
        # account), NOT a capped recent window. Under a 24/7 every-few-minutes schedule
        # the run table exceeds any fixed row cap within weeks, so a windowed peak would
        # silently age the real peak out, under-state drawdown, and let buys through
        # during a genuine decline. The aggregate never truncates.
        stmt = select(func.max(RoboRun.total_value))
        if account_id:
            stmt = stmt.where(RoboRun.account_id == account_id)
        peak_value = session.scalar(stmt)
    except Exception as exc:  # noqa: BLE001 - history read must never break a run
        logger.warning("Drawdown history read failed ({e}); breaker not evaluated", e=exc)
        return False
    recorded_peak = Decimal(str(peak_value)) if peak_value is not None else Decimal("0")
    peak = max(current_total, recorded_peak)
    if peak <= 0:
        return False
    drawdown_pct = float((peak - current_total) / peak * 100)
    return drawdown_pct >= config.caps.max_drawdown_pct


def _utcnow():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def _persist_refused(
    settings: Settings, run_id: str, dry_run: bool, *,
    account_id: str, reason: str, status: str,
) -> None:
    """Record a refused/failed run so it shows up in ``robo status``."""
    try:
        init_db(settings.db_path)
        with get_session() as session:
            save_robo_run(session, RoboRun(
                run_id=run_id, dry_run=dry_run, account_id=account_id,
                status=status, notes=reason, finished_at=_utcnow(),
            ))
    except Exception as exc:  # noqa: BLE001 - never mask the original failure
        logger.warning("Could not persist refused run: {e}", e=exc)
