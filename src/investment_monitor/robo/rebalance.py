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

from investment_monitor.config import Settings
from investment_monitor.robo.audit import AuditLogger
from investment_monitor.robo.broker import BrokerError, PublicBroker
from investment_monitor.robo.config import RoboConfig
from investment_monitor.robo.gate import validate_orders
from investment_monitor.robo.llm import RoboProposer
from investment_monitor.robo.market_hours import is_market_open
from investment_monitor.robo.models import AccountState, GateDecision
from investment_monitor.robo.signals import fetch_signals
from investment_monitor.storage import (
    RoboOrder,
    RoboRun,
    count_placed_orders_today,
    finalize_robo_run,
    get_active_symbols,
    get_active_theses,
    get_recent_robo_runs,
    get_session,
    init_db,
    save_robo_order,
    save_robo_run,
)


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


def _reconcile_fill_costs(session, account: AccountState) -> None:
    """Persist the broker's real cost basis onto each active thesis.

    Writes ``entry_conditions['fill_cost']`` = the position's broker unit cost, so the
    feedback loop scores realized return against the *actual fill price* rather than the
    quote captured when the idea was written — and only for names we genuinely hold.
    Fully fail-open: a reconcile error must never abort a rebalance run, and it is a
    no-op in paper/dry-run (no broker cost basis), keeping that path byte-identical.
    """
    try:
        for thesis in get_active_theses(session, account.account_id or None):
            pos = account.get_position(thesis.symbol)
            if pos is None or pos.unit_cost is None or pos.unit_cost <= 0:
                continue
            new_cost = float(pos.unit_cost)
            entry = dict(thesis.entry_conditions or {})
            if entry.get("fill_cost") != new_cost:
                entry["fill_cost"] = new_cost
                thesis.entry_conditions = entry  # reassign so the JSON column is dirtied
    except Exception as exc:  # noqa: BLE001 - feedback bookkeeping must never break a run
        logger.warning("fill-cost reconcile failed: {e}", e=exc)


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
        # Record the broker's real cost basis onto held theses so the feedback loop
        # measures realized return from the actual fill, not the quote at idea time.
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
        for order in orders:
            audit.proposal(order)

        # In autonomous mode the tradeable universe is the set of names with a live
        # thesis, plus anything currently held (so positions can always be exited).
        # The gate then enforces it for free via its existing allowlist check, and
        # the additive `no_active_thesis` guard restricts BUYs to thesis names.
        gate_config = config
        active_symbols: set[str] | None = None
        if config.mode == "autonomous":
            active_symbols = get_active_symbols(session, account.account_id or None)
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

        run_row = RoboRun(
            run_id=run_id, dry_run=dry_run, account_id=account.account_id, source=source,
            total_value=float(account.total_value), settled_cash=float(account.settled_cash),
            num_proposed=len(orders), status="running",
        )
        save_robo_run(session, run_row)

        # --- 6. Guardrail gate ---------------------------------------------------
        orders_today = count_placed_orders_today(session)
        decisions = validate_orders(
            orders, account, gate_config, prices, orders_today=orders_today,
            active_symbols=active_symbols, halt_buys=halt_buys,
        )

        num_accepted = num_rejected = num_placed = 0
        final_status = "completed"
        try:
            for decision in decisions:
                audit.gate_decision(decision)
                order = decision.order
                order_row = RoboOrder(
                    run_id=run_id, symbol=order.symbol,
                    side=order.side.value, order_type=order.order_type.value,
                    quantity=float(order.quantity) if order.quantity is not None else None,
                    notional=float(order.notional) if order.notional is not None else None,
                    limit_price=float(order.limit_price) if order.limit_price is not None else None,
                    source=order.source, reason=order.reason,
                    gate_accepted=decision.accepted, gate_code=decision.code,
                    gate_reason=decision.reason,
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
                    else:
                        order_row.placed = True
                        order_row.broker_order_id = placed.order_id
                        order_row.status = placed.status or "placed"
                        num_placed += 1
                        audit.order_result(order, simulated=False, placed=True,
                                           broker_order_id=placed.order_id, status=placed.status)
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
        # Wide lookback so the true peak isn't silently truncated (the breaker would
        # otherwise under-state drawdown and let buys through during a real decline).
        runs = get_recent_robo_runs(session, limit=5000)
    except Exception as exc:  # noqa: BLE001 - history read must never break a run
        logger.warning("Drawdown history read failed ({e}); breaker not evaluated", e=exc)
        return False
    peaks = [
        Decimal(str(r.total_value))
        for r in runs
        if r.total_value and (not account_id or r.account_id == account_id)
    ]
    peak = max([current_total, *peaks])
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
