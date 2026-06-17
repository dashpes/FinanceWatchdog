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

import uuid
from dataclasses import dataclass, field
from decimal import Decimal

from loguru import logger

from investment_monitor.config import Settings
from investment_monitor.robo.audit import AuditLogger
from investment_monitor.robo.broker import BrokerError, PublicBroker, SafetyViolation
from investment_monitor.robo.config import RoboConfig
from investment_monitor.robo.gate import validate_orders
from investment_monitor.robo.llm import RoboProposer
from investment_monitor.robo.models import AccountState, GateDecision
from investment_monitor.robo.signals import fetch_signals
from investment_monitor.storage import (
    RoboOrder,
    RoboRun,
    count_gate_accepted_orders_today,
    finalize_robo_run,
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

        orders, source = proposer.propose(account, signals=snapshot)
        for order in orders:
            audit.proposal(order)

        run_row = RoboRun(
            run_id=run_id, dry_run=dry_run, account_id=account.account_id, source=source,
            total_value=float(account.total_value), settled_cash=float(account.settled_cash),
            num_proposed=len(orders), status="running",
        )
        save_robo_run(session, run_row)

        # --- 6. Guardrail gate ---------------------------------------------------
        orders_today = count_gate_accepted_orders_today(session)
        decisions = validate_orders(orders, account, config, prices, orders_today=orders_today)

        num_accepted = num_rejected = num_placed = 0
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

            # --- 7. Preflight ----------------------------------------------------
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

            # --- 8. Place or simulate -------------------------------------------
            if dry_run:
                order_row.simulated = True
                order_row.status = "simulated"
                num_placed += 1
                audit.order_result(order, simulated=True, placed=False, status="simulated")
            else:
                try:
                    placed = broker.place_order(order)
                    order_row.placed = True
                    order_row.broker_order_id = placed.order_id
                    order_row.status = placed.status or "placed"
                    num_placed += 1
                    audit.order_result(order, simulated=False, placed=True,
                                       broker_order_id=placed.order_id, status=placed.status)
                except (BrokerError, SafetyViolation) as exc:
                    order_row.status = "place_failed"
                    logger.error("Order placement failed for {s}: {e}", s=order.symbol, e=exc)
                    audit.order_result(order, simulated=False, placed=False, detail=str(exc))
            save_robo_order(session, order_row)

        # --- 9. Finalize ---------------------------------------------------------
        finalize_robo_run(
            session, run_id,
            finished_at=_utcnow(), status="completed",
            num_accepted=num_accepted, num_rejected=num_rejected, num_placed=num_placed,
        )

    audit.run_summary(
        num_proposed=len(orders), num_accepted=num_accepted, num_rejected=num_rejected,
        num_placed=num_placed, dry_run=dry_run, status="completed",
    )

    result = RebalanceResult(
        run_id=run_id, dry_run=dry_run, source=source, status="completed",
        account_id=account.account_id, total_value=account.total_value,
        settled_cash=account.settled_cash, num_proposed=len(orders),
        num_accepted=num_accepted, num_rejected=num_rejected, num_placed=num_placed,
        decisions=decisions,
    )
    logger.info("Rebalance complete: {s}", s=result.summary_line())
    return result


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
