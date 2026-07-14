"""Command-line interface for the robo advisor.

Usage:
    investment-robo check-safety [--config DIR] [--raw]
    investment-robo run [--dry-run | --live] [--config DIR]
    investment-robo status [--limit N] [--run-id RUN_ID]

`check-safety` connects to Public, confirms the account is cash-only, and exits
non-zero if margin is present. `run` executes one rebalance (dry-run unless live
is explicitly enabled AND the env kill-switch is off). `status` shows recent runs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer

from investment_monitor.config import get_settings
from investment_monitor.robo import tunables
from investment_monitor.robo.broker import BrokerError, PublicBroker
from investment_monitor.robo.config import ConfigError, RoboConfig
from investment_monitor.robo.notify import (
    format_daily_summary,
    notifications_configured,
    notify_error,
    notify_run,
    send_daily_summary,
    send_test,
    todays_trade_rows,
    trade_text_lines,
)
from investment_monitor.robo.rebalance import rebalance_run
from investment_monitor.storage import (
    accuracy_stats_for_symbol,
    get_filled_robo_orders,
    get_outcome_symbols,
    get_recent_robo_runs,
    get_robo_orders_for_run,
    get_session,
    init_db,
)

app = typer.Typer(
    name="investment-robo",
    help="Cash-only, long-only robo advisor for a Public.com account.",
    no_args_is_help=True,
)


def _load_config(config_dir: Path | None) -> RoboConfig:
    settings = get_settings()
    cfg_dir = config_dir or settings.config_dir
    try:
        return RoboConfig.from_yaml(Path(cfg_dir) / "robo.yaml")
    except ConfigError as exc:
        # A bad robo.yaml must never crash a CLI command or a launchd daemon
        # with a raw traceback (which would silently halt the autonomous
        # trader). Surface one clear, actionable message and exit non-zero.
        typer.secho(f"Config error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


def _config_path(config_dir: Path | None) -> Path:
    cfg_dir = config_dir or get_settings().config_dir
    return Path(cfg_dir) / "robo.yaml"


# --- `config`: view + tune settings (the same schema a GUI would render) ----------

config_app = typer.Typer(
    name="config",
    help="View and tune robo settings. The catalog is shared with any future GUI.",
    no_args_is_help=True,
)
app.add_typer(config_app)

# Settings whose change affects real-money behavior — flagged on `set`.
_SAFETY_KEYS = {"dry_run", "mode"}


def _fmt_range(t: tunables.Tunable) -> str:
    if t.choices:
        return "{" + " | ".join(t.choices) + "}"
    if t.minimum is not None or t.maximum is not None:
        lo = "" if t.minimum is None else f"{t.minimum:g}"
        hi = "" if t.maximum is None else f"{t.maximum:g}"
        return f"[{lo}…{hi}]"
    return ""


@config_app.command("list")
def config_list(
    group: str = typer.Option(None, "--group", "-g", help="Filter to one group."),
    as_json: bool = typer.Option(False, "--json", help="Emit the catalog + current values as JSON."),
    config_dir: Path = typer.Option(None, "--config", help="Config directory."),
) -> None:
    """List every tunable setting with its current value, default, and bounds."""
    cfg = _load_config(config_dir)
    items = [t for t in tunables.catalog() if group is None or t.group.lower() == group.lower()]
    if as_json:
        payload = [{**t.as_dict(), "current": tunables.get_value(cfg, t.key)} for t in items]
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    if not items:
        typer.echo("No settings match.")
        return
    current_group = None
    for t in items:
        if t.group != current_group:
            current_group = t.group
            typer.secho(f"\n{current_group}", fg=typer.colors.CYAN, bold=True)
        value = tunables.get_value(cfg, t.key)
        rng = _fmt_range(t)
        line = f"  {t.key:<24} = {str(value):<12} default {str(t.default):<8} {rng}"
        typer.echo(line.rstrip())
        if t.description:
            typer.secho(f"      {t.description}", fg=typer.colors.BRIGHT_BLACK)


@config_app.command("get")
def config_get(
    key: str = typer.Argument(..., help="Dotted setting key, e.g. caps.max_positions."),
    config_dir: Path = typer.Option(None, "--config", help="Config directory."),
) -> None:
    """Show one setting's current value and metadata."""
    t = {x.key: x for x in tunables.catalog()}.get(key)
    if t is None:
        typer.secho(f"Unknown setting '{key}'. Run `config list`.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    value = tunables.get_value(_load_config(config_dir), key)
    typer.secho(f"{t.key} = {value}", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  {t.title} — {t.description}")
    typer.echo(f"  type: {t.type}   default: {t.default}   {_fmt_range(t)}".rstrip())
    if t.unit:
        typer.echo(f"  unit: {t.unit}")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Dotted setting key, e.g. caps.max_positions."),
    value: str = typer.Argument(..., help="New value (validated against the schema)."),
    config_dir: Path = typer.Option(None, "--config", help="Config directory."),
) -> None:
    """Validate and write a setting to robo.yaml (comments preserved)."""
    path = _config_path(config_dir)
    # The OLD value is shown for context only. Loading the current config to read
    # it must NOT block the write: this command is the operator's tool for REPAIRING
    # a robo.yaml that is currently out of bounds (which makes _load_config exit) or
    # otherwise unloadable. The authoritative safety check is tunables.set_value,
    # which validates the FULL merged config before writing — so a bad sibling field
    # still blocks an invalid write, but a valid single-key repair succeeds.
    try:
        old = tunables.get_value(_load_config(config_dir), key)
    except (AttributeError, typer.Exit):
        old = "(unavailable)"
    try:
        new = tunables.set_value(path, key, value)
    except ValueError as exc:
        typer.secho(f"Rejected: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho(f"Set {key}: {old} → {new}", fg=typer.colors.GREEN)
    if key in _SAFETY_KEYS:
        typer.secho(
            "  ⚠ this affects live trading behavior — confirm it's intended.",
            fg=typer.colors.YELLOW,
        )
    typer.secho(
        "  A running daemon picks this up on its next scheduled run.",
        fg=typer.colors.BRIGHT_BLACK,
    )


@app.command("accounts")
def accounts(
    config: Path = typer.Option(None, "--config", "-c", help="Config directory"),
) -> None:
    """List all accounts your token can see, so you can pick the cash one."""
    settings = get_settings()
    broker = PublicBroker(
        api_token=settings.public_api_token,
        base_url=settings.public_api_base_url,
        dry_run=True,
    )
    try:
        accts = broker.list_accounts()
    except BrokerError as exc:
        typer.secho(f"ERROR: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(f"{'account_id':<24} {'type':<14} cash?")
    for a in accts:
        flag = "CASH ✓" if a["is_cash"] else (a["brokerage_account_type"] or "?")
        color = typer.colors.GREEN if a["is_cash"] else typer.colors.YELLOW
        typer.secho(f"{a['account_id']:<24} {a['account_type']:<14} {flag}", fg=color)
    cash = [a for a in accts if a["is_cash"]]
    if cash:
        typer.echo(
            f"\nSet `account_id: \"{cash[0]['account_id']}\"` in config/robo.yaml "
            "(the CASH account)."
        )
    else:
        typer.secho(
            "\nNo CASH account found — the robo advisor requires one and will refuse to run.",
            fg=typer.colors.RED,
        )


@app.command("check-safety")
def check_safety(
    config: Path = typer.Option(None, "--config", "-c", help="Config directory"),
    raw: bool = typer.Option(False, "--raw", help="Print raw account/portfolio payloads"),
) -> None:
    """Confirm the configured account is cash-only and print balances. Exits non-zero on margin."""
    settings = get_settings()
    cfg = _load_config(config)
    broker = PublicBroker(
        api_token=settings.public_api_token,
        account_id=cfg.account_id,
        base_url=settings.public_api_base_url,
        dry_run=True,
    )
    try:
        account = broker.get_account_state()
    except BrokerError as exc:
        typer.secho(f"ERROR: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Account:        {account.account_id} ({account.account_type})")
    typer.echo(f"Cash account:   {account.is_cash_account}")
    typer.echo(f"Margin enabled: {account.has_margin}")
    typer.echo(f"Settled cash:   ${account.settled_cash}")
    typer.echo(f"Positions value:${account.positions_value}")
    typer.echo(f"Total value:    ${account.total_value}")
    if account.positions:
        typer.echo("Positions:")
        for p in account.positions:
            line = f"  {p.symbol:<6} {p.quantity} @ ${p.price} = ${p.market_value}"
            if p.unit_cost is not None:
                gain = p.unrealized_gain
                gain_str = f"${gain:+.2f}" if gain is not None else "n/a"
                line += f"  (cost ${p.unit_cost}/sh, unrealized {gain_str})"
            typer.echo(line)
    total_unrl = account.total_unrealized_gain
    if total_unrl is not None:
        typer.echo(f"Unrealized P&L: ${total_unrl:+.2f}")
    if raw:
        typer.echo("\nRaw payloads (verify field mapping):")
        typer.echo(str(account.raw))

    if not account.is_cash_account or account.has_margin:
        typer.secho(
            "\nUNSAFE: account is not cash-only / has margin. The robo advisor will refuse to run.",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)
    typer.secho("\nSAFE: cash-only account confirmed.", fg=typer.colors.GREEN)


@app.command("run")
def run(
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Force simulation (no real orders)"),
    live: bool = typer.Option(False, "--live", help="Attempt live trading (still gated by env + config)"),
    config: Path = typer.Option(None, "--config", "-c", help="Config directory"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the live-trading confirmation prompt"),
) -> None:
    """Run one rebalance. Dry-run unless --live is set AND ROBO_FORCE_DRY_RUN is false."""
    if dry_run and live:
        typer.secho("Choose only one of --dry-run / --live", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    settings = get_settings()
    cfg = _load_config(config)

    override: bool | None = True if dry_run else (False if live else None)
    # Confirm before any potentially-live run.
    if live and not settings.robo_force_dry_run and not yes:
        if not typer.confirm("This may place REAL orders with REAL money. Continue?"):
            raise typer.Exit(code=1)

    try:
        result = rebalance_run(cfg, settings, dry_run_override=override)
    except BrokerError as exc:
        notify_error(settings, message=str(exc), dry_run=override)
        typer.secho(f"ERROR: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    notify_run(result, settings)
    typer.echo(result.summary_line())
    if result.status == "refused":
        typer.secho(f"Refused: {result.message}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    if result.status == "failed":
        typer.secho(f"Failed: {result.message}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    rejected = [d for d in result.decisions if not d.accepted]
    if rejected:
        typer.echo("Rejected orders:")
        for d in rejected:
            o = d.order
            size = f"${o.notional}" if o.notional is not None else f"{o.quantity} sh"
            typer.echo(f"  {o.side.value:<4} {o.symbol:<6} {size:<10} -> {d.code}: {d.reason}")


@app.command("thesis-run")
def thesis_run(
    config: Path = typer.Option(None, "--config", "-c", help="Config directory"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Force simulation (no real orders)"),
    skip_maintenance: bool = typer.Option(
        False, "--skip-maintenance", help="Skip LLM thesis re-eval; just rebalance to conviction"
    ),
    discover: bool = typer.Option(
        False, "--discover", help="Run research discovery first to source new candidates"
    ),
    no_trade: bool = typer.Option(
        False, "--no-trade", help="Research + maintain theses only; skip the rebalance/trading step"
    ),
) -> None:
    """Autonomous loop: discover -> promote -> re-evaluate theses -> rebalance to conviction.

    Forces autonomous mode regardless of robo.yaml. Live trading still requires
    ROBO_FORCE_DRY_RUN=false AND dry_run=false (both off by default), so this is a
    safe paper-trading loop out of the box.
    """
    from investment_monitor.analysis.model_router import ModelRouter
    from investment_monitor.analysis.thesis_evaluator import (
        ThesisEvaluator,
        refresh_target_weights,
    )
    from investment_monitor.storage import get_active_theses

    settings = get_settings()
    cfg = _load_config(config)
    auto_cfg = cfg if cfg.mode == "autonomous" else cfg.model_copy(update={"mode": "autonomous"})
    acct = auto_cfg.account_id or None

    init_db(settings.db_path)
    # Thesis synthesis uses the stronger 'synthesis'-role model (Phase 5 routing).
    synth_llm = None
    if auto_cfg.use_llm:
        try:
            from investment_monitor.analysis.local_llm import LocalLLM

            synth_llm = LocalLLM(
                model=ModelRouter(settings).resolve("synthesis", base_url=settings.ollama_host),
                base_url=settings.ollama_host,
            )
        except ImportError:
            synth_llm = None
    evaluator = ThesisEvaluator(synth_llm, auto_cfg)

    # 0. The agent runs its OWN research: discover + score candidates into the funnel.
    if discover or auto_cfg.autonomy.discover:
        import asyncio

        from investment_monitor.research.discovery import DiscoveryPipeline
        from investment_monitor.research_cli import _load_research_config

        research_config = _load_research_config(settings.config_dir)
        typer.echo("Running research discovery (collect + AI-score the universe)...")
        with get_session() as session:
            pipeline = DiscoveryPipeline(
                session=session,
                config=settings,
                research_config=research_config,
                ollama_model=ModelRouter(settings).resolve("scoring", base_url=settings.ollama_host),
            )
            result = asyncio.run(pipeline.run_discovery(dry_run=False))
        typer.echo(
            f"Discovery: {result.scored_candidates} scored, "
            f"{len(result.watchlist_additions)} reached the watchlist "
            f"({result.duration_seconds:.0f}s)"
        )

    # 1. Autonomous selection: promote eligible discovery candidates to theses.
    if auto_cfg.autonomy.enabled:
        from investment_monitor.robo.promotion import promote_candidates

        with get_session() as session:
            promoted = promote_candidates(session, auto_cfg, evaluator=evaluator, account_id=acct)
        if promoted:
            typer.echo(f"Promoted {len(promoted)} new name(s): {', '.join(promoted)}")

    # 1.5 Shadow ledger: sweep gate-rejects + discovery near-misses into the ledger,
    #     mark open counterfactuals, close those past horizon. Fail-open bookkeeping.
    try:
        from investment_monitor.robo.shadow import maintain_shadow_ledger

        with get_session() as session:
            sh = maintain_shadow_ledger(
                session,
                score_floor=float(auto_cfg.autonomy.score_floor),
                account_id=acct,
            )
        if any(sh.values()):
            typer.echo(
                f"Shadow ledger: +{sh['gate']} gate, +{sh['discovery']} discovery, "
                f"{sh['marked']} marked, {sh['closed']} closed"
            )
    except Exception as exc:  # noqa: BLE001 - shadow bookkeeping must never block a run
        typer.secho(f"shadow ledger maintenance skipped: {exc}", fg=typer.colors.YELLOW)

    # 2. Maintain existing theses (deterministic invalidation, then LLM re-eval).
    if not skip_maintenance:
        actions = {"invalidated": 0, "exited": 0, "updated": 0, "unchanged": 0}
        with get_session() as session:
            for thesis in get_active_theses(session, acct):
                actions[evaluator.evaluate(session, thesis, account_id=acct)] += 1
        typer.echo(
            f"Thesis maintenance: {actions['updated']} updated, "
            f"{actions['invalidated']} invalidated, {actions['exited']} exited (profit/horizon), "
            f"{actions['unchanged']} unchanged"
        )

    # 3. Recompute sized target weights from current convictions.
    with get_session() as session:
        refresh_target_weights(session, auto_cfg, account_id=acct)

    # 4. Trade — unless this is a research-only run (24/7 schedule uses --no-trade).
    if no_trade:
        typer.echo("Research/maintenance complete (--no-trade: skipped the rebalance step).")
        return

    try:
        result = rebalance_run(auto_cfg, settings, dry_run_override=(True if dry_run else None))
    except BrokerError as exc:
        notify_error(settings, message=str(exc), dry_run=True if dry_run else None)
        typer.secho(f"ERROR: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    notify_run(result, settings)
    typer.echo(result.summary_line())
    if result.status in ("refused", "failed"):
        typer.secho(f"{result.status}: {result.message}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command("status")
def status(
    limit: int = typer.Option(10, "--limit", help="Number of recent runs to show"),
    run_id: str = typer.Option("", "--run-id", help="Show orders for a specific run"),
) -> None:
    """Show recent rebalance runs (and orders for a given run)."""
    settings = get_settings()
    init_db(settings.db_path)
    with get_session() as session:
        if run_id:
            orders = get_robo_orders_for_run(session, run_id)
            if not orders:
                typer.echo(f"No orders for run {run_id}")
                return
            for o in orders:
                size = f"${o.notional}" if o.notional is not None else f"{o.quantity} sh"
                state = o.status or ("accepted" if o.gate_accepted else "rejected")
                fill = f" filled {o.fill_quantity}@${o.fill_price}" if o.fill_price is not None else ""
                typer.echo(
                    f"  {o.side:<4} {o.symbol:<6} {size:<10} {state:<16} "
                    f"{o.gate_code or ''} {o.gate_reason or ''}{fill}"
                )
            return

        runs = get_recent_robo_runs(session, limit=limit)
        if not runs:
            typer.echo("No robo runs recorded yet.")
            return
        typer.echo(
            f"{'started':<17} {'mode':<7} {'status':<10} {'value':>10} {'unreal':>9} "
            f"{'p/a/r/x':<10} run_id"
        )
        for r in runs:
            mode = "dry-run" if r.dry_run else "LIVE"
            started = r.started_at.strftime("%Y-%m-%d %H:%M") if r.started_at else "?"
            counts = f"{r.num_proposed}/{r.num_accepted}/{r.num_rejected}/{r.num_placed}"
            value = f"${r.total_value:,.0f}" if r.total_value is not None else "-"
            unreal = f"${r.unrealized_pnl:+,.0f}" if r.unrealized_pnl is not None else "-"
            typer.echo(
                f"{started:<17} {mode:<7} {r.status:<10} {value:>10} {unreal:>9} "
                f"{counts:<10} {r.run_id}"
            )


@app.command("pnl")
def pnl(
    config: Path = typer.Option(None, "--config", "-c", help="Config directory"),
) -> None:
    """Show live positions with unrealized P&L (read straight from the broker)."""
    settings = get_settings()
    cfg = _load_config(config)
    broker = PublicBroker(
        api_token=settings.public_api_token,
        account_id=cfg.account_id,
        base_url=settings.public_api_base_url,
        dry_run=True,
    )
    try:
        account = broker.get_account_state()
    except BrokerError as exc:
        typer.secho(f"ERROR: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Account: {account.account_id}    Total value: ${account.total_value}")
    if not account.positions:
        typer.echo("No open positions.")
        return

    header = f"{'symbol':<6} {'qty':>10} {'cost/sh':>10} {'price':>10} {'mkt val':>12} {'unreal $':>12} {'unreal %':>9}"
    typer.echo(header)
    for p in sorted(account.positions, key=lambda x: x.symbol):
        cost = f"${p.unit_cost}" if p.unit_cost is not None else "n/a"
        gain = f"{p.unrealized_gain:+.2f}" if p.unrealized_gain is not None else "n/a"
        ret = p.unrealized_return
        ret_str = f"{ret * 100:+.1f}%" if ret is not None else "n/a"
        typer.echo(
            f"{p.symbol:<6} {str(p.quantity):>10} {cost:>10} {f'${p.price}':>10} "
            f"{f'${p.market_value:.2f}':>12} {gain:>12} {ret_str:>9}"
        )

    total_unrl = account.total_unrealized_gain
    basis = account.total_cost_basis
    typer.echo(f"\nSettled cash:   ${account.settled_cash}")
    typer.echo(f"Positions value:${account.positions_value:.2f}")
    if basis is not None:
        typer.echo(f"Cost basis:     ${basis:.2f}")
    if total_unrl is not None:
        pct = f" ({total_unrl / basis * 100:+.1f}%)" if basis and basis > 0 else ""
        typer.echo(f"Unrealized P&L: ${total_unrl:+.2f}{pct}")
    else:
        typer.echo("Unrealized P&L: n/a (broker reported no cost basis)")

    # Realized P&L, reconstructed from the bot's OWN filled orders (not the shared
    # account's history) so manual/pre-existing positions never count as robo gains.
    realized_total = None
    try:
        from investment_monitor.robo.pnl import realized_pnl, trades_from_fills

        init_db(settings.db_path)
        with get_session() as session:
            rp = realized_pnl(trades_from_fills(get_filled_robo_orders(session)))
        realized_total = rp.total_realized
        realized_syms = {s: sp for s, sp in rp.per_symbol.items() if sp.realized != 0}
        if realized_syms:
            typer.echo("\nRealized P&L (robo trades only):")
            for sym, sp in sorted(realized_syms.items()):
                typer.echo(f"  {sym:<6} ${sp.realized:+.2f}")
        typer.echo(f"Realized P&L:   ${realized_total:+.2f}")
    except Exception as exc:  # noqa: BLE001 - reporting only; never fail the command
        typer.secho(f"Realized P&L:   unavailable ({exc})", fg=typer.colors.YELLOW)

    if realized_total is not None and total_unrl is not None:
        typer.echo(f"Total P&L:      ${realized_total + total_unrl:+.2f}  (realized + unrealized)")


@app.command("daily-summary")
def daily_summary(
    config: Path = typer.Option(None, "--config", "-c", help="Config directory"),
) -> None:
    """Text a daily portfolio + P&L summary via iMessage (and print it).

    Read-only: fetches live account state from the broker, never trades. Intended to
    run once a day after the close (see scripts/robo-schedule.sh). A no-op delivery
    when IMESSAGE_TO is unset — the summary still prints.
    """
    settings = get_settings()
    cfg = _load_config(config)
    broker = PublicBroker(
        api_token=settings.public_api_token,
        account_id=cfg.account_id,
        base_url=settings.public_api_base_url,
        dry_run=True,
    )
    try:
        account = broker.get_account_state()
    except BrokerError as exc:
        # A broken summary run is itself worth knowing about.
        notify_error(settings, message=f"daily-summary: {exc}")
        typer.secho(f"ERROR: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    realized = None
    try:
        from investment_monitor.robo.pnl import realized_pnl, trades_from_fills

        init_db(settings.db_path)
        with get_session() as session:
            realized = realized_pnl(trades_from_fills(get_filled_robo_orders(session)))
    except Exception as exc:  # noqa: BLE001 - realized P&L is best-effort, never required
        typer.secho(f"(realized P&L unavailable: {exc})", fg=typer.colors.YELLOW)

    trade_rows = todays_trade_rows(settings)
    trades = trade_text_lines(trade_rows)
    typer.echo(format_daily_summary(account, realized, trades))
    sent = send_daily_summary(
        settings,
        account,
        realized,
        trades,
        trade_rows=trade_rows,
        dry_run=settings.robo_force_dry_run or cfg.dry_run,
    )
    if notifications_configured(settings) and not sent:
        typer.secho(
            "(summary not sent — check notification config / logs)",
            fg=typer.colors.YELLOW,
        )


@app.command("notify-test")
def notify_test() -> None:
    """Send a test notification (email or iMessage) to confirm the channel is wired up."""
    settings = get_settings()
    email_on = bool((settings.smtp_host or "").strip() and (settings.email_to or "").strip())
    imsg_on = bool((settings.imessage_to or "").strip())
    if not (email_on or imsg_on):
        typer.secho(
            "No notification channel configured. Set SMTP_HOST + EMAIL_TO (email, "
            "recommended) or IMESSAGE_TO (iMessage) in .env.",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=1)

    channel, target = ("email", settings.email_to) if email_on else ("iMessage", settings.imessage_to)
    if send_test(settings):
        typer.secho(f"Sent test {channel} to {target}.", fg=typer.colors.GREEN)
    else:
        hint = (
            "Check SMTP_HOST/PORT/USERNAME/PASSWORD (Gmail needs an App Password) and logs."
            if email_on
            else "Grant Automation permission to the runner (System Settings > Privacy & "
            "Security > Automation) and ensure Messages.app is signed in."
        )
        typer.secho(f"Failed to send {channel}. {hint}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command("pause")
def pause(
    reason: str = typer.Option("", "--reason", "-r", help="Why trading is paused."),
) -> None:
    """Pause trading (the next trade runs record 'paused' and skip the broker).

    Research, discovery, and data collection keep running. Resume with
    `investment-robo resume`. The dashboard uses the same control file.
    """
    from investment_monitor.robo import control

    state = control.set_paused(get_settings().db_path, True, reason=reason, updated_by="cli")
    suffix = f" ({state.reason})" if state.reason else ""
    typer.secho(f"Trading PAUSED{suffix}. Resume with: investment-robo resume", fg=typer.colors.YELLOW)


@app.command("resume")
def resume() -> None:
    """Resume trading after a pause. (Never arms live mode by itself.)"""
    from investment_monitor.robo import control

    settings = get_settings()
    control.set_paused(settings.db_path, False, updated_by="cli")
    state = control.load_control(settings.db_path)
    typer.secho("Trading resumed.", fg=typer.colors.GREEN)
    if state.force_dry_run or settings.robo_force_dry_run:
        typer.echo("Note: paper mode is still forced (control file / ROBO_FORCE_DRY_RUN).")


@app.command("learning")
def learning(
    limit: int = typer.Option(20, "--limit", help="Recent outcomes per symbol to aggregate"),
) -> None:
    """Show what the feedback loop has learned per symbol (from learning_events).

    The ledger holds the full history; this prints only the compact aggregates that
    actually feed sizing and the re-eval prompt — hit rate, recency-weighted hit
    rate, and calibration (1 - Brier; higher is better-calibrated).
    """
    settings = get_settings()
    init_db(settings.db_path)
    with get_session() as session:
        symbols = get_outcome_symbols(session)
        if not symbols:
            typer.echo("No learning outcomes recorded yet.")
            return
        typer.echo(f"{'symbol':<8} {'n':>4} {'hit%':>6} {'ewma%':>6} {'calib':>6}")
        for sym in symbols:
            st = accuracy_stats_for_symbol(session, sym, recent_window=limit)
            typer.echo(
                f"{sym:<8} {st['n']:>4} {st['hit_rate'] * 100:>5.0f}% "
                f"{st['ewma_hit_rate'] * 100:>5.0f}% {1.0 - st['brier']:>6.2f}"
            )


@app.command("backtest")
def backtest(
    days: int = typer.Option(180, "--days", help="Replay window ending today"),
    step: int = typer.Option(5, "--step", help="Days between signal re-evaluations"),
    horizon: int = typer.Option(90, "--horizon", help="Max holding period (days)"),
    min_score: float = typer.Option(4.0, "--min-score", help="Promotion score floor to test"),
    profit_target: float = typer.Option(
        None, "--profit-target", help="Take-profit at this % gain from entry (off if omitted)"
    ),
    trailing_stop: float = typer.Option(
        None, "--trailing-stop", help="Exit this % below the post-entry high (off if omitted)"
    ),
    trailing_arm: float = typer.Option(
        10.0, "--trailing-arm", help="Gain % from entry required before the trailing stop arms"
    ),
) -> None:
    """Walk-forward replay of confluence -> promotion -> exits over stored history.

    Uses the REAL production scoring and guards as-of each past date (insider +
    volume sources; no look-ahead). Depth is bounded by ingested history — run
    'investment-monitor --type collect-broad --days-back N' first to backfill
    EDGAR, and note retention windows cap what is kept.
    """
    from datetime import date, timedelta

    from investment_monitor.simulation.backtest import run_confluence_backtest

    settings = get_settings()
    init_db(settings.db_path)
    end = date.today()
    start = end - timedelta(days=days)
    with get_session() as session:
        result = run_confluence_backtest(
            session, start=start, end=end, step_days=step,
            horizon_days=horizon, promote_min_score=min_score,
            profit_target_pct=profit_target, trailing_stop_pct=trailing_stop,
            trailing_arm_pct=trailing_arm,
        )
    s = result.summary()

    def _fmt(st: dict) -> str:
        if not st["n"]:
            return "n=0"
        return (
            f"n={st['n']:<3} hit {st['hit_rate'] * 100:>3.0f}%  "
            f"avg {st['avg'] * 100:+6.1f}%  med {st['median'] * 100:+6.1f}%  "
            f"best {st['best'] * 100:+.0f}%  worst {st['worst'] * 100:+.0f}%"
        )

    typer.echo(f"Backtest {s['start']} -> {s['end']} ({s['steps']} steps, "
               f"{s['n_trades']} trades, {s['n_closed']} closed)")
    typer.echo(f"  overall      {_fmt(s['overall'])}")
    for band, st in s["by_score_band"].items():
        typer.echo(f"  score {band:<6} {_fmt(st)}")
    for reason, st in s["by_exit_reason"].items():
        typer.echo(f"  exit {reason:<9} {_fmt(st)}")
    if not s["n_trades"]:
        typer.echo("No trades — likely not enough insider/price history ingested "
                   "for this window.")


@app.command("sentinel")
def sentinel(
    config: Path = typer.Option(None, "--config", "-c", help="Config directory"),
) -> None:
    """Intraday watchdog over open positions: invalidate/flag only, never buy.

    No-op outside regular trading hours, so an hourly timer needs no market-
    calendar logic. A tripped invalidation zeroes the thesis; the actual sell
    happens at the next scheduled, fully-gated trade run.
    """
    from investment_monitor.robo.sentinel import run_sentinel

    settings = get_settings()
    cfg = _load_config(config)
    init_db(settings.db_path)
    result = run_sentinel(settings, cfg)
    if result["status"] == "market_closed":
        typer.echo("Market closed — sentinel pass skipped.")
        return
    typer.echo(
        f"Sentinel: {result['checked']} position(s) checked, "
        f"{len(result['tripped'])} invalidated, {len(result['exited'])} exited, "
        f"{len(result['flagged'])} flagged"
    )
    for line in result["tripped"] + result["exited"] + result["flagged"]:
        typer.echo(f"  - {line}")


@app.command("shadow")
def shadow(
    limit: int = typer.Option(15, "--limit", help="Open entries to list"),
    evaluate: bool = typer.Option(
        False, "--evaluate", help="Run a maintenance pass (sync + mark/close) first"
    ),
) -> None:
    """Traded vs skipped: how the theses we did NOT take are performing.

    The shadow ledger tracks every considered-but-skipped thesis (promotion floor,
    caps, liquidity/run-up guards, gate rejects) at its skip-day price, so the skip
    policy itself gets a report card next to the real-money outcomes.
    """
    from investment_monitor.robo.shadow import maintain_shadow_ledger, shadow_report
    from investment_monitor.storage import SHADOW_STATUS_OPEN, get_shadow_entries

    settings = get_settings()
    init_db(settings.db_path)
    with get_session() as session:
        if evaluate:
            maintain_shadow_ledger(session)
        report = shadow_report(session)
        entries = get_shadow_entries(session, status=SHADOW_STATUS_OPEN, limit=limit)

        real = report["real"]
        if real["n"]:
            typer.echo(
                f"real outcomes: n={real['n']}  hit {real['hit_rate'] * 100:.0f}%  "
                f"avg {real['avg_return'] * 100:+.1f}%"
            )
        else:
            typer.echo("real outcomes: none recorded yet")
        for source, st in sorted(report["shadow"].items()):
            hit = f"{st['hit_rate'] * 100:.0f}%" if st["hit_rate"] is not None else "—"
            avg = f"{st['avg_return'] * 100:+.1f}%" if st["avg_return"] is not None else "—"
            mark = f"{st['open_mark'] * 100:+.1f}%" if st["open_mark"] is not None else "—"
            typer.echo(
                f"shadow[{source}]: open {st['open']} (mark {mark})  "
                f"closed {st['closed']}  hit {hit}  avg {avg}"
            )
        if entries:
            typer.echo(f"\n{'symbol':<8} {'source':<12} {'reason':<20} {'entry':>8} {'mark':>7}")
            for e in entries:
                mark = f"{e.realized_return * 100:+.1f}%" if e.realized_return is not None else "—"
                price = f"${e.entry_price:,.2f}" if e.entry_price else "—"
                typer.echo(
                    f"{e.symbol:<8} {e.source:<12} {e.skip_reason:<20} {price:>8} {mark:>7}"
                )


@app.command("prune")
def prune() -> None:
    """Prune old market data to keep the SQLite DB bounded (weekly on the Pi).

    Applies the RETENTION_* windows from settings to the broad, market-wide tables
    (insider / news / prices / confluence findings) and VACUUMs to reclaim space. A
    window of 0 keeps everything for that source. Read-mostly and safe to run anytime;
    routed through the shared run-lock by systemd so it never prunes mid-trade.
    """
    from investment_monitor.storage.retention import RetentionConfig, prune_old_data

    settings = get_settings()
    init_db(settings.db_path)
    cfg = RetentionConfig(
        insider_days=settings.retention_insider_days,
        news_days=settings.retention_news_days,
        price_days=settings.retention_price_days,
        findings_days=settings.retention_findings_days,
        events_days=settings.retention_events_days,
    )
    if not cfg.any_enabled():
        typer.echo("Retention disabled (all windows 0) — nothing to prune.")
        return
    with get_session() as session:
        deleted = prune_old_data(session, cfg)
    for table, n in sorted(deleted.items()):
        typer.echo(f"  {table:<22} {n:>8,} rows pruned")
    typer.echo(f"Prune complete ({sum(deleted.values()):,} rows removed).")


@app.command("init")
def init(
    config: Path = typer.Option(None, "--config", "-c", help="Config directory"),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", help="Scaffold files with defaults; ask nothing"
    ),
) -> None:
    """Onboarding wizard: write .env (0600) + config/robo.yaml, dry-run forced ON.

    Safe to re-run — existing values become the defaults. This NEVER arms live
    trading: it forces ROBO_FORCE_DRY_RUN=true and robo.yaml dry_run:true. Flip both
    only after reviewing a dry-run cycle (see the printed go-live checklist).
    """
    from investment_monitor.robo.onboarding import parse_env, set_yaml_scalar, upsert_env

    # Anchor .env to $FW_HOME when set (the installer and systemd units run the app from
    # there, and that is where pydantic reads .env at runtime), else the current dir.
    # Without this, a `sudo -u` install would write .env to the caller's CWD (e.g. /root)
    # where the services never read it.
    root = Path(os.environ["FW_HOME"]) if os.environ.get("FW_HOME") else Path.cwd()
    env_path = root / ".env"
    env_example = root / ".env.example"
    cfg_dir = Path(config) if config else get_settings().config_dir
    yaml_path = cfg_dir / "robo.yaml"
    yaml_example = cfg_dir / "robo.yaml.example"

    # 1) Seed .env from the example if it doesn't exist yet (owner-only from birth).
    if not env_path.exists():
        env_path.write_text(env_example.read_text() if env_example.exists() else "")
        os.chmod(env_path, 0o600)
    existing = parse_env(env_path.read_text())

    def ask(key: str, prompt: str, *, secret: bool = False, required: bool = False) -> str:
        cur = existing.get(key, "")
        if non_interactive:
            return cur
        shown = "********" if (secret and cur) else cur
        val = typer.prompt(prompt, default=shown, show_default=bool(shown))
        if secret and val == "********":  # kept the masked existing secret
            return cur
        val = val.strip()
        if required and not val:
            typer.secho(f"{key} is required to run the robo advisor.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        return val

    updates: dict[str, str] = {}
    updates["PUBLIC_API_TOKEN"] = ask(
        "PUBLIC_API_TOKEN", "Public.com API token", secret=True, required=not non_interactive
    )
    updates["SEC_CONTACT_EMAIL"] = ask("SEC_CONTACT_EMAIL", "Your email (SEC EDGAR contact)")
    updates["SMTP_HOST"] = ask("SMTP_HOST", "SMTP host for email (blank = no email)")
    if updates["SMTP_HOST"]:
        updates["SMTP_PORT"] = ask("SMTP_PORT", "SMTP port") or "587"
        updates["SMTP_USERNAME"] = ask("SMTP_USERNAME", "SMTP username")
        updates["SMTP_PASSWORD"] = ask("SMTP_PASSWORD", "SMTP / app password", secret=True)
        updates["EMAIL_FROM"] = ask("EMAIL_FROM", "From address") or updates["SMTP_USERNAME"]
        updates["EMAIL_TO"] = ask("EMAIL_TO", "Send summaries/alerts to")
    updates["ANTHROPIC_API_KEY"] = ask(
        "ANTHROPIC_API_KEY", "Anthropic API key (optional; blank = local models only)", secret=True
    )
    updates["ROBO_FORCE_DRY_RUN"] = "true"  # safety: never arm live trading at init.

    env_path.write_text(upsert_env(env_path.read_text(), updates))
    os.chmod(env_path, 0o600)
    typer.secho(f"Wrote {env_path} (permissions 0600).", fg=typer.colors.GREEN)

    # 2) Seed robo.yaml from the example and force dry_run on.
    if not yaml_path.exists():
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(
            yaml_example.read_text()
            if yaml_example.exists()
            else 'mode: advisory\ndry_run: true\naccount_id: ""\n'
        )
    ytext = set_yaml_scalar(yaml_path.read_text(), "dry_run", "true")

    # 3) Discover the CASH account with the just-entered token (best-effort).
    if updates.get("PUBLIC_API_TOKEN"):
        try:
            broker = PublicBroker(
                api_token=updates["PUBLIC_API_TOKEN"],
                base_url=existing.get("PUBLIC_API_BASE_URL", ""),
                dry_run=True,
            )
            cash = [a for a in broker.list_accounts() if a.get("is_cash")]
            if cash:
                ytext = set_yaml_scalar(ytext, "account_id", f'"{cash[0]["account_id"]}"')
                typer.secho(f"Configured CASH account {cash[0]['account_id']}.", fg=typer.colors.GREEN)
            else:
                typer.secho(
                    "No CASH account found — set account_id manually; the robo requires one.",
                    fg=typer.colors.YELLOW,
                )
        except Exception as exc:  # noqa: BLE001 - discovery is best-effort at onboarding.
            typer.secho(
                f"Couldn't list accounts ({exc}). Set account_id later via `investment-robo accounts`.",
                fg=typer.colors.YELLOW,
            )
    yaml_path.write_text(ytext)
    typer.secho(f"Wrote {yaml_path} (dry_run: true).", fg=typer.colors.GREEN)

    typer.echo(
        "\nNext:\n"
        "  1. Pull the models:  ollama pull phi3:mini nomic-embed-text qwen2.5:14b\n"
        "  2. Verify email:     investment-robo notify-test\n"
        "  3. Dry-run a cycle:  investment-robo thesis-run --discover --no-trade\n"
        "  4. Confirm the account is cash-only:  investment-robo check-safety\n"
        "  GO LIVE (only when ready): set ROBO_FORCE_DRY_RUN=false in .env AND dry_run: false\n"
        "  in config/robo.yaml — both are required to place real orders."
    )


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    sys.exit(app())
