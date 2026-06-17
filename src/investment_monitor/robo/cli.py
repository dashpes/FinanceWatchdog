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

import sys
from pathlib import Path

import typer

from investment_monitor.config import get_settings
from investment_monitor.robo.broker import BrokerError, PublicBroker
from investment_monitor.robo.config import RoboConfig
from investment_monitor.robo.rebalance import rebalance_run
from investment_monitor.storage import (
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
    return RoboConfig.from_yaml(Path(cfg_dir) / "robo.yaml")


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
            typer.echo(f"  {p.symbol:<6} {p.quantity} @ ${p.price} = ${p.market_value}")
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
        typer.secho(f"ERROR: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

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
                model=ModelRouter(settings).get_model("synthesis"),
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
                ollama_model=ModelRouter(settings).get_model("scoring"),
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

    # 2. Maintain existing theses (deterministic invalidation, then LLM re-eval).
    if not skip_maintenance:
        actions = {"invalidated": 0, "updated": 0, "unchanged": 0}
        with get_session() as session:
            for thesis in get_active_theses(session, acct):
                actions[evaluator.evaluate(session, thesis, account_id=acct)] += 1
        typer.echo(
            f"Thesis maintenance: {actions['updated']} updated, "
            f"{actions['invalidated']} invalidated, {actions['unchanged']} unchanged"
        )

    # 3. Recompute sized target weights from current convictions.
    with get_session() as session:
        refresh_target_weights(session, auto_cfg, account_id=acct)

    try:
        result = rebalance_run(auto_cfg, settings, dry_run_override=(True if dry_run else None))
    except BrokerError as exc:
        typer.secho(f"ERROR: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

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
                typer.echo(
                    f"  {o.side:<4} {o.symbol:<6} {size:<10} {state:<16} "
                    f"{o.gate_code or ''} {o.gate_reason or ''}"
                )
            return

        runs = get_recent_robo_runs(session, limit=limit)
        if not runs:
            typer.echo("No robo runs recorded yet.")
            return
        typer.echo(f"{'started':<20} {'mode':<8} {'status':<10} prop/acc/rej/placed  run_id")
        for r in runs:
            mode = "dry-run" if r.dry_run else "LIVE"
            started = r.started_at.strftime("%Y-%m-%d %H:%M") if r.started_at else "?"
            counts = f"{r.num_proposed}/{r.num_accepted}/{r.num_rejected}/{r.num_placed}"
            typer.echo(f"{started:<20} {mode:<8} {r.status:<10} {counts:<20} {r.run_id}")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    sys.exit(app())
