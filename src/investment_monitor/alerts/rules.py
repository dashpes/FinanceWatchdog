"""Alert rules for detecting notable market conditions.

Each rule function takes a database session, portfolio, and configuration,
then returns a list of AlertMessage objects for any triggered conditions.
"""

from collections import Counter
from datetime import date, datetime, timedelta

from sqlalchemy.orm import Session

from investment_monitor.models import (
    AlertsConfig,
    EarningsAlertSettings,
    InsiderAlertSettings,
    NewsAlertSettings,
    Portfolio,
    PriceAlertSettings,
    VolumeAlertSettings,
)
from investment_monitor.notifications import AlertMessage, Priority
from investment_monitor.storage import (
    alert_exists_by_dedup_key,
    get_insider_transactions,
    get_prices,
    get_recent_news,
    get_upcoming_earnings,
)


def _make_dedup_key(alert_type: str, ticker: str, detail: str) -> str:
    """Create a deduplication key for an alert.

    Format: {alert_type}:{ticker}:{detail}:{date}
    The date ensures alerts can be re-triggered on different days.
    """
    today = date.today().isoformat()
    return f"{alert_type}:{ticker}:{detail}:{today}"


def check_price_alerts(
    session: Session,
    portfolio: Portfolio,
    config: PriceAlertSettings,
) -> list[AlertMessage]:
    """Check price-based alert conditions.

    Checks:
    - Daily price drop > threshold
    - Daily price rise > threshold
    - Weekly price drop > threshold
    - Price below cost basis (for holdings only)

    Args:
        session: Database session
        portfolio: Portfolio with holdings and watchlist
        config: Price alert configuration

    Returns:
        List of triggered AlertMessage objects
    """
    alerts: list[AlertMessage] = []

    for ticker in portfolio.all_tickers:
        # Get recent prices (need at least 7 days for weekly check)
        prices = get_prices(session, ticker, days=10)

        if not prices:
            continue

        # Prices are ordered descending by date
        latest = prices[0]

        # Daily change check (need at least 2 days)
        if len(prices) >= 2:
            prev = prices[1]
            if prev.close and prev.close > 0:
                daily_change_pct = ((latest.close - prev.close) / prev.close) * 100

                # Daily drop alert
                if daily_change_pct <= -config.daily_drop_pct:
                    dedup_key = _make_dedup_key("price_daily_drop", ticker, f"{abs(daily_change_pct):.1f}")
                    if not alert_exists_by_dedup_key(session, dedup_key):
                        alerts.append(AlertMessage(
                            title=f"{ticker} dropped {abs(daily_change_pct):.1f}% today",
                            body=(
                                f"{ticker} fell from ${prev.close:.2f} to ${latest.close:.2f} "
                                f"({daily_change_pct:.1f}%) on {latest.date}."
                            ),
                            ticker=ticker,
                            alert_type="price_daily_drop",
                            priority=Priority.HIGH if abs(daily_change_pct) >= config.daily_drop_pct * 2 else Priority.MEDIUM,
                        ))

                # Daily rise alert
                if daily_change_pct >= config.daily_rise_pct:
                    dedup_key = _make_dedup_key("price_daily_rise", ticker, f"{daily_change_pct:.1f}")
                    if not alert_exists_by_dedup_key(session, dedup_key):
                        alerts.append(AlertMessage(
                            title=f"{ticker} rose {daily_change_pct:.1f}% today",
                            body=(
                                f"{ticker} rose from ${prev.close:.2f} to ${latest.close:.2f} "
                                f"(+{daily_change_pct:.1f}%) on {latest.date}."
                            ),
                            ticker=ticker,
                            alert_type="price_daily_rise",
                            priority=Priority.MEDIUM,
                        ))

        # Weekly change check (need at least 6 days of data)
        if len(prices) >= 6:
            # Find price from approximately 5 trading days ago
            week_ago = prices[5] if len(prices) > 5 else prices[-1]
            if week_ago.close and week_ago.close > 0:
                weekly_change_pct = ((latest.close - week_ago.close) / week_ago.close) * 100

                if weekly_change_pct <= -config.weekly_drop_pct:
                    dedup_key = _make_dedup_key("price_weekly_drop", ticker, f"{abs(weekly_change_pct):.1f}")
                    if not alert_exists_by_dedup_key(session, dedup_key):
                        alerts.append(AlertMessage(
                            title=f"{ticker} down {abs(weekly_change_pct):.1f}% this week",
                            body=(
                                f"{ticker} fell from ${week_ago.close:.2f} to ${latest.close:.2f} "
                                f"({weekly_change_pct:.1f}%) over the past week."
                            ),
                            ticker=ticker,
                            alert_type="price_weekly_drop",
                            priority=Priority.MEDIUM,
                        ))

        # Below cost basis check (holdings only)
        if config.below_cost_basis:
            cost_basis = portfolio.get_cost_basis(ticker)
            if cost_basis is not None:
                cost_basis_float = float(cost_basis)
                if latest.close < cost_basis_float:
                    pct_below = ((cost_basis_float - latest.close) / cost_basis_float) * 100
                    dedup_key = _make_dedup_key("price_below_cost", ticker, "below")
                    if not alert_exists_by_dedup_key(session, dedup_key):
                        alerts.append(AlertMessage(
                            title=f"{ticker} trading below cost basis",
                            body=(
                                f"{ticker} at ${latest.close:.2f} is {pct_below:.1f}% below "
                                f"your cost basis of ${cost_basis_float:.2f}."
                            ),
                            ticker=ticker,
                            alert_type="price_below_cost",
                            priority=Priority.LOW,
                        ))

    return alerts


def check_volume_alerts(
    session: Session,
    portfolio: Portfolio,
    config: VolumeAlertSettings,
) -> list[AlertMessage]:
    """Check for unusual trading volume.

    Triggers when today's volume exceeds N times the lookback period average.

    Args:
        session: Database session
        portfolio: Portfolio with holdings and watchlist
        config: Volume alert configuration

    Returns:
        List of triggered AlertMessage objects
    """
    alerts: list[AlertMessage] = []

    for ticker in portfolio.all_tickers:
        prices = get_prices(session, ticker, days=config.lookback_days + 5)

        if len(prices) < 2:
            continue

        latest = prices[0]
        if latest.volume is None or latest.volume == 0:
            continue

        # Calculate average volume from historical data (excluding latest)
        historical = [p for p in prices[1:] if p.volume and p.volume > 0]
        if len(historical) < config.lookback_days // 2:
            # Not enough data for meaningful average
            continue

        avg_volume = sum(p.volume for p in historical[:config.lookback_days]) / len(historical[:config.lookback_days])

        if avg_volume > 0:
            volume_multiple = latest.volume / avg_volume

            if volume_multiple >= config.multiplier:
                dedup_key = _make_dedup_key("volume_spike", ticker, f"{volume_multiple:.1f}x")
                if not alert_exists_by_dedup_key(session, dedup_key):
                    alerts.append(AlertMessage(
                        title=f"{ticker} volume {volume_multiple:.1f}x normal",
                        body=(
                            f"{ticker} traded {latest.volume:,} shares today, which is "
                            f"{volume_multiple:.1f}x the {config.lookback_days}-day average "
                            f"of {avg_volume:,.0f} shares."
                        ),
                        ticker=ticker,
                        alert_type="volume_spike",
                        priority=Priority.MEDIUM if volume_multiple < config.multiplier * 2 else Priority.HIGH,
                    ))

    return alerts


def check_insider_alerts(
    session: Session,
    portfolio: Portfolio,
    config: InsiderAlertSettings,
) -> list[AlertMessage]:
    """Check for notable insider trading activity.

    Checks:
    - Insider buys > min value threshold
    - Insider sells > min value threshold
    - CEO/CFO transactions (any size if configured)
    - Cluster buying/selling (multiple insiders)

    Args:
        session: Database session
        portfolio: Portfolio with holdings and watchlist
        config: Insider alert configuration

    Returns:
        List of triggered AlertMessage objects
    """
    alerts: list[AlertMessage] = []

    for ticker in portfolio.all_tickers:
        transactions = get_insider_transactions(session, ticker, days=config.cluster_days + 7)

        if not transactions:
            continue

        # Track recent transactions for cluster detection
        recent_buys: list = []
        recent_sells: list = []

        for txn in transactions:
            is_buy = txn.transaction_type.upper() in ("P", "A", "BUY")
            is_sell = txn.transaction_type.upper() in ("S", "D", "SELL")
            value = txn.total_value or 0

            # Check if this is a CEO/CFO transaction
            is_executive = False
            if txn.owner_title:
                title_upper = txn.owner_title.upper()
                is_executive = any(t in title_upper for t in ("CEO", "CFO", "CHIEF EXECUTIVE", "CHIEF FINANCIAL"))

            # Significant buy alert
            if is_buy and value >= config.min_buy_value:
                dedup_key = _make_dedup_key("insider_buy", ticker, f"{txn.owner_name}:{txn.trade_date}")
                if not alert_exists_by_dedup_key(session, dedup_key):
                    priority = Priority.HIGH if is_executive else Priority.MEDIUM
                    price_str = f"${txn.price_per_share:.2f}" if txn.price_per_share else "N/A"
                    alerts.append(AlertMessage(
                        title=f"Insider buy: {txn.owner_name} bought ${value:,.0f} of {ticker}",
                        body=(
                            f"{txn.owner_name} ({txn.owner_title or 'Insider'}) purchased "
                            f"{txn.shares:,} shares of {ticker} at {price_str} "
                            f"(${value:,.0f} total) on {txn.trade_date}."
                        ),
                        ticker=ticker,
                        alert_type="insider_buy",
                        priority=priority,
                        url=txn.sec_url,
                    ))

            # Significant sell alert
            if is_sell and value >= config.min_sell_value:
                dedup_key = _make_dedup_key("insider_sell", ticker, f"{txn.owner_name}:{txn.trade_date}")
                if not alert_exists_by_dedup_key(session, dedup_key):
                    priority = Priority.HIGH if is_executive else Priority.MEDIUM
                    price_str = f"${txn.price_per_share:.2f}" if txn.price_per_share else "N/A"
                    alerts.append(AlertMessage(
                        title=f"Insider sell: {txn.owner_name} sold ${value:,.0f} of {ticker}",
                        body=(
                            f"{txn.owner_name} ({txn.owner_title or 'Insider'}) sold "
                            f"{txn.shares:,} shares of {ticker} at {price_str} "
                            f"(${value:,.0f} total) on {txn.trade_date}."
                        ),
                        ticker=ticker,
                        alert_type="insider_sell",
                        priority=priority,
                        url=txn.sec_url,
                    ))

            # CEO/CFO any transaction alert (if enabled and not already alerted above)
            if config.alert_ceo_cfo_any and is_executive:
                if (is_buy and value < config.min_buy_value) or (is_sell and value < config.min_sell_value):
                    dedup_key = _make_dedup_key("insider_exec", ticker, f"{txn.owner_name}:{txn.trade_date}")
                    if not alert_exists_by_dedup_key(session, dedup_key):
                        action = "bought" if is_buy else "sold"
                        alerts.append(AlertMessage(
                            title=f"Executive {action}: {txn.owner_name} at {ticker}",
                            body=(
                                f"{txn.owner_name} ({txn.owner_title}) {action} "
                                f"{txn.shares:,} shares of {ticker} (${value:,.0f}) on {txn.trade_date}."
                            ),
                            ticker=ticker,
                            alert_type="insider_executive",
                            priority=Priority.MEDIUM,
                            url=txn.sec_url,
                        ))

            # Track for cluster detection
            cutoff_date = date.today() - timedelta(days=config.cluster_days)
            if txn.trade_date >= cutoff_date:
                if is_buy:
                    recent_buys.append(txn.owner_name)
                elif is_sell:
                    recent_sells.append(txn.owner_name)

        # Cluster buying alert
        unique_buyers = len(set(recent_buys))
        if unique_buyers >= config.cluster_threshold:
            dedup_key = _make_dedup_key("insider_cluster_buy", ticker, f"{unique_buyers}")
            if not alert_exists_by_dedup_key(session, dedup_key):
                alerts.append(AlertMessage(
                    title=f"Cluster buying: {unique_buyers} insiders bought {ticker}",
                    body=(
                        f"{unique_buyers} different insiders have purchased {ticker} stock "
                        f"in the past {config.cluster_days} days. This may signal insider confidence."
                    ),
                    ticker=ticker,
                    alert_type="insider_cluster_buy",
                    priority=Priority.HIGH,
                ))

        # Cluster selling alert
        unique_sellers = len(set(recent_sells))
        if unique_sellers >= config.cluster_threshold:
            dedup_key = _make_dedup_key("insider_cluster_sell", ticker, f"{unique_sellers}")
            if not alert_exists_by_dedup_key(session, dedup_key):
                alerts.append(AlertMessage(
                    title=f"Cluster selling: {unique_sellers} insiders sold {ticker}",
                    body=(
                        f"{unique_sellers} different insiders have sold {ticker} stock "
                        f"in the past {config.cluster_days} days. This warrants attention."
                    ),
                    ticker=ticker,
                    alert_type="insider_cluster_sell",
                    priority=Priority.HIGH,
                ))

    return alerts


def check_earnings_alerts(
    session: Session,
    portfolio: Portfolio,
    config: EarningsAlertSettings,
) -> list[AlertMessage]:
    """Check for upcoming earnings announcements.

    Alerts when a holding's earnings date falls within the lookahead window.

    Args:
        session: Database session
        portfolio: Portfolio with holdings and watchlist
        config: Earnings alert configuration

    Returns:
        List of triggered AlertMessage objects
    """
    alerts: list[AlertMessage] = []

    upcoming = get_upcoming_earnings(session, portfolio.all_tickers, days_ahead=config.lookahead_days)

    for earnings in upcoming:
        days_until = (earnings.earnings_date - date.today()).days

        # Create different dedup keys based on proximity
        if days_until <= 1:
            dedup_key = _make_dedup_key("earnings_imminent", earnings.ticker, "tomorrow")
            priority = Priority.HIGH
            timing = "tomorrow" if days_until == 1 else "today"
        elif days_until <= 3:
            dedup_key = _make_dedup_key("earnings_soon", earnings.ticker, f"{days_until}d")
            priority = Priority.MEDIUM
            timing = f"in {days_until} days"
        else:
            dedup_key = _make_dedup_key("earnings_upcoming", earnings.ticker, f"{days_until}d")
            priority = Priority.LOW
            timing = f"in {days_until} days"

        if not alert_exists_by_dedup_key(session, dedup_key):
            confirmed_text = " (confirmed)" if earnings.confirmed else " (estimated)"
            alerts.append(AlertMessage(
                title=f"{earnings.ticker} earnings {timing}",
                body=(
                    f"{earnings.ticker} is scheduled to report earnings on "
                    f"{earnings.earnings_date}{confirmed_text}. "
                    f"Consider reviewing your position and thesis."
                ),
                ticker=earnings.ticker,
                alert_type="earnings_upcoming",
                priority=priority,
            ))

    return alerts


def check_news_keyword_alerts(
    session: Session,
    portfolio: Portfolio,
    config: NewsAlertSettings,
) -> list[AlertMessage]:
    """Check news headlines for configured keywords.

    Scans recent news for keywords that may indicate important events.

    Args:
        session: Database session
        portfolio: Portfolio with holdings and watchlist
        config: News alert configuration

    Returns:
        List of triggered AlertMessage objects
    """
    alerts: list[AlertMessage] = []

    # Get recent news for all portfolio tickers
    for ticker in portfolio.all_tickers:
        news_items = get_recent_news(session, ticker=ticker, hours=24)

        for item in news_items:
            # Check relevance score if available
            if item.relevance_score is not None and item.relevance_score < config.min_relevance_score:
                continue

            # Check headline for keywords
            headline_lower = item.headline.lower()
            matched_keywords = [kw for kw in config.keywords if kw.lower() in headline_lower]

            if matched_keywords:
                dedup_key = _make_dedup_key("news_keyword", ticker or "general", item.url[:50])
                if not alert_exists_by_dedup_key(session, dedup_key):
                    # Determine priority based on keyword severity
                    high_priority_keywords = {"lawsuit", "sec", "investigation", "fraud", "bankruptcy"}
                    has_high_priority = any(kw.lower() in high_priority_keywords for kw in matched_keywords)

                    alerts.append(AlertMessage(
                        title=f"News alert for {ticker}: {matched_keywords[0]}",
                        body=(
                            f"Headline: {item.headline}\n\n"
                            f"Keywords matched: {', '.join(matched_keywords)}\n"
                            f"Source: {item.source}"
                        ),
                        ticker=ticker,
                        alert_type="news_keyword",
                        priority=Priority.HIGH if has_high_priority else Priority.MEDIUM,
                        url=item.url,
                    ))

    return alerts
