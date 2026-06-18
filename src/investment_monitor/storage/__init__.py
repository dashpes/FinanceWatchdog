"""Database storage module."""

from .database import get_session, init_db
from .models import (
    AlertSent,
    Base,
    EarningsDate,
    ETFHolding,
    InsiderTransaction,
    NewsItem,
    Price,
)
from .operations import (
    alert_exists_by_dedup_key,
    get_etf_holdings,
    get_insider_transactions,
    get_latest_price,
    get_prices,
    get_recent_alerts,
    get_recent_news,
    get_upcoming_earnings,
    get_unscored_news,
    insider_transaction_exists,
    news_exists,
    price_exists,
    save_alert,
    save_earnings_date,
    save_etf_holdings,
    save_insider_transaction,
    save_news_item,
    save_price,
    save_prices,
)
from .research_models import (
    CANDIDATE_STATUSES,
    CandidateScore,
    CongressionalTrade,
    PerformanceTracker,
    ResearchProfile,
    ResearchReport,
    SimulationResult,
    StockCandidate,
)
from .research_operations import (
    get_candidate_by_ticker,
    get_candidates_by_status,
    get_high_scoring_candidates,
    get_latest_report,
    get_latest_score,
    get_or_create_default_profile,
    get_records_needing_update,
    get_score_history,
    get_simulation_results,
    get_top_candidates,
    get_trades_for_ticker,
    save_candidate,
    save_congressional_trade,
    save_performance_record,
    save_profile,
    save_report,
    save_score,
    save_simulation_result,
)
from .robo_models import RoboOrder, RoboRun
from .robo_operations import (
    count_placed_orders_today,
    finalize_robo_run,
    get_recent_robo_runs,
    get_robo_orders_for_run,
    save_robo_order,
    save_robo_run,
)
from .memory_models import MemoryEmbedding
from .memory_operations import (
    cosine_similarity,
    is_duplicate,
    save_embedding,
    search_similar,
)
from .thesis_models import LIVE_THESIS_STATUSES, Thesis, ThesisStatus
from .thesis_operations import (
    get_active_symbols,
    get_active_theses,
    get_all_theses,
    get_thesis,
    invalidate_thesis,
    record_conviction_update,
    save_thesis,
    set_target_weight,
)

__all__ = [
    # Database
    "init_db",
    "get_session",
    "Base",
    # Core models
    "Price",
    "InsiderTransaction",
    "NewsItem",
    "AlertSent",
    "EarningsDate",
    "ETFHolding",
    # Core operations
    "save_price",
    "save_prices",
    "get_latest_price",
    "get_prices",
    "price_exists",
    "save_insider_transaction",
    "get_insider_transactions",
    "insider_transaction_exists",
    "save_news_item",
    "news_exists",
    "get_unscored_news",
    "get_recent_news",
    "save_alert",
    "get_recent_alerts",
    "alert_exists_by_dedup_key",
    "save_earnings_date",
    "get_upcoming_earnings",
    "save_etf_holdings",
    "get_etf_holdings",
    # Research models
    "CANDIDATE_STATUSES",
    "ResearchProfile",
    "StockCandidate",
    "CandidateScore",
    "ResearchReport",
    "PerformanceTracker",
    "CongressionalTrade",
    "SimulationResult",
    # Research operations - Profile
    "get_or_create_default_profile",
    "save_profile",
    # Research operations - Candidate
    "save_candidate",
    "get_candidate_by_ticker",
    "get_candidates_by_status",
    "get_top_candidates",
    # Research operations - Score
    "save_score",
    "get_latest_score",
    "get_score_history",
    # Research operations - Report
    "save_report",
    "get_latest_report",
    # Research operations - Performance
    "save_performance_record",
    "get_records_needing_update",
    # Research operations - Congressional Trade
    "save_congressional_trade",
    "get_trades_for_ticker",
    # Research operations - Simulation
    "save_simulation_result",
    "get_simulation_results",
    "get_high_scoring_candidates",
    # Robo advisor models
    "RoboRun",
    "RoboOrder",
    # Robo advisor operations
    "save_robo_run",
    "finalize_robo_run",
    "save_robo_order",
    "get_recent_robo_runs",
    "get_robo_orders_for_run",
    "count_placed_orders_today",
    # Thesis store (autonomous investor)
    "Thesis",
    "ThesisStatus",
    "LIVE_THESIS_STATUSES",
    "get_active_theses",
    "get_all_theses",
    "get_thesis",
    "save_thesis",
    "record_conviction_update",
    "set_target_weight",
    "invalidate_thesis",
    "get_active_symbols",
    # Semantic memory (Phase 5)
    "MemoryEmbedding",
    "save_embedding",
    "search_similar",
    "is_duplicate",
    "cosine_similarity",
]
