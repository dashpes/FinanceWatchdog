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
    get_filled_robo_orders,
    get_recent_robo_runs,
    get_robo_orders_for_run,
    get_unfilled_placed_orders,
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
from .learning_models import (
    LEARNING_KIND_ACCURACY_MODIFIER,
    LEARNING_KIND_OUTCOME,
    LEARNING_KIND_SHADOW_OUTCOME,
    LEARNING_KIND_WEIGHT_ADAPTATION,
    LearningEvent,
)
from .learning_operations import (
    accuracy_stats_for_symbol,
    get_outcome_symbols,
    get_recent_outcomes,
    outcome_exists_for_date,
    outcome_metrics,
    record_learning_event,
    record_thesis_outcome,
)
from .shadow_models import (
    SHADOW_SOURCE_CONFLUENCE,
    SHADOW_SOURCE_DISCOVERY,
    SHADOW_SOURCE_GATE,
    SHADOW_STATUS_CLOSED,
    SHADOW_STATUS_OPEN,
    ShadowEntry,
)
from .shadow_operations import (
    close_shadow_entry,
    get_open_shadow_entries,
    get_shadow_entries,
    has_open_shadow,
    mark_shadow_entry,
    record_shadow_entry,
    shadow_ref_ids,
    shadow_summary,
)
from .insight_models import (
    FINDING_INSIDER_CLUSTER,
    FINDING_MULTI_SOURCE,
    ConfluenceFinding,
)
from .insight_operations import (
    finding_exists_for_date,
    get_recent_findings,
    save_finding,
)
from .retention import RetentionConfig, prune_old_data

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
    "get_filled_robo_orders",
    "get_unfilled_placed_orders",
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
    # Learning / feedback ledger (Phase 6)
    "LearningEvent",
    "LEARNING_KIND_OUTCOME",
    "LEARNING_KIND_ACCURACY_MODIFIER",
    "LEARNING_KIND_SHADOW_OUTCOME",
    "LEARNING_KIND_WEIGHT_ADAPTATION",
    "record_learning_event",
    "record_thesis_outcome",
    "get_recent_outcomes",
    "get_outcome_symbols",
    "outcome_exists_for_date",
    "accuracy_stats_for_symbol",
    "outcome_metrics",
    # Shadow ledger (considered-but-not-traded theses)
    "ShadowEntry",
    "SHADOW_SOURCE_CONFLUENCE",
    "SHADOW_SOURCE_DISCOVERY",
    "SHADOW_SOURCE_GATE",
    "SHADOW_STATUS_OPEN",
    "SHADOW_STATUS_CLOSED",
    "record_shadow_entry",
    "has_open_shadow",
    "shadow_ref_ids",
    "get_open_shadow_entries",
    "get_shadow_entries",
    "mark_shadow_entry",
    "close_shadow_entry",
    "shadow_summary",
    # Confluence / insight engine
    "ConfluenceFinding",
    "FINDING_INSIDER_CLUSTER",
    "FINDING_MULTI_SOURCE",
    "save_finding",
    "finding_exists_for_date",
    "get_recent_findings",
    # Retention / cleanup
    "RetentionConfig",
    "prune_old_data",
    # Semantic memory (Phase 5)
    "MemoryEmbedding",
    "save_embedding",
    "search_similar",
    "is_duplicate",
    "cosine_similarity",
]
