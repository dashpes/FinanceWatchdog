"""Research module for stock discovery and analysis."""

from .discovery import DiscoveryPipeline, DiscoveryResult
from .orchestrator import ResearchOrchestrator, ResearchResult, SIMULATION_SCORE_THRESHOLD
from .performance import PerformanceAnalyzer
from .queue import ResearchQueue
from .watchlist_sync import WatchlistSync

__all__ = [
    "DiscoveryPipeline",
    "DiscoveryResult",
    "PerformanceAnalyzer",
    "ResearchOrchestrator",
    "ResearchQueue",
    "ResearchResult",
    "SIMULATION_SCORE_THRESHOLD",
    "WatchlistSync",
]
