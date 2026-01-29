"""Logging configuration using loguru."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> None:
    """Configure loguru for console and file output."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler - concise format
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # File handler - detailed format with rotation
    logger.add(
        log_path / "monitor.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

    logger.info("Logging configured", log_dir=str(log_path), level=log_level)
