"""Portfolio and holding models."""

from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, computed_field


class Holding(BaseModel):
    """A single stock holding."""

    ticker: str = Field(..., pattern=r"^[A-Z]{1,5}$")
    shares: Decimal = Field(..., gt=0)
    cost_basis: Decimal = Field(..., gt=0)
    thesis: str = Field(default="", max_length=500)
    account: str = Field(default="", max_length=50)

    @computed_field
    @property
    def total_cost(self) -> Decimal:
        """Total cost basis for this holding."""
        return self.shares * self.cost_basis


class WatchlistItem(BaseModel):
    """A stock on the watchlist."""

    ticker: str = Field(..., pattern=r"^[A-Z]{1,5}$")
    reason: str = Field(default="")
    target_price: Decimal | None = None


class Portfolio(BaseModel):
    """Portfolio configuration with holdings and watchlist."""

    holdings: list[Holding] = Field(default_factory=list)
    watchlist: list[WatchlistItem] = Field(default_factory=list)

    @computed_field
    @property
    def all_tickers(self) -> list[str]:
        """All tickers to monitor (holdings + watchlist, deduplicated)."""
        tickers = set()
        tickers.update(h.ticker for h in self.holdings)
        tickers.update(w.ticker for w in self.watchlist)
        return sorted(tickers)

    @computed_field
    @property
    def holding_tickers(self) -> list[str]:
        """Just the holding tickers."""
        return [h.ticker for h in self.holdings]

    def get_holding(self, ticker: str) -> Holding | None:
        """Get a holding by ticker."""
        for h in self.holdings:
            if h.ticker == ticker:
                return h
        return None

    def get_thesis(self, ticker: str) -> str | None:
        """Get investment thesis for a ticker."""
        holding = self.get_holding(ticker)
        if holding:
            return holding.thesis if holding.thesis else None
        return None

    def get_cost_basis(self, ticker: str) -> Decimal | None:
        """Get cost basis for a ticker."""
        holding = self.get_holding(ticker)
        return holding.cost_basis if holding else None

    @classmethod
    def from_yaml(cls, path: Path) -> "Portfolio":
        """Load portfolio from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_dict(self) -> dict:
        """Serialize to a plain, YAML-friendly dict (no computed fields/Decimals).

        Optional empty fields (thesis, account, reason, target_price) are omitted
        to keep the written file clean.
        """
        holdings = []
        for h in self.holdings:
            entry: dict = {
                "ticker": h.ticker,
                "shares": float(h.shares),
                "cost_basis": float(h.cost_basis),
            }
            if h.thesis:
                entry["thesis"] = h.thesis
            if h.account:
                entry["account"] = h.account
            holdings.append(entry)

        watchlist = []
        for w in self.watchlist:
            entry = {"ticker": w.ticker}
            if w.reason:
                entry["reason"] = w.reason
            if w.target_price is not None:
                entry["target_price"] = float(w.target_price)
            watchlist.append(entry)

        return {"holdings": holdings, "watchlist": watchlist}

    def to_yaml(self, path: Path) -> None:
        """Write the portfolio to a YAML file (counterpart to from_yaml)."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
