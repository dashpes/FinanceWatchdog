"""Universe collector for discovering stock candidates from indices and ETFs."""

from datetime import datetime

import httpx
import yfinance as yf
from bs4 import BeautifulSoup
from loguru import logger
from sqlalchemy.orm import Session

from ..config import Settings
from ..storage import StockCandidate, get_candidate_by_ticker, save_candidate
from .base import BaseCollector, CollectorResult


class UniverseCollector(BaseCollector):
    """
    Collector for discovering stock universe from indices and ETF holdings.

    Gathers stock tickers from:
    - S&P 500 (Wikipedia table)
    - NASDAQ 100 (Wikipedia table)
    - Custom ETF holdings (yfinance)

    Saves discovered tickers as StockCandidate records with status "discovered".
    """

    name = "universe"
    rate_limit_calls = 10  # Conservative for Wikipedia
    rate_limit_period = 60
    max_retries = 3
    retry_delay = 2.0

    # Wikipedia URLs for index constituents
    SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

    def __init__(
        self,
        session: Session,
        config: Settings,
        collect_sp500: bool = True,
        collect_nasdaq100: bool = True,
    ):
        """
        Initialize the universe collector.

        Args:
            session: SQLAlchemy database session
            config: Application settings
            collect_sp500: Whether to collect S&P 500 constituents
            collect_nasdaq100: Whether to collect NASDAQ 100 constituents
        """
        super().__init__(session, config)
        self._collect_sp500 = collect_sp500
        self._collect_nasdaq100 = collect_nasdaq100

    async def _fetch_url(self, url: str) -> str:
        """
        Fetch content from a URL with proper headers.

        Args:
            url: URL to fetch

        Returns:
            HTML content as string

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        # Wikipedia's User-Agent policy BLOCKS spoofed/generic browser agents (the old
        # fake-Chrome string here got 403'd). Send a descriptive UA that identifies the
        # app and a real contact, per https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy.
        contact = (self.config.sec_contact_email or "").strip() or "contact@financewatchdog.app"
        headers = {
            "User-Agent": f"FinanceWatchdog/1.0 (https://github.com/dashpes/FinanceWatchdog; {contact})",
            "Accept": "text/html,application/xhtml+xml",
        }
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text

    async def collect_sp500(self) -> list[str]:
        """
        Fetch S&P 500 constituents from Wikipedia.

        Parses the HTML table from the Wikipedia page to extract
        ticker symbols.

        Returns:
            List of ticker symbols
        """

        async def _fetch_sp500() -> list[str]:
            html = await self._fetch_url(self.SP500_URL)
            soup = BeautifulSoup(html, "lxml")

            # Find the constituents table (first table with id="constituents")
            table = soup.find("table", {"id": "constituents"})
            if not table:
                # Fallback: find first wikitable
                table = soup.find("table", {"class": "wikitable"})

            if not table:
                logger.warning("Could not find S&P 500 constituents table")
                return []

            tickers = []
            rows = table.find_all("tr")[1:]  # Skip header row

            for row in rows:
                cells = row.find_all("td")
                if cells:
                    # Symbol is typically in the first column
                    ticker_cell = cells[0]
                    ticker = ticker_cell.get_text(strip=True)
                    # Clean the ticker (remove any trailing characters)
                    ticker = ticker.replace(".", "-")  # BRK.B -> BRK-B
                    if ticker:
                        tickers.append(ticker)

            logger.info(f"Collected {len(tickers)} tickers from S&P 500")
            return tickers

        return await self._retry_with_backoff(_fetch_sp500)

    async def collect_nasdaq100(self) -> list[str]:
        """
        Fetch NASDAQ 100 constituents from Wikipedia.

        Parses the HTML table from the Wikipedia page to extract
        ticker symbols.

        Returns:
            List of ticker symbols
        """

        async def _fetch_nasdaq100() -> list[str]:
            html = await self._fetch_url(self.NASDAQ100_URL)
            soup = BeautifulSoup(html, "lxml")

            # Find the constituents table
            # Look for table with "Components" in nearby header
            tables = soup.find_all("table", {"class": "wikitable"})

            tickers = []

            for table in tables:
                # Check if this table has ticker symbols
                rows = table.find_all("tr")[1:]  # Skip header

                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        # NASDAQ 100 table typically has Company, Ticker columns
                        # Ticker is often in the second column
                        for i, cell in enumerate(cells):
                            text = cell.get_text(strip=True)
                            # Check if this looks like a ticker (all caps, 1-5 chars)
                            if (
                                text
                                and text.isupper()
                                and 1 <= len(text) <= 5
                                and text.isalpha()
                            ):
                                tickers.append(text)
                                break

                # If we found enough tickers, this is probably the right table
                if len(tickers) >= 90:  # NASDAQ 100 should have ~100 stocks
                    break
                else:
                    tickers = []  # Reset and try next table

            # If we didn't find enough in the expected format, try alternative parsing
            if len(tickers) < 90:
                tickers = self._parse_nasdaq100_alternative(soup)

            logger.info(f"Collected {len(tickers)} tickers from NASDAQ 100")
            return tickers

        return await self._retry_with_backoff(_fetch_nasdaq100)

    def _parse_nasdaq100_alternative(self, soup: BeautifulSoup) -> list[str]:
        """
        Alternative parsing method for NASDAQ 100 table.

        Args:
            soup: BeautifulSoup object of the page

        Returns:
            List of ticker symbols
        """
        tickers = []
        tables = soup.find_all("table", {"class": "wikitable"})

        for table in tables:
            # Get header to determine column structure
            header_row = table.find("tr")
            if not header_row:
                continue

            headers = [th.get_text(strip=True).lower() for th in header_row.find_all(["th", "td"])]

            # Find ticker/symbol column index
            ticker_col = None
            for i, header in enumerate(headers):
                if "ticker" in header or "symbol" in header:
                    ticker_col = i
                    break

            if ticker_col is None:
                continue

            # Extract tickers from the identified column
            rows = table.find_all("tr")[1:]
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) > ticker_col:
                    ticker = cells[ticker_col].get_text(strip=True)
                    # Clean and validate ticker
                    ticker = ticker.replace(".", "-")
                    if ticker and len(ticker) <= 5:
                        tickers.append(ticker)

            if len(tickers) >= 90:
                break
            else:
                tickers = []

        return tickers

    async def collect_etf_holdings(self, etf_ticker: str) -> list[str]:
        """
        Fetch ETF holdings using yfinance.

        Args:
            etf_ticker: ETF ticker symbol (e.g., "QQQ", "SPY", "VTI")

        Returns:
            List of holding ticker symbols
        """

        def _fetch_etf_holdings() -> list[str]:
            try:
                etf = yf.Ticker(etf_ticker)

                # Try to get holdings from fund_holding_info or similar
                # yfinance API varies, try multiple approaches
                holdings = []

                # Method 1: Try get_holdings() if available
                if hasattr(etf, "get_holdings"):
                    try:
                        holdings_data = etf.get_holdings()
                        if holdings_data is not None and not holdings_data.empty:
                            holdings = holdings_data.index.tolist()
                    except Exception as e:
                        logger.debug(f"Method 1 (get_holdings) failed for {etf_ticker}: {e}")

                # Method 2: Try funds_data
                if not holdings:
                    try:
                        funds_data = etf.funds_data
                        if funds_data and hasattr(funds_data, "top_holdings"):
                            top_holdings = funds_data.top_holdings
                            if top_holdings is not None:
                                holdings = list(top_holdings.index)
                    except Exception as e:
                        logger.debug(f"Method 2 (funds_data) failed for {etf_ticker}: {e}")

                # Method 3: Try institutional_holders as last resort
                if not holdings:
                    try:
                        inst_holders = etf.institutional_holders
                        if inst_holders is not None and not inst_holders.empty:
                            if "Holder" in inst_holders.columns:
                                # This is typically institution names, not stock tickers
                                pass
                    except Exception as e:
                        logger.debug(f"Method 3 (institutional_holders) failed for {etf_ticker}: {e}")

                # Clean tickers
                cleaned_holdings = []
                for h in holdings:
                    if isinstance(h, str):
                        ticker = h.strip().upper()
                        # Basic validation
                        if ticker and len(ticker) <= 10:
                            cleaned_holdings.append(ticker)

                logger.info(
                    f"Collected {len(cleaned_holdings)} holdings from ETF {etf_ticker}"
                )
                return cleaned_holdings

            except Exception as e:
                logger.warning(f"Failed to fetch holdings for {etf_ticker}: {e}")
                return []

        return await self._retry_with_backoff(_fetch_etf_holdings)

    def _deduplicate_tickers(self, tickers: list[str]) -> list[str]:
        """
        Deduplicate and clean a list of tickers.

        Args:
            tickers: List of ticker symbols (may contain duplicates)

        Returns:
            Deduplicated list of ticker symbols, preserving order of first occurrence
        """
        seen = set()
        deduplicated = []

        for ticker in tickers:
            # Normalize ticker
            clean_ticker = ticker.strip().upper()

            # Skip empty or invalid tickers
            if not clean_ticker:
                continue

            # Skip if already seen
            if clean_ticker in seen:
                continue

            seen.add(clean_ticker)
            deduplicated.append(clean_ticker)

        return deduplicated

    def _save_candidates(
        self, tickers: list[str], source: str
    ) -> tuple[int, list[str]]:
        """
        Save tickers as StockCandidate records, avoiding duplicates.

        Args:
            tickers: List of ticker symbols to save
            source: Discovery source (e.g., "sp500", "nasdaq100", "etf:QQQ")

        Returns:
            Tuple of (records_saved, errors)
        """
        records_saved = 0
        errors = []

        for ticker in tickers:
            try:
                # Check if candidate already exists
                existing = get_candidate_by_ticker(self.session, ticker)

                if existing:
                    logger.debug(f"Ticker {ticker} already exists, skipping")
                    continue

                # Create new candidate
                candidate = StockCandidate(
                    ticker=ticker,
                    discovery_source=source,
                    status="discovered",
                )
                save_candidate(self.session, candidate)
                records_saved += 1
                logger.debug(f"Saved new candidate: {ticker} from {source}")

            except Exception as e:
                error_msg = f"Failed to save {ticker}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        return records_saved, errors

    async def collect(self, tickers: list[str]) -> CollectorResult:
        """
        Main collection method.

        Collects stocks from configured indices and any provided ETF tickers.

        Args:
            tickers: Optional list of ETF tickers to analyze for holdings

        Returns:
            CollectorResult with collection statistics
        """
        started_at = datetime.now()
        all_tickers: list[str] = []
        errors: list[str] = []
        records_collected = 0

        # Collect from S&P 500
        if self._collect_sp500:
            try:
                sp500_tickers = await self.collect_sp500()
                records, save_errors = self._save_candidates(sp500_tickers, "sp500")
                records_collected += records
                errors.extend(save_errors)
                all_tickers.extend(sp500_tickers)
            except Exception as e:
                error_msg = f"S&P 500 collection failed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Collect from NASDAQ 100
        if self._collect_nasdaq100:
            try:
                nasdaq_tickers = await self.collect_nasdaq100()
                records, save_errors = self._save_candidates(nasdaq_tickers, "nasdaq100")
                records_collected += records
                errors.extend(save_errors)
                all_tickers.extend(nasdaq_tickers)
            except Exception as e:
                error_msg = f"NASDAQ 100 collection failed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Collect from provided ETF tickers
        for etf_ticker in tickers:
            try:
                etf_holdings = await self.collect_etf_holdings(etf_ticker)
                source = f"etf:{etf_ticker}"
                records, save_errors = self._save_candidates(etf_holdings, source)
                records_collected += records
                errors.extend(save_errors)
                all_tickers.extend(etf_holdings)
            except Exception as e:
                error_msg = f"ETF {etf_ticker} collection failed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Commit all changes
        try:
            self.session.commit()
        except Exception as e:
            error_msg = f"Failed to commit changes: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

        # Log summary
        unique_tickers = len(self._deduplicate_tickers(all_tickers))
        logger.info(
            f"Universe collection complete: {records_collected} new records "
            f"from {unique_tickers} unique tickers"
        )

        finished_at = datetime.now()

        return CollectorResult(
            collector_name=self.name,
            success=len(errors) == 0,
            records_collected=records_collected,
            errors=errors,
            started_at=started_at,
            finished_at=finished_at,
        )

    async def collect_single(self, ticker: str) -> int:
        """
        Process a single ETF ticker for its holdings.

        This is primarily used for collecting holdings from a single ETF.

        Args:
            ticker: ETF ticker symbol

        Returns:
            Number of records saved
        """
        try:
            etf_holdings = await self.collect_etf_holdings(ticker)
            source = f"etf:{ticker}"
            records_saved, _ = self._save_candidates(etf_holdings, source)
            self.session.commit()
            return records_saved
        except Exception as e:
            logger.error(f"Failed to collect holdings for ETF {ticker}: {e}")
            raise
