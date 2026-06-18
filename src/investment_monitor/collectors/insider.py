"""Insider Transaction Collector for SEC Form 4 filings."""

import re
from datetime import date, datetime, timedelta

import httpx
from bs4 import BeautifulSoup
from loguru import logger

from .base import BaseCollector, CollectorResult
from ..storage import InsiderTransaction, insider_transaction_exists, save_insider_transaction


class InsiderCollector(BaseCollector):
    """
    Collector for SEC Form 4 insider transaction filings.

    Fetches Form 4 filings from SEC EDGAR to track insider buying and selling.
    SEC requests max 10 requests per second, so we set rate limiting accordingly.
    """

    name = "insider"
    rate_limit_calls = 10  # SEC asks for max 10 requests/second
    rate_limit_period = 1  # 1 second window

    SEC_SEARCH_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
    SEC_CIK_LOOKUP_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
    SEC_BASE_URL = "https://www.sec.gov"

    # User-Agent required by SEC - they block requests without a proper UA.
    # SEC asks for a real contact; broad ingestion makes many requests, so include one.
    USER_AGENT = "FinanceWatchdog/1.0 (daniel.ashpes@gmail.com)"
    EDGAR_ARCHIVES = "https://www.sec.gov/Archives"

    async def collect(self, tickers: list[str]) -> CollectorResult:
        """
        Collect Form 4 insider transaction data for given tickers.

        Args:
            tickers: List of ticker symbols to collect insider transactions for

        Returns:
            CollectorResult with success status, records count, and any errors
        """
        started_at = datetime.now()
        records = 0
        errors: list[str] = []

        for ticker in tickers:
            try:
                count = await self._retry_with_backoff(self.collect_single, ticker)
                records += count
                logger.debug(f"{self.name}: Collected {count} transactions for {ticker}")
            except Exception as e:
                error_msg = f"{ticker}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"{self.name}: Failed to collect for {ticker}: {e}")

        finished_at = datetime.now()

        return CollectorResult(
            collector_name=self.name,
            success=len(errors) == 0,
            records_collected=records,
            errors=errors,
            started_at=started_at,
            finished_at=finished_at,
        )

    async def collect_single(self, ticker: str) -> int:
        """
        Collect Form 4 filings for a single ticker.

        1. Look up CIK for the ticker
        2. Search SEC EDGAR for recent Form 4 filings
        3. For each new filing, parse details and extract transactions
        4. Save to database with deduplication by sec_url

        Args:
            ticker: Ticker symbol to collect insider transactions for

        Returns:
            Number of records saved
        """
        # Get CIK for ticker
        cik = await self._get_cik_for_ticker(ticker)
        if not cik:
            logger.debug(f"{self.name}: No CIK found for {ticker}")
            return 0

        # Get Form 4 filings list
        filings = await self._get_form4_filings(cik)
        if not filings:
            logger.debug(f"{self.name}: No Form 4 filings found for {ticker} (CIK: {cik})")
            return 0

        records_saved = 0
        for filing_url in filings:
            # Check if we already have this filing
            if insider_transaction_exists(self.session, filing_url):
                logger.debug(f"{self.name}: Skipping existing filing: {filing_url}")
                continue

            try:
                # Fetch and parse the filing
                transactions = await self._fetch_and_parse_filing(filing_url, ticker)

                for txn in transactions:
                    # Double-check deduplication
                    if not insider_transaction_exists(self.session, txn.sec_url):
                        save_insider_transaction(self.session, txn)
                        records_saved += 1

            except Exception as e:
                logger.warning(f"{self.name}: Failed to parse filing {filing_url}: {e}")
                continue

        # Commit after processing all filings for this ticker
        self.session.commit()
        return records_saved

    # ----------------------------------------------------------------------- #
    # Broad, universe-independent collection (SEC EDGAR daily index)
    # ----------------------------------------------------------------------- #
    async def collect_all(
        self, *, days_back: int = 1, limit: int | None = None
    ) -> CollectorResult:
        """Retain ALL Form 4 insider transactions market-wide (broad collection).

        Iterates the EDGAR *daily index* for the last ``days_back`` business days,
        filters to Form 4, and parses each filing's ownership XML into
        InsiderTransaction rows — across the WHOLE market, not a configured universe
        (the issuer ticker is taken from the filing itself). Deduped by ``sec_url``.

        Args:
            days_back: how many recent business days of indexes to ingest.
            limit: cap on filings to parse this run (safety/testing bound).
        """
        started_at = datetime.now()
        records = 0
        errors: list[str] = []
        seen: set[str] = set()

        async with httpx.AsyncClient(
            headers={"User-Agent": self.USER_AGENT}, timeout=30.0
        ) as client:
            filing_urls: list[str] = []
            for d in self._recent_business_dates(days_back):
                try:
                    idx = await self._get(client, self._daily_index_url(d))
                    filing_urls.extend(self._parse_index_for_form4(idx))
                except Exception as e:  # noqa: BLE001 - a missing/holiday index is fine
                    logger.debug(f"{self.name}: daily index {d} unavailable: {e}")

            if limit is not None:
                filing_urls = filing_urls[:limit]
            logger.info(f"{self.name}: {len(filing_urls)} Form 4 filings to parse (broad)")

            for form_url in filing_urls:
                try:
                    txt = await self._get(client, form_url)
                    xml = self._extract_ownership_xml(txt)
                    if not xml:
                        continue
                    for txn in self._parse_form4(xml, None, form_url):
                        if txn.sec_url in seen:
                            continue
                        seen.add(txn.sec_url)
                        if not insider_transaction_exists(self.session, txn.sec_url):
                            self.session.add(txn)  # commit once below, not per row
                            records += 1
                except Exception as e:  # noqa: BLE001 - one bad filing must not abort the run
                    logger.debug(f"{self.name}: failed filing {form_url}: {e}")

        try:
            self.session.commit()
        except Exception as e:  # noqa: BLE001
            self.session.rollback()
            error_msg = f"Failed to commit broad insider transactions: {str(e)}"
            errors.append(error_msg)
            logger.error(f"{self.name}: {error_msg}")

        return CollectorResult(
            collector_name=self.name,
            success=len(errors) == 0,
            records_collected=records,
            errors=errors,
            started_at=started_at,
            finished_at=datetime.now(),
        )

    async def _get(self, client: httpx.AsyncClient, url: str) -> str:
        """Rate-limited GET (respects SEC's 10 req/s)."""
        await self._rate_limit()
        response = await client.get(url)
        response.raise_for_status()
        return response.text

    def _recent_business_dates(self, days_back: int) -> list[date]:
        """The last ``days_back`` calendar days, weekdays only (newest first)."""
        out: list[date] = []
        today = date.today()
        for i in range(max(1, days_back)):
            day = today - timedelta(days=i)
            if day.weekday() < 5:  # Mon-Fri (markets/EDGAR don't file weekends)
                out.append(day)
        return out

    def _daily_index_url(self, d: date) -> str:
        qtr = (d.month - 1) // 3 + 1
        return f"{self.EDGAR_ARCHIVES}/edgar/daily-index/{d.year}/QTR{qtr}/form.{d:%Y%m%d}.idx"

    def _parse_index_for_form4(self, idx_text: str) -> list[str]:
        """Pull Form 4 submission URLs out of an EDGAR daily ``form.*.idx``.

        Lines are: ``Form Type  Company Name  CIK  Date Filed  File Name``. The last
        whitespace token is the submission path; the first is the form type.
        """
        urls: list[str] = []
        for line in idx_text.splitlines():
            parts = line.split()
            if len(parts) < 5 or parts[0] != "4":  # exact Form 4 (not 4/A amendments)
                continue
            file_name = parts[-1]
            if file_name.startswith("edgar/data/") and file_name.endswith(".txt"):
                urls.append(f"{self.EDGAR_ARCHIVES}/{file_name}")
        return urls

    def _extract_ownership_xml(self, submission_text: str) -> str | None:
        """Extract the ``<ownershipDocument>`` Form 4 XML from a full submission .txt."""
        m = re.search(
            r"<ownershipDocument>.*?</ownershipDocument>",
            submission_text,
            re.DOTALL | re.IGNORECASE,
        )
        return m.group(0) if m else None

    async def _get_cik_for_ticker(self, ticker: str) -> str | None:
        """
        Look up SEC CIK for a ticker symbol.

        Args:
            ticker: Stock ticker symbol

        Returns:
            CIK number as string, or None if not found
        """
        params = {
            "action": "getcompany",
            "company": ticker,
            "type": "4",
            "dateb": "",
            "owner": "include",
            "count": "10",
            "output": "atom",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.SEC_CIK_LOOKUP_URL,
                params=params,
                headers={"User-Agent": self.USER_AGENT},
                timeout=30.0,
            )
            response.raise_for_status()

            # Parse the atom feed to extract CIK
            soup = BeautifulSoup(response.text, "xml")

            # Look for company-info element with CIK
            company_info = soup.find("company-info")
            if company_info:
                cik_elem = company_info.find("cik")
                if cik_elem:
                    return cik_elem.text.strip()

            # Alternative: extract from entry links
            entry = soup.find("entry")
            if entry:
                link = entry.find("link")
                if link and link.get("href"):
                    href = link.get("href", "")
                    # Extract CIK from URL pattern like /cgi-bin/browse-edgar?action=getcompany&CIK=0000320193
                    if "CIK=" in href:
                        cik_part = href.split("CIK=")[1]
                        cik = cik_part.split("&")[0]
                        return cik.lstrip("0") or "0"

            return None

    async def _get_form4_filings(self, cik: str) -> list[str]:
        """
        Get list of recent Form 4 filing URLs for a company.

        Args:
            cik: SEC CIK number

        Returns:
            List of filing URLs
        """
        params = {
            "action": "getcompany",
            "CIK": cik,
            "type": "4",
            "dateb": "",
            "owner": "include",
            "count": "40",  # Get last 40 filings
            "output": "atom",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.SEC_SEARCH_URL,
                params=params,
                headers={"User-Agent": self.USER_AGENT},
                timeout=30.0,
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "xml")
            filing_urls: list[str] = []

            for entry in soup.find_all("entry"):
                # Get the link to the filing
                link = entry.find("link")
                if link and link.get("href"):
                    # This gives us the index page, we need the actual filing
                    index_url = link.get("href", "")
                    if index_url:
                        filing_urls.append(index_url)

            return filing_urls

    async def _fetch_and_parse_filing(
        self, filing_index_url: str, ticker: str
    ) -> list[InsiderTransaction]:
        """
        Fetch and parse a Form 4 filing from SEC EDGAR.

        Args:
            filing_index_url: URL to the filing index page
            ticker: Ticker symbol for the company

        Returns:
            List of InsiderTransaction records from this filing
        """
        async with httpx.AsyncClient() as client:
            # First, get the index page to find the actual XML file
            response = await client.get(
                filing_index_url,
                headers={"User-Agent": self.USER_AGENT},
                timeout=30.0,
            )
            response.raise_for_status()

            # Find the XML file link in the index
            xml_url = self._find_xml_url(response.text, filing_index_url)
            if not xml_url:
                logger.debug(f"{self.name}: No XML file found in {filing_index_url}")
                return []

            # Fetch the XML file
            xml_response = await client.get(
                xml_url,
                headers={"User-Agent": self.USER_AGENT},
                timeout=30.0,
            )
            xml_response.raise_for_status()

            return self._parse_form4(xml_response.text, ticker, xml_url)

    def _find_xml_url(self, index_html: str, base_url: str) -> str | None:
        """
        Find the XML file URL from the filing index page.

        Args:
            index_html: HTML content of the index page
            base_url: Base URL for constructing absolute URLs

        Returns:
            URL to the XML file, or None if not found
        """
        soup = BeautifulSoup(index_html, "html.parser")

        # Look for links to XML files
        for link in soup.find_all("a"):
            href = link.get("href", "")
            if href.endswith(".xml") and "form4" in href.lower():
                # Construct absolute URL
                if href.startswith("/"):
                    return f"{self.SEC_BASE_URL}{href}"
                elif href.startswith("http"):
                    return href
                else:
                    # Relative URL - construct from base
                    base_path = base_url.rsplit("/", 1)[0]
                    return f"{base_path}/{href}"

        # Try finding any XML file
        for link in soup.find_all("a"):
            href = link.get("href", "")
            if href.endswith(".xml"):
                if href.startswith("/"):
                    return f"{self.SEC_BASE_URL}{href}"
                elif href.startswith("http"):
                    return href
                else:
                    base_path = base_url.rsplit("/", 1)[0]
                    return f"{base_path}/{href}"

        return None

    def _parse_form4(
        self, xml_content: str, ticker: str | None, sec_url: str
    ) -> list[InsiderTransaction]:
        """
        Parse Form 4 XML into transaction records.

        Args:
            xml_content: XML content of the Form 4 filing
            ticker: Ticker symbol for the company
            sec_url: URL of the SEC filing

        Returns:
            List of InsiderTransaction records
        """
        transactions: list[InsiderTransaction] = []
        soup = BeautifulSoup(xml_content, "xml")

        # Broad mode passes ticker=None; derive the issuer symbol from the filing itself.
        if not ticker:
            sym = soup.find("issuerTradingSymbol")
            ticker = sym.text.strip().upper() if sym and sym.text else None
            if not ticker:
                return []  # can't attribute a transaction without an issuer ticker

        # Extract owner information
        owner_name = self._extract_owner_name(soup)
        owner_title = self._extract_owner_title(soup)

        # Extract filing date
        filing_date = self._extract_filing_date(soup)

        # Parse non-derivative transactions
        for txn in soup.find_all("nonDerivativeTransaction"):
            transaction = self._parse_transaction_element(
                txn, ticker, owner_name, owner_title, filing_date, sec_url
            )
            if transaction:
                transactions.append(transaction)

        # Parse derivative transactions
        for txn in soup.find_all("derivativeTransaction"):
            transaction = self._parse_transaction_element(
                txn, ticker, owner_name, owner_title, filing_date, sec_url
            )
            if transaction:
                transactions.append(transaction)

        return transactions

    def _extract_owner_name(self, soup: BeautifulSoup) -> str:
        """Extract the owner name from Form 4."""
        # Try reportingOwner > reportingOwnerId > rptOwnerName
        owner_id = soup.find("reportingOwnerId")
        if owner_id:
            name_elem = owner_id.find("rptOwnerName")
            if name_elem:
                return name_elem.text.strip()

        # Fallback
        name_elem = soup.find("rptOwnerName")
        if name_elem:
            return name_elem.text.strip()

        return "Unknown"

    def _extract_owner_title(self, soup: BeautifulSoup) -> str | None:
        """Extract the owner's title/relationship from Form 4."""
        # Try reportingOwner > reportingOwnerRelationship > officerTitle
        relationship = soup.find("reportingOwnerRelationship")
        if relationship:
            title = relationship.find("officerTitle")
            if title and title.text.strip():
                return title.text.strip()

            # Check relationship flags
            if relationship.find("isDirector") and self._get_bool(
                relationship.find("isDirector")
            ):
                return "Director"
            if relationship.find("isOfficer") and self._get_bool(
                relationship.find("isOfficer")
            ):
                return "Officer"
            if relationship.find("isTenPercentOwner") and self._get_bool(
                relationship.find("isTenPercentOwner")
            ):
                return "10% Owner"
            if relationship.find("isOther") and self._get_bool(
                relationship.find("isOther")
            ):
                other_text = relationship.find("otherText")
                if other_text:
                    return other_text.text.strip()

        return None

    def _get_bool(self, element) -> bool:
        """Get boolean value from XML element."""
        if element is None:
            return False
        text = element.text.strip().lower() if element.text else ""
        return text in ("1", "true", "yes")

    def _extract_filing_date(self, soup: BeautifulSoup) -> date:
        """Extract the filing date from Form 4."""
        # Try periodOfReport first (the date of the transaction)
        period = soup.find("periodOfReport")
        if period and period.text:
            try:
                return datetime.strptime(period.text.strip(), "%Y-%m-%d").date()
            except ValueError:
                pass

        # Fallback to current date
        return date.today()

    def _parse_transaction_element(
        self,
        txn_elem,
        ticker: str,
        owner_name: str,
        owner_title: str | None,
        filing_date: date,
        base_sec_url: str,
    ) -> InsiderTransaction | None:
        """
        Parse a single transaction element from Form 4.

        Args:
            txn_elem: BeautifulSoup element for the transaction
            ticker: Ticker symbol
            owner_name: Name of the insider
            owner_title: Title/role of the insider
            filing_date: Date of the filing
            base_sec_url: URL of the SEC filing

        Returns:
            InsiderTransaction or None if parsing fails
        """
        try:
            # Get transaction date
            trade_date = filing_date
            date_elem = txn_elem.find("transactionDate")
            if date_elem:
                value = date_elem.find("value")
                if value and value.text:
                    try:
                        trade_date = datetime.strptime(value.text.strip(), "%Y-%m-%d").date()
                    except ValueError:
                        pass

            # Get transaction type (A=Acquisition/P=Purchase, D=Disposition/S=Sale)
            transaction_type = "P"  # Default to purchase
            coding = txn_elem.find("transactionCoding")
            if coding:
                code = coding.find("transactionCode")
                if code and code.text:
                    code_text = code.text.strip().upper()
                    # P = Purchase, S = Sale, A = Award/Grant, D = Sale (disposition)
                    # M = Exercise of derivative
                    if code_text in ("S", "D"):
                        transaction_type = "S"
                    elif code_text in ("P", "A", "M"):
                        transaction_type = "P"
                    else:
                        transaction_type = code_text

            # Get shares
            shares = 0
            amounts = txn_elem.find("transactionAmounts")
            if amounts:
                shares_elem = amounts.find("transactionShares")
                if shares_elem:
                    value = shares_elem.find("value")
                    if value and value.text:
                        try:
                            shares = int(float(value.text.strip()))
                        except ValueError:
                            pass

            # Get price per share
            price_per_share: float | None = None
            if amounts:
                price_elem = amounts.find("transactionPricePerShare")
                if price_elem:
                    value = price_elem.find("value")
                    if value and value.text:
                        try:
                            price_per_share = float(value.text.strip())
                        except ValueError:
                            pass

            # Calculate total value
            total_value: float | None = None
            if shares and price_per_share:
                total_value = shares * price_per_share

            # Create unique sec_url for this specific transaction
            # Use base URL + transaction details for uniqueness
            sec_url = f"{base_sec_url}#{owner_name}_{trade_date}_{transaction_type}_{shares}"

            if shares == 0:
                return None

            return InsiderTransaction(
                ticker=ticker,
                filing_date=filing_date,
                trade_date=trade_date,
                owner_name=owner_name,
                owner_title=owner_title,
                transaction_type=transaction_type,
                shares=shares,
                price_per_share=price_per_share,
                total_value=total_value,
                sec_url=sec_url,
            )

        except Exception as e:
            logger.debug(f"{self.name}: Failed to parse transaction: {e}")
            return None
