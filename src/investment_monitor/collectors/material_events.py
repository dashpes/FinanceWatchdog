"""Collector for SEC 8-K material corporate events (broad, market-wide).

Reuses the EDGAR daily-index plumbing from :class:`InsiderCollector` (rate
limiting, trading-day bridge, index URL/format) but reads only each filing's
SGML *header* — the first few KB carry the Item codes, filer CIK/name, and filed
date — so a full day of 8-Ks ingests without downloading exhibit bodies. Tickers
come from the SEC's official ``company_tickers.json`` CIK map (fetched once per
run); CIK-only filers (funds, private issuers) are stored with ``ticker=None``.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime

import httpx
from loguru import logger

from ..storage.event_models import MaterialEvent
from ..storage.event_operations import material_event_exists
from .base import CollectorResult
from .insider import InsiderCollector

# Only the header block is needed; Range keeps a 2000-filing day cheap. Servers
# that ignore Range simply return the full body — correctness is unaffected.
_HEADER_RANGE_BYTES = 32_768

_CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

# EDGAR header description -> 8-K Item code. Descriptions are the standardized
# ITEM INFORMATION strings EDGAR writes; match on stable keyword prefixes.
_ITEM_PATTERNS: list[tuple[str, str]] = [
    ("entry into a material definitive agreement", "1.01"),
    ("termination of a material definitive agreement", "1.02"),
    ("bankruptcy or receivership", "1.03"),
    ("completion of acquisition or disposition", "2.01"),
    ("results of operations and financial condition", "2.02"),
    ("creation of a direct financial obligation", "2.03"),
    ("triggering events that accelerate", "2.04"),
    ("costs associated with exit or disposal", "2.05"),
    ("material impairments", "2.06"),
    ("notice of delisting", "3.01"),
    ("unregistered sales of equity securities", "3.02"),
    ("material modification to rights of security holders", "3.03"),
    ("changes in registrant's certifying accountant", "4.01"),
    ("non-reliance on previously issued financial statements", "4.02"),
    ("changes in control of registrant", "5.01"),
    ("departure of directors or certain officers", "5.02"),
    ("amendments to articles of incorporation", "5.03"),
    ("temporary suspension of trading", "5.04"),
    ("amendments to the registrant's code of ethics", "5.05"),
    ("change in shell company status", "5.06"),
    ("submission of matters to a vote of security holders", "5.07"),
    ("shareholder director nominations", "5.08"),
    ("regulation fd disclosure", "7.01"),
    ("other events", "8.01"),
    ("financial statements and exhibits", "9.01"),
]


def map_item_descriptions(descriptions: list[str]) -> list[str]:
    """Map EDGAR ITEM INFORMATION description lines to 8-K Item codes (pure)."""
    codes: list[str] = []
    for desc in descriptions:
        d = desc.strip().lower()
        for prefix, code in _ITEM_PATTERNS:
            if d.startswith(prefix) and code not in codes:
                codes.append(code)
                break
    return codes


def parse_sgml_header(text: str) -> dict:
    """Extract cik / company / filed date / item descriptions from a submission header."""
    items = re.findall(r"ITEM INFORMATION:\s*(.+)", text)
    cik_m = re.search(r"CENTRAL INDEX KEY:\s*(\d+)", text)
    name_m = re.search(r"COMPANY CONFORMED NAME:\s*(.+)", text)
    date_m = re.search(r"FILED AS OF DATE:\s*(\d{8})", text)
    filed = None
    if date_m:
        raw = date_m.group(1)
        try:
            filed = date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
        except ValueError:
            filed = None
    return {
        "cik": cik_m.group(1).lstrip("0") or "0" if cik_m else None,
        "company_name": name_m.group(1).strip() if name_m else None,
        "filed_date": filed,
        "item_descriptions": [i.strip() for i in items],
    }


class MaterialEventsCollector(InsiderCollector):
    """Broad 8-K ingestion via the same EDGAR daily indexes as Form 4."""

    name = "material_events"

    async def collect_all(
        self, *, days_back: int = 1, limit: int | None = None
    ) -> CollectorResult:
        """Ingest ALL 8-K filings for the last ``days_back`` trading days."""
        started_at = datetime.now()
        records = 0
        errors: list[str] = []

        async with httpx.AsyncClient(
            headers={"User-Agent": self.USER_AGENT}, timeout=30.0
        ) as client:
            cik_to_ticker = await self._cik_ticker_map(client)

            filing_urls: list[str] = []
            for d in self._recent_business_dates(days_back):
                try:
                    idx = await self._get(client, self._daily_index_url(d))
                    filing_urls.extend(self._parse_index_for_8k(idx))
                except Exception as e:  # noqa: BLE001 - a missing/holiday index is fine
                    logger.debug(f"{self.name}: daily index {d} unavailable: {e}")

            if limit is not None:
                filing_urls = filing_urls[:limit]
            logger.info(f"{self.name}: {len(filing_urls)} 8-K filings to parse (broad)")

            for form_url in filing_urls:
                try:
                    if material_event_exists(self.session, form_url):
                        continue
                    header = parse_sgml_header(await self._get_header(client, form_url))
                    if not header["cik"] or not header["filed_date"]:
                        continue
                    self.session.add(MaterialEvent(
                        ticker=cik_to_ticker.get(header["cik"]),
                        cik=header["cik"],
                        company_name=header["company_name"],
                        form_type="8-K",
                        items=map_item_descriptions(header["item_descriptions"]),
                        item_descriptions="\n".join(header["item_descriptions"]) or None,
                        filed_date=header["filed_date"],
                        sec_url=form_url,
                    ))
                    records += 1
                except Exception as e:  # noqa: BLE001 - one bad filing must not abort the run
                    logger.debug(f"{self.name}: failed filing {form_url}: {e}")

        try:
            self.session.commit()
        except Exception as e:  # noqa: BLE001
            self.session.rollback()
            msg = f"Failed to commit material events: {e}"
            errors.append(msg)
            logger.error(f"{self.name}: {msg}")

        return CollectorResult(
            collector_name=self.name,
            success=len(errors) == 0,
            records_collected=records,
            errors=errors,
            started_at=started_at,
            finished_at=datetime.now(),
        )

    async def _get_header(self, client: httpx.AsyncClient, url: str) -> str:
        """Rate-limited GET of just the submission's leading bytes (SGML header)."""
        await self._rate_limit()
        response = await client.get(
            url, headers={"Range": f"bytes=0-{_HEADER_RANGE_BYTES - 1}"}
        )
        response.raise_for_status()
        return response.text

    async def _cik_ticker_map(self, client: httpx.AsyncClient) -> dict[str, str]:
        """SEC's official CIK -> ticker map (one fetch per run; empty map on failure)."""
        try:
            raw = await self._get(client, _CIK_MAP_URL)
            data = json.loads(raw)
            return {
                str(row["cik_str"]): str(row["ticker"]).upper()
                for row in data.values()
                if row.get("cik_str") and row.get("ticker")
            }
        except Exception as e:  # noqa: BLE001 - events still ingest, just ticker-less
            logger.warning(f"{self.name}: CIK->ticker map unavailable: {e}")
            return {}

    def _parse_index_for_8k(self, idx_text: str) -> list[str]:
        """Pull 8-K submission URLs out of an EDGAR daily ``form.*.idx``."""
        urls: list[str] = []
        for line in idx_text.splitlines():
            parts = line.split()
            if len(parts) < 5 or parts[0] != "8-K":  # exact 8-K (not 8-K/A amendments)
                continue
            file_name = parts[-1]
            if file_name.startswith("edgar/data/") and file_name.endswith(".txt"):
                urls.append(f"{self.EDGAR_ARCHIVES}/{file_name}")
        return urls
