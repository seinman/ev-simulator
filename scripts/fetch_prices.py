#!/usr/bin/env python3
"""
Fetches one year of APX day-ahead half-hourly prices from the Elexon BMRS API
and caches them to data/input/prices_apx_2023.csv.

The BMRS endpoint returns market index data from APX Power UK (APXMIDP), which
replaced N2EX as the active UK day-ahead price provider. Prices are in £/MWh
at 30-minute settlement period resolution (48 periods per day).

Run once before starting the simulator:
    python scripts/fetch_prices.py

Requires: requests (poetry install)
"""

import csv
import time
from datetime import date, timedelta
from pathlib import Path

import requests

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
DATA_PROVIDER = "APXMIDP"
FETCH_YEAR = 2025
WINDOW_DAYS = 7  # API maximum per request
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "input" / f"prices_apx_{FETCH_YEAR}.csv"
RATE_LIMIT_SECONDS = 0.5
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = [2, 4]  # waits before 2nd and 3rd attempts


def fetch_window(session: requests.Session, from_date: date, to_date: date) -> list[dict]:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            wait = RETRY_BACKOFF_SECONDS[attempt - 1]
            print(f"  Retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)
        try:
            response = session.get(
                f"{BASE_URL}/balancing/pricing/market-index",
                params={
                    "from": f"{from_date.isoformat()}T00:00:00Z",
                    "to": f"{to_date.isoformat()}T23:30:00Z",
                    "dataProviders": DATA_PROVIDER,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["data"]
        except requests.RequestException as e:
            print(f"  Request failed: {e}")
            last_exc = e
    raise RuntimeError(
        f"Failed to fetch {from_date} → {to_date} after {MAX_RETRIES} attempts."
    ) from last_exc


def _validate_output(rows: list[dict], year: int) -> None:
    is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    expected_days = 366 if is_leap else 365
    expected_rows = expected_days * 48

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} records ({expected_days} days × 48 periods), "
            f"got {len(rows)}. The download may be incomplete; re-run the script."
        )

    periods_by_date: dict[str, set[int]] = {}
    for r in rows:
        periods_by_date.setdefault(r["settlement_date"], set()).add(r["settlement_period"])

    incomplete = [d for d, ps in periods_by_date.items() if len(ps) != 48]
    if incomplete:
        raise RuntimeError(
            f"{len(incomplete)} dates have fewer than 48 settlement periods: "
            f"{incomplete[:5]}{'...' if len(incomplete) > 5 else ''}. "
            "Re-run the script to fetch a complete dataset."
        )


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    year_start = date(FETCH_YEAR, 1, 1)
    year_end = date(FETCH_YEAR, 12, 31)
    current = year_start
    all_rows: list[dict] = []

    with requests.Session() as session:
        while current <= year_end:
            window_end = min(current + timedelta(days=WINDOW_DAYS - 1), year_end)
            print(f"Fetching {current} → {window_end}...")

            records = fetch_window(session, current, window_end)
            for r in records:
                if year_start.isoformat() <= r["settlementDate"] <= year_end.isoformat():
                    all_rows.append({
                        "settlement_date": r["settlementDate"],
                        "settlement_period": r["settlementPeriod"],
                        "price_gbp_per_mwh": r["price"],
                    })

            current = window_end + timedelta(days=1)
            time.sleep(RATE_LIMIT_SECONDS)

    all_rows.sort(key=lambda r: (r["settlement_date"], r["settlement_period"]))
    _validate_output(all_rows, FETCH_YEAR)

    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["settlement_date", "settlement_period", "price_gbp_per_mwh"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
