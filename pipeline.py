"""
Data Collection Pipeline
=========================
Collects financial data for a set of retail / consumer / industrial tickers
from SEC EDGAR, yfinance, and FRED, computes derived metrics, and saves
each variable to its own CSV file.  A missing-data-reasons log is also
produced.
"""

# ──────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────────────
import os
import time
import json
import math
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fredapi import Fred

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# 2. SETUP / CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
TICKERS = [
    "WMT", "TGT", "COST", "KR", "DG", "DLTR", "BBY", "HD", "LOW",
    "NKE", "PVH", "RL", "LEVI", "CROX", "DECK", "KHC", "GIS", "HSY",
    "CAG", "CPB", "PG", "CL", "KMB", "CHD", "SJM", "MKC", "HRL",
    "TSN", "CAT", "DE", "PCAR", "GWW", "FAST", "GPC", "LKQ",
]

FRED_API_KEY = os.environ.get("FRED_API_KEY", "8799cbbb3b6d618fdc00c495fda28939")

SEC_TICKER_MAP_URLS = [
    "https://www.sec.gov/files/company_tickers.json",
    "https://www.sec.gov/files/company_tickers_exchange.json",
]
SEC_COMPANYFACTS_URL = (
    "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
)

SEC_HEADERS = {
    "User-Agent": "DataPipeline research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

OUTPUT_DIR = "data_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global list to accumulate missing-data reasons
missing_reasons: list[dict] = []


def record_missing(ticker: str, field: str, reason: str) -> None:
    """Append a missing-data note."""
    missing_reasons.append(
        {"ticker": ticker, "field": field, "reason": reason}
    )


# ──────────────────────────────────────────────────────────────────────
# 3. SEC EDGAR HELPERS
# ──────────────────────────────────────────────────────────────────────
def build_ticker_cik_map() -> dict:
    """Return {TICKER: zero-padded CIK string} from SEC JSON endpoints."""
    mapping: dict[str, str] = {}
    for url in SEC_TICKER_MAP_URLS:
        try:
            resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # company_tickers.json  → dict of index→{cik_str, ticker, title}
            if isinstance(data, dict) and "0" in data:
                for entry in data.values():
                    t = entry.get("ticker", "").upper()
                    cik = str(entry.get("cik_str", ""))
                    if t and cik:
                        mapping[t] = cik.zfill(10)
            # company_tickers_exchange.json → {"data": [[cik, name, ticker, exchange], ...]}
            elif isinstance(data, dict) and "data" in data:
                for row in data["data"]:
                    t = str(row[2]).upper()
                    cik = str(row[0])
                    if t and cik:
                        mapping[t] = cik.zfill(10)
        except Exception as exc:
            log.warning("Could not load %s: %s", url, exc)
    return mapping


def get_company_facts(cik: str) -> dict:
    """Fetch the full XBRL companyfacts JSON for a given CIK."""
    url = SEC_COMPANYFACTS_URL.format(cik=cik)
    resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_quarterly_values(
    facts: dict,
    tags: list[str],
    *,
    namespace: str = "us-gaap",
    prefer_instant: bool = False,
) -> pd.DataFrame:
    """
    Walk through a list of XBRL tags (in priority order) and return the
    first tag that has data.  Returns a DataFrame with columns
    [end, val, filed] containing quarterly-granularity values.
    """
    ns_data = facts.get("facts", {}).get(namespace, {})
    for tag in tags:
        tag_data = ns_data.get(tag, {})
        units = tag_data.get("units", {})
        # pick USD, shares, or pure — whatever is present
        rows = []
        for unit_key, entries in units.items():
            for e in entries:
                form = e.get("form", "")
                if form not in ("10-Q", "10-K"):
                    continue
                start = e.get("start")
                end = e.get("end")
                val = e.get("val")
                filed = e.get("filed")
                if val is None or end is None:
                    continue
                # For duration items keep only quarterly rows (<= 100 days)
                if start and not prefer_instant:
                    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
                    if days > 100:
                        continue
                rows.append({"end": end, "val": val, "filed": filed})
        if rows:
            df = pd.DataFrame(rows)
            df["end"] = pd.to_datetime(df["end"])
            df = df.sort_values("end").drop_duplicates("end", keep="last")
            return df.reset_index(drop=True)
    return pd.DataFrame(columns=["end", "val", "filed"])


# ──────────────────────────────────────────────────────────────────────
# 4. SEC DATA COLLECTION — RAW FIELDS
# ──────────────────────────────────────────────────────────────────────
def collect_sec_data(ticker: str, cik: str) -> pd.DataFrame:
    """
    Pull every raw SEC field for one ticker and return a single
    quarterly DataFrame indexed by quarter-end date.
    """
    try:
        facts = get_company_facts(cik)
    except Exception as exc:
        log.error("SEC facts failed for %s (CIK %s): %s", ticker, cik, exc)
        record_missing(ticker, "ALL_SEC", f"companyfacts request failed: {exc}")
        return pd.DataFrame()

    time.sleep(0.15)  # polite rate limit

    field_specs: dict[str, dict] = {
        "Revenue": {
            "tags": [
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet",
                "Revenues",
            ]
        },
        "COGS": {
            "tags": [
                "CostOfGoodsSold",
                "CostOfSales",
                "CostOfGoodsAndServicesSold",
                "CostOfRevenue",
            ]
        },
        "GrossProfit": {
            "tags": ["GrossProfit"]
        },
        "OperatingCFO": {
            "tags": [
                "NetCashProvidedByUsedInOperatingActivities",
                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
            ]
        },
        "Inventory": {
            "tags": [
                "InventoryNet",
                "InventoryFinishedGoodsAndWorkInProcess",
                "InventoryGross",
            ],
            "instant": True,
        },
        "Receivables": {
            "tags": [
                "AccountsReceivableNetCurrent",
                "ReceivablesNetCurrent",
                "AccountsNotesAndLoansReceivableNetCurrent",
            ],
            "instant": True,
        },
        "Payables": {
            "tags": [
                "AccountsPayableCurrent",
                "AccountsPayableAndAccruedLiabilitiesCurrent",
            ],
            "instant": True,
        },
        "TotalAssets": {
            "tags": ["Assets"],
            "instant": True,
        },
        "CashEq": {
            "tags": [
                "CashAndCashEquivalentsAtCarryingValue",
                "CashCashEquivalentsAndShortTermInvestments",
            ],
            "instant": True,
        },
        "LTDebt": {
            "tags": ["LongTermDebtNoncurrent", "LongTermDebt"],
            "instant": True,
        },
        "STDebt_LTCurrent": {
            "tags": ["LongTermDebtCurrent"],
            "instant": True,
        },
        "STDebt_ShortBorrow": {
            "tags": ["ShortTermBorrowings"],
            "instant": True,
        },
        "STDebt_CommPaper": {
            "tags": ["CommercialPaper"],
            "instant": True,
        },
        "SharesOut": {
            "tags": [
                "EntityCommonStockSharesOutstanding",
                "CommonStockSharesOutstanding",
            ],
            "namespace": "dei",
            "instant": True,
        },
    }

    merged = None
    for field_name, spec in field_specs.items():
        ns = spec.get("namespace", "us-gaap")
        instant = spec.get("instant", False)
        df = extract_quarterly_values(
            facts, spec["tags"], namespace=ns, prefer_instant=instant
        )
        if df.empty:
            # try alternate namespace
            if ns == "dei":
                df = extract_quarterly_values(
                    facts, spec["tags"], namespace="us-gaap",
                    prefer_instant=instant,
                )
            elif ns == "us-gaap":
                df = extract_quarterly_values(
                    facts, spec["tags"], namespace="dei",
                    prefer_instant=instant,
                )
        if df.empty:
            record_missing(ticker, field_name,
                           "No SEC data found for any candidate tag")
            continue
        df = df.rename(columns={"val": field_name, "filed": f"{field_name}_filed"})
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="end", how="outer")

    if merged is None or merged.empty:
        record_missing(ticker, "ALL_SEC", "No SEC quarterly data found")
        return pd.DataFrame()

    merged = merged.sort_values("end").reset_index(drop=True)
    merged.insert(0, "ticker", ticker)
    return merged


# ──────────────────────────────────────────────────────────────────────
# 5. COMPUTE DERIVED SEC METRICS
# ──────────────────────────────────────────────────────────────────────
def compute_sec_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed columns to SEC quarterly data for one ticker."""
    if df.empty:
        return df

    d = df.copy()

    # --- Gross Profit (fill if not reported) ---
    if "GrossProfit" not in d.columns:
        d["GrossProfit"] = np.nan
    mask = d["GrossProfit"].isna()
    if "Revenue" in d.columns and "COGS" in d.columns:
        d.loc[mask, "GrossProfit"] = d.loc[mask, "Revenue"] - d.loc[mask, "COGS"]

    # --- Short-Term Debt ---
    for c in ("STDebt_LTCurrent", "STDebt_ShortBorrow", "STDebt_CommPaper"):
        if c not in d.columns:
            d[c] = 0.0
    d["STDebt"] = (
        d["STDebt_LTCurrent"].fillna(0)
        + d["STDebt_ShortBorrow"].fillna(0)
        + d["STDebt_CommPaper"].fillna(0)
    )

    # --- Total Debt ---
    if "LTDebt" not in d.columns:
        d["LTDebt"] = 0.0
    d["TotalDebt"] = d["STDebt"].fillna(0) + d["LTDebt"].fillna(0)

    # --- Margins ---
    if "Revenue" in d.columns:
        d["GrossMargin"] = d["GrossProfit"] / d["Revenue"].replace(0, np.nan)
    if "Revenue" in d.columns and "OperatingCFO" in d.columns:
        d["CFOMargin"] = d["OperatingCFO"] / d["Revenue"].replace(0, np.nan)

    # --- Leverage, Cash Ratio, Size ---
    if "TotalAssets" in d.columns:
        ta = d["TotalAssets"].replace(0, np.nan)
        d["Leverage"] = d["TotalDebt"] / ta
        if "CashEq" in d.columns:
            d["CashRatio"] = d["CashEq"] / ta
        d["Size"] = np.log(d["TotalAssets"].replace(0, np.nan))

    # --- TTM Revenue & COGS ---
    if "Revenue" in d.columns:
        d["TTM_Revenue"] = d["Revenue"].rolling(4, min_periods=4).sum()
    if "COGS" in d.columns:
        d["TTM_COGS"] = d["COGS"].rolling(4, min_periods=4).sum()

    # --- Averages (current + prior quarter) ---
    for raw, avg in [("Inventory", "AvgInventory"),
                     ("Receivables", "AvgReceivables"),
                     ("Payables", "AvgPayables")]:
        if raw in d.columns:
            d[avg] = (d[raw] + d[raw].shift(1)) / 2

    # --- DIO, DSO, DPO, CCC ---
    if "AvgInventory" in d.columns and "TTM_COGS" in d.columns:
        d["DIO"] = 365 * d["AvgInventory"] / d["TTM_COGS"].replace(0, np.nan)
    if "AvgReceivables" in d.columns and "TTM_Revenue" in d.columns:
        d["DSO"] = 365 * d["AvgReceivables"] / d["TTM_Revenue"].replace(0, np.nan)
    if "AvgPayables" in d.columns and "TTM_COGS" in d.columns:
        d["DPO"] = 365 * d["AvgPayables"] / d["TTM_COGS"].replace(0, np.nan)
    for c in ("DIO", "DSO", "DPO"):
        if c not in d.columns:
            d[c] = np.nan
    d["CCC"] = d["DIO"] + d["DSO"] - d["DPO"]

    # --- YoY Growth (log) ---
    if "Revenue" in d.columns:
        d["SalesGrowthYoY"] = np.log(
            d["Revenue"] / d["Revenue"].shift(4).replace(0, np.nan)
        )
    if "Inventory" in d.columns:
        d["InventoryGrowthYoY"] = np.log(
            d["Inventory"] / d["Inventory"].shift(4).replace(0, np.nan)
        )

    # --- Raw Abnormal Inventory Growth ---
    if "InventoryGrowthYoY" in d.columns and "SalesGrowthYoY" in d.columns:
        d["RawAbnInvGrowth"] = d["InventoryGrowthYoY"] - d["SalesGrowthYoY"]

    # --- Future Margin Changes (lead 4 quarters) ---
    if "GrossMargin" in d.columns:
        d["FutureGrossMarginChg"] = d["GrossMargin"].shift(-4) - d["GrossMargin"]
    if "CFOMargin" in d.columns:
        d["FutureCFOMarginChg"] = d["CFOMargin"].shift(-4) - d["CFOMargin"]

    return d


# ──────────────────────────────────────────────────────────────────────
# 6. COLLECT ALL SEC DATA
# ──────────────────────────────────────────────────────────────────────
def collect_all_sec(tickers: list[str]) -> pd.DataFrame:
    """Collect and merge SEC data for every ticker."""
    log.info("Building SEC ticker → CIK map …")
    cik_map = build_ticker_cik_map()
    frames = []
    for tkr in tickers:
        cik = cik_map.get(tkr)
        if not cik:
            record_missing(tkr, "CIK", "Ticker not found in SEC mapping")
            log.warning("No CIK for %s — skipping SEC", tkr)
            continue
        log.info("Collecting SEC data for %s (CIK %s) …", tkr, cik)
        raw = collect_sec_data(tkr, cik)
        derived = compute_sec_derived(raw)
        frames.append(derived)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# 7. RESIDUAL ABNORMAL INVENTORY GROWTH (panel regression)
# ──────────────────────────────────────────────────────────────────────
def compute_residual_abnormal_inv(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Run: InventoryGrowthYoY = β·SalesGrowthYoY + firm FE + time FE + ε
    and append the residual as ResidualAbnInvGrowth.
    """
    df = panel.copy()
    needed = ["ticker", "end", "InventoryGrowthYoY", "SalesGrowthYoY"]
    if not all(c in df.columns for c in needed):
        df["ResidualAbnInvGrowth"] = np.nan
        return df

    sub = df[needed].dropna()
    if len(sub) < 10:
        df["ResidualAbnInvGrowth"] = np.nan
        return df

    sub = sub.copy()
    sub["quarter"] = sub["end"].dt.to_period("Q").astype(str)

    # firm and time dummies
    firm_dummies = pd.get_dummies(sub["ticker"], prefix="firm", drop_first=True,
                                  dtype=float)
    time_dummies = pd.get_dummies(sub["quarter"], prefix="qtr", drop_first=True,
                                  dtype=float)
    X = pd.concat(
        [sub[["SalesGrowthYoY"]], firm_dummies, time_dummies], axis=1
    )
    X = sm.add_constant(X)
    y = sub["InventoryGrowthYoY"]

    try:
        model = OLS(y, X).fit()
        sub["ResidualAbnInvGrowth"] = model.resid
    except Exception as exc:
        log.warning("Panel regression failed: %s", exc)
        sub["ResidualAbnInvGrowth"] = np.nan

    df = df.merge(
        sub[["ticker", "end", "ResidualAbnInvGrowth"]],
        on=["ticker", "end"],
        how="left",
    )
    return df


# ──────────────────────────────────────────────────────────────────────
# 8. YFINANCE DATA COLLECTION
# ──────────────────────────────────────────────────────────────────────
def collect_yfinance_data(
    panel: pd.DataFrame, tickers: list[str]
) -> pd.DataFrame:
    """
    For each ticker/quarter row, attach:
      - EntryPrice  (first close on or after the SEC filing date)
      - Next12MReturn
      - BenchmarkReturn (SPY)
      - ExcessReturn
      - SharesOut (yfinance fallback)
    """
    df = panel.copy()
    if df.empty:
        return df

    # Determine date range for price download
    filed_cols = [c for c in df.columns if c.endswith("_filed")]
    available_dates = pd.Series(dtype="datetime64[ns]")
    for fc in filed_cols:
        available_dates = pd.concat(
            [available_dates, pd.to_datetime(df[fc], errors="coerce")]
        )
    if available_dates.dropna().empty:
        # fall back to quarter-end dates
        available_dates = pd.to_datetime(df["end"], errors="coerce")

    min_date = available_dates.min() - timedelta(days=30)
    max_date = available_dates.max() + timedelta(days=400)
    today = datetime.today()
    if max_date > today:
        max_date = today

    # Download price data
    all_tickers = list(set(tickers)) + ["SPY"]
    log.info("Downloading yfinance prices for %d tickers …", len(all_tickers))
    try:
        prices = yf.download(
            all_tickers,
            start=min_date.strftime("%Y-%m-%d"),
            end=max_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if isinstance(prices.columns, pd.MultiIndex):
            close = prices["Close"]
        else:
            close = prices[["Close"]].rename(columns={"Close": all_tickers[0]})
    except Exception as exc:
        log.error("yfinance download failed: %s", exc)
        for col in ("EntryPrice", "Next12MReturn", "BenchmarkReturn",
                     "ExcessReturn"):
            df[col] = np.nan
        return df

    # Helper: find first valid close on or after a date
    def first_close_on_or_after(tkr_col, date):
        if tkr_col not in close.columns:
            return np.nan
        subset = close.loc[close.index >= pd.Timestamp(date), tkr_col].dropna()
        return subset.iloc[0] if len(subset) > 0 else np.nan

    # Helper: close N business days later
    def close_after_n_bdays(tkr_col, date, n=252):
        if tkr_col not in close.columns:
            return np.nan
        target = pd.Timestamp(date) + pd.tseries.offsets.BDay(n)
        subset = close.loc[close.index >= target, tkr_col].dropna()
        return subset.iloc[0] if len(subset) > 0 else np.nan

    # Determine the "available_date" for each row (earliest filed date)
    def row_available_date(row):
        dates = []
        for fc in filed_cols:
            v = row.get(fc)
            if pd.notna(v):
                dates.append(pd.Timestamp(v))
        if not dates:
            return row["end"]
        return max(dates)

    df["available_date"] = df.apply(row_available_date, axis=1)
    df["available_date"] = pd.to_datetime(df["available_date"])

    # Vectorised-ish computation (iterating rows for clarity)
    entry_prices, returns_12m, bench_12m = [], [], []
    for _, row in df.iterrows():
        tkr = row["ticker"]
        ad = row["available_date"]

        ep = first_close_on_or_after(tkr, ad)
        entry_prices.append(ep)

        ep_future = close_after_n_bdays(tkr, ad, 252)
        ret = (ep_future / ep - 1) if (pd.notna(ep) and pd.notna(ep_future) and ep != 0) else np.nan
        returns_12m.append(ret)

        spy0 = first_close_on_or_after("SPY", ad)
        spy1 = close_after_n_bdays("SPY", ad, 252)
        br = (spy1 / spy0 - 1) if (pd.notna(spy0) and pd.notna(spy1) and spy0 != 0) else np.nan
        bench_12m.append(br)

    df["EntryPrice"] = entry_prices
    df["Next12MReturn"] = returns_12m
    df["BenchmarkReturn"] = bench_12m
    df["ExcessReturn"] = df["Next12MReturn"] - df["BenchmarkReturn"]

    # Shares outstanding fallback from yfinance
    if "SharesOut" in df.columns:
        mask = df["SharesOut"].isna()
        if mask.any():
            for tkr in df.loc[mask, "ticker"].unique():
                try:
                    info = yf.Ticker(tkr).info
                    shares = info.get("sharesOutstanding")
                    if shares:
                        df.loc[(df["ticker"] == tkr) & mask, "SharesOut"] = shares
                except Exception:
                    pass
    else:
        df["SharesOut"] = np.nan
        for tkr in df["ticker"].unique():
            try:
                info = yf.Ticker(tkr).info
                shares = info.get("sharesOutstanding")
                if shares:
                    df.loc[df["ticker"] == tkr, "SharesOut"] = shares
            except Exception:
                pass

    return df


# ──────────────────────────────────────────────────────────────────────
# 9. VALUATION METRICS
# ──────────────────────────────────────────────────────────────────────
def compute_valuation(df: pd.DataFrame) -> pd.DataFrame:
    """Market Cap, Enterprise Value, EV/Sales."""
    d = df.copy()
    if "EntryPrice" in d.columns and "SharesOut" in d.columns:
        d["MarketCap"] = d["EntryPrice"] * d["SharesOut"]
    else:
        d["MarketCap"] = np.nan

    for c in ("TotalDebt", "CashEq"):
        if c not in d.columns:
            d[c] = 0.0

    d["EnterpriseValue"] = d["MarketCap"] + d["TotalDebt"].fillna(0) - d["CashEq"].fillna(0)

    if "TTM_Revenue" in d.columns:
        d["EV_Sales"] = d["EnterpriseValue"] / d["TTM_Revenue"].replace(0, np.nan)
    else:
        d["EV_Sales"] = np.nan
    return d


# ──────────────────────────────────────────────────────────────────────
# 10. FRED DATA COLLECTION
# ──────────────────────────────────────────────────────────────────────
def collect_fred_data(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach Fed Funds, 10Y Treasury, Recession dummy to each row."""
    df = panel.copy()
    try:
        fred = Fred(api_key=FRED_API_KEY)
    except Exception as exc:
        log.error("FRED init failed: %s", exc)
        for c in ("FedFunds", "Treasury10Y", "RecessionDummy"):
            df[c] = np.nan
            record_missing("ALL", c, f"FRED API init failed: {exc}")
        return df

    # Fetch series once
    series_map = {
        "FedFunds": "FEDFUNDS",
        "Treasury10Y": "DGS10",
        "RecessionDummy": "USREC",
    }
    fred_data: dict[str, pd.Series] = {}
    for col, sid in series_map.items():
        try:
            s = fred.get_series(sid)
            s.index = pd.to_datetime(s.index)
            fred_data[col] = s
        except Exception as exc:
            log.warning("FRED series %s failed: %s", sid, exc)
            record_missing("ALL", col, f"FRED series fetch failed: {exc}")

    ad = pd.to_datetime(df["available_date"]) if "available_date" in df.columns else pd.to_datetime(df["end"])

    for col, sid in series_map.items():
        if col not in fred_data:
            df[col] = np.nan
            continue
        s = fred_data[col].sort_index()
        vals = []
        for dt in ad:
            subset = s.loc[s.index <= dt]
            vals.append(subset.iloc[-1] if len(subset) > 0 else np.nan)
        df[col] = vals

    return df


# ──────────────────────────────────────────────────────────────────────
# 11. SAVE TO CSV FILES (one per variable)
# ──────────────────────────────────────────────────────────────────────

# Mapping of output CSV filename → columns to include
CSV_GROUPS: dict[str, list[str]] = {
    "revenue.csv":              ["ticker", "end", "Revenue"],
    "cogs.csv":                 ["ticker", "end", "COGS"],
    "gross_profit.csv":         ["ticker", "end", "GrossProfit"],
    "gross_margin.csv":         ["ticker", "end", "GrossMargin"],
    "operating_cfo.csv":        ["ticker", "end", "OperatingCFO"],
    "cfo_margin.csv":           ["ticker", "end", "CFOMargin"],
    "inventory.csv":            ["ticker", "end", "Inventory"],
    "receivables.csv":          ["ticker", "end", "Receivables"],
    "payables.csv":             ["ticker", "end", "Payables"],
    "total_assets.csv":         ["ticker", "end", "TotalAssets"],
    "cash_equivalents.csv":     ["ticker", "end", "CashEq"],
    "short_term_debt.csv":      ["ticker", "end", "STDebt"],
    "long_term_debt.csv":       ["ticker", "end", "LTDebt"],
    "total_debt.csv":           ["ticker", "end", "TotalDebt"],
    "shares_outstanding.csv":   ["ticker", "end", "SharesOut"],
    "leverage.csv":             ["ticker", "end", "Leverage"],
    "cash_ratio.csv":           ["ticker", "end", "CashRatio"],
    "size.csv":                 ["ticker", "end", "Size"],
    "ttm_revenue.csv":          ["ticker", "end", "TTM_Revenue"],
    "ttm_cogs.csv":             ["ticker", "end", "TTM_COGS"],
    "avg_inventory.csv":        ["ticker", "end", "AvgInventory"],
    "avg_receivables.csv":      ["ticker", "end", "AvgReceivables"],
    "avg_payables.csv":         ["ticker", "end", "AvgPayables"],
    "dio.csv":                  ["ticker", "end", "DIO"],
    "dso.csv":                  ["ticker", "end", "DSO"],
    "dpo.csv":                  ["ticker", "end", "DPO"],
    "ccc.csv":                  ["ticker", "end", "CCC"],
    "sales_growth_yoy.csv":     ["ticker", "end", "SalesGrowthYoY"],
    "inventory_growth_yoy.csv": ["ticker", "end", "InventoryGrowthYoY"],
    "raw_abn_inv_growth.csv":   ["ticker", "end", "RawAbnInvGrowth"],
    "residual_abn_inv_growth.csv": ["ticker", "end", "ResidualAbnInvGrowth"],
    "future_gross_margin_chg.csv": ["ticker", "end", "FutureGrossMarginChg"],
    "future_cfo_margin_chg.csv":   ["ticker", "end", "FutureCFOMarginChg"],
    "entry_price.csv":          ["ticker", "end", "available_date", "EntryPrice"],
    "next_12m_return.csv":      ["ticker", "end", "Next12MReturn"],
    "benchmark_return.csv":     ["ticker", "end", "BenchmarkReturn"],
    "excess_return.csv":        ["ticker", "end", "ExcessReturn"],
    "market_cap.csv":           ["ticker", "end", "MarketCap"],
    "enterprise_value.csv":     ["ticker", "end", "EnterpriseValue"],
    "ev_sales.csv":             ["ticker", "end", "EV_Sales"],
    "fed_funds.csv":            ["ticker", "end", "FedFunds"],
    "treasury_10y.csv":         ["ticker", "end", "Treasury10Y"],
    "recession_dummy.csv":      ["ticker", "end", "RecessionDummy"],
}


def save_csvs(panel: pd.DataFrame) -> None:
    """Write one CSV per variable."""
    for fname, cols in CSV_GROUPS.items():
        present = [c for c in cols if c in panel.columns]
        if not present or len(present) <= 2:  # only ticker+end
            log.warning("Skipping %s — no data column present", fname)
            continue
        out = panel[present].dropna(subset=[c for c in present
                                            if c not in ("ticker", "end",
                                                         "available_date")])
        path = os.path.join(OUTPUT_DIR, fname)
        out.to_csv(path, index=False)
        log.info("Saved %s  (%d rows)", fname, len(out))


def save_missing_reasons() -> None:
    """Write missing-data reasons to CSV."""
    path = os.path.join(OUTPUT_DIR, "missing_data_reasons.csv")
    pd.DataFrame(missing_reasons).to_csv(path, index=False)
    log.info("Saved missing_data_reasons.csv  (%d entries)", len(missing_reasons))


# ──────────────────────────────────────────────────────────────────────
# 12. MAIN — run everything
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("=" * 60)
    log.info("DATA COLLECTION PIPELINE — START")
    log.info("=" * 60)

    # Step 1: SEC raw + derived
    panel = collect_all_sec(TICKERS)
    if panel.empty:
        log.error("No SEC data collected — aborting.")
        save_missing_reasons()
        return

    # Step 2: Residual abnormal inventory growth
    panel = compute_residual_abnormal_inv(panel)

    # Step 3: yfinance prices & returns
    panel = collect_yfinance_data(panel, TICKERS)

    # Step 4: Valuation metrics
    panel = compute_valuation(panel)

    # Step 5: FRED macro variables
    panel = collect_fred_data(panel)

    # Step 6: Save full panel
    full_path = os.path.join(OUTPUT_DIR, "full_panel.csv")
    panel.to_csv(full_path, index=False)
    log.info("Saved full_panel.csv  (%d rows × %d cols)",
             len(panel), len(panel.columns))

    # Step 7: Save individual CSVs
    save_csvs(panel)

    # Step 8: Missing data reasons
    save_missing_reasons()

    log.info("=" * 60)
    log.info("DATA COLLECTION PIPELINE — DONE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
