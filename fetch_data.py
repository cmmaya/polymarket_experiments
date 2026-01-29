import os
import time
import json
import requests
import pandas as pd
from collections import defaultdict
from typing import Any, Dict, List, Tuple

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
CLOB_PRICE_HISTORY_URL = "https://clob.polymarket.com/prices-history"
DATA_TRADES_URL = "https://data-api.polymarket.com/trades"

# -----------------------
# Core helpers
# -----------------------
def ensure_data_dir(path: str = "data") -> None:
    os.makedirs(path, exist_ok=True)

def fetch_json(url: str, params: Dict[str, Any], timeout: int = 30) -> Any:
    r = requests.get(url, params=params, timeout=timeout)
    if not r.ok:
        raise requests.HTTPError(
            f"{r.status_code} {r.reason} for {r.url}\nResponse: {r.text}",
            response=r,
        )
    return r.json()

def parse_json_maybe(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value

# -----------------------
# Gamma: slug -> IDs
# -----------------------
def get_market_by_slug(slug: str) -> Dict[str, Any]:
    markets = fetch_json(GAMMA_MARKETS_URL, {"slug": slug})
    if not markets:
        raise ValueError(f"No market found for slug={slug}")
    return markets[0]

def get_yes_no_token_ids(market: Dict[str, Any]) -> Tuple[str, str]:
    """
    Prefer deterministic YES/NO mapping via outcomes; fallback to ordering.
    """
    clob_ids = parse_json_maybe(market.get("clobTokenIds"))
    outcomes = parse_json_maybe(market.get("outcomes"))

    if not isinstance(clob_ids, list) or len(clob_ids) < 2:
        raise ValueError("Could not parse clobTokenIds (need 2 token IDs for YES/NO).")
    clob_ids = [str(x) for x in clob_ids]

    if isinstance(outcomes, list) and len(outcomes) >= 2:
        outcomes_norm = [str(x).strip().lower() for x in outcomes]
        if "yes" in outcomes_norm and "no" in outcomes_norm:
            yes_i = outcomes_norm.index("yes")
            no_i = outcomes_norm.index("no")
            return clob_ids[yes_i], clob_ids[no_i]

    return clob_ids[0], clob_ids[1]

# -----------------------
# Prices: /prices-history (chunked)
# -----------------------
def price_history_chunk(token_id: str, start_ts: int, end_ts: int, fidelity_minutes: int) -> pd.DataFrame:
    """
    IMPORTANT:
      When using startTs/endTs, DO NOT pass interval.
      Use fidelity (minutes) for resolution.
    """
    params = {
        "market": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": fidelity_minutes,
    }
    data = fetch_json(CLOB_PRICE_HISTORY_URL, params)
    hist = data.get("history", [])
    df = pd.DataFrame(hist)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "price"])
    df = df.rename(columns={"t": "timestamp", "p": "price"})[["timestamp", "price"]]
    df["timestamp"] = df["timestamp"].astype(int)
    df["price"] = df["price"].astype(float)
    return df.sort_values("timestamp").reset_index(drop=True)

def get_price_history_chunked(
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity_minutes: int,
    chunk_days: int = 7,
) -> pd.DataFrame:
    """
    Split [start_ts, end_ts] into smaller windows to avoid:
      {"error":"invalid filters: 'startTs' and 'endTs' interval is too long"}
    """
    chunk_seconds = chunk_days * 24 * 3600
    cur = start_ts
    frames: List[pd.DataFrame] = []
    while cur < end_ts:
        nxt = min(cur + chunk_seconds, end_ts)
        frames.append(price_history_chunk(token_id, cur, nxt, fidelity_minutes))
        cur = nxt

    if not frames:
        return pd.DataFrame(columns=["timestamp", "price"])

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out

# -----------------------
# Trades -> volume (bucketed)
# -----------------------
def get_all_trades(condition_id: str, limit: int = 1000, max_pages: int = 200) -> List[Dict[str, Any]]:
    all_trades: List[Dict[str, Any]] = []
    for page in range(max_pages):
        offset = page * limit
        batch = fetch_json(
            DATA_TRADES_URL,
            {"market": condition_id, "limit": limit, "offset": offset, "takerOnly": "true"},
        )
        if not batch:
            break
        all_trades.extend(batch)
        if len(batch) < limit:
            break
    return all_trades

def bucket_volume(trades: List[Dict[str, Any]], bucket_seconds: int) -> pd.DataFrame:
    vol = defaultdict(float)
    for tr in trades:
        try:
            ts = int(tr["timestamp"])
            size = float(tr["size"])
        except (KeyError, ValueError, TypeError):
            continue
        b = (ts // bucket_seconds) * bucket_seconds
        vol[b] += size

    df = pd.DataFrame({"timestamp": list(vol.keys()), "volume": list(vol.values())})
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "volume"])
    df["timestamp"] = df["timestamp"].astype(int)
    df["volume"] = df["volume"].astype(float)
    return df.sort_values("timestamp").reset_index(drop=True)

# -----------------------
# FIX: keep hourly samples correctly
# -----------------------
def _to_bucket(df: pd.DataFrame, ts_col: str, bucket_seconds: int, bucket_col: str = "bucket_ts") -> pd.DataFrame:
    df = df.copy()
    df[bucket_col] = (df[ts_col] // bucket_seconds) * bucket_seconds
    return df

def _hourly_last_price(prices: pd.DataFrame, bucket_seconds: int) -> pd.DataFrame:
    """
    Convert irregular price ticks to strict bucket timestamps by taking the LAST tick per bucket.
    """
    if prices.empty:
        return pd.DataFrame(columns=["timestamp", "price_yes", "price_no"])

    prices = prices.sort_values("timestamp").reset_index(drop=True)
    prices = _to_bucket(prices, "timestamp", bucket_seconds, bucket_col="bucket_ts")

    hourly = (
        prices.groupby("bucket_ts", as_index=False)
              .agg({"price_yes": "last", "price_no": "last"})
              .rename(columns={"bucket_ts": "timestamp"})
              .sort_values("timestamp")
              .reset_index(drop=True)
    )

    # forward-fill across buckets (so you get a continuous price series)
    hourly["price_yes"] = hourly["price_yes"].ffill()
    hourly["price_no"] = hourly["price_no"].ffill()
    return hourly

def export_slug_csv(slug: str, days_back: int = 30, bucket: str = "1h", price_chunk_days: int = 7) -> str:
    """
    Export STRICT bucket samples:
      timestamp (bucket start),
      price_yes (last price in bucket),
      price_no  (last price in bucket),
      volume    (sum trade sizes in bucket)

    This edits your previously-working version by:
    - preserving your chunked price fetching (works)
    - resampling prices to the bucket grid (fixes “empty hourly”)
    - keeping volume bucketed (already correct)
    - merging on bucket timestamps (correct)
    """
    ensure_data_dir("data")

    bucket_to_seconds = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "6h": 21600, "1d": 86400}
    if bucket not in bucket_to_seconds:
        raise ValueError(f"Unsupported bucket={bucket}. Choose one of {list(bucket_to_seconds.keys())}.")
    bucket_seconds = bucket_to_seconds[bucket]
    fidelity_minutes = max(1, bucket_seconds // 60)  # align price sampling to bucket

    market = get_market_by_slug(slug)
    condition_id = market.get("conditionId")
    if not condition_id:
        raise ValueError("Gamma response missing conditionId (needed for trades/volume).")
    yes_token_id, no_token_id = get_yes_no_token_ids(market)

    end_ts = int(time.time())
    start_ts = end_ts - days_back * 24 * 3600

    # -------------------
    # Prices: chunked (working)
    # -------------------
    yes_df = get_price_history_chunked(
        yes_token_id, start_ts, end_ts, fidelity_minutes, chunk_days=price_chunk_days
    ).rename(columns={"price": "price_yes"})

    no_df = get_price_history_chunked(
        no_token_id, start_ts, end_ts, fidelity_minutes, chunk_days=price_chunk_days
    ).rename(columns={"price": "price_no"})

    # Merge raw ticks
    raw_prices = (
        pd.merge(yes_df, no_df, on="timestamp", how="outer")
          .sort_values("timestamp")
          .reset_index(drop=True)
    )

    # IMPORTANT: if one side has missing ticks, forward-fill BEFORE hourly aggregation
    raw_prices["price_yes"] = raw_prices["price_yes"].ffill()
    raw_prices["price_no"] = raw_prices["price_no"].ffill()

    # Now create strict hourly prices (last tick per bucket)
    hourly_prices = _hourly_last_price(raw_prices, bucket_seconds=bucket_seconds)

    # -------------------
    # Volume: bucketed trades (working)
    # -------------------
    trades = get_all_trades(condition_id)
    vol_df = bucket_volume(trades, bucket_seconds=bucket_seconds)

    # Filter volume to window (bucket domain)
    start_bucket = (start_ts // bucket_seconds) * bucket_seconds
    end_bucket = (end_ts // bucket_seconds) * bucket_seconds
    vol_df = vol_df[(vol_df["timestamp"] >= start_bucket) & (vol_df["timestamp"] <= end_bucket)]

    # -------------------
    # Build continuous bucket grid
    # -------------------
    grid = pd.DataFrame({"timestamp": list(range(start_bucket, end_bucket + bucket_seconds, bucket_seconds))})

    out = (
        grid.merge(hourly_prices, on="timestamp", how="left")
            .merge(vol_df, on="timestamp", how="left")
            .sort_values("timestamp")
            .reset_index(drop=True)
    )

    out["volume"] = out["volume"].fillna(0.0)
    out["price_yes"] = out["price_yes"].ffill()
    out["price_no"] = out["price_no"].ffill()

    out_path = os.path.join("data", f"politics-{slug}.csv")
    out.to_csv(out_path, index=False)

    # Quick sanity prints (remove if you want)
    print("YES ticks:", len(yes_df), "NO ticks:", len(no_df), "trades:", len(trades))
    print("Hourly rows:", len(out), "Nonzero vol rows:", int((out["volume"] > 0).sum()))
    return out_path

if __name__ == "__main__":
    slugs = [
    "will-hezbollah-to-disarm-before-in-2025"

    ]

    failed = []

    # If you still hit interval limits, reduce price_chunk_days (e.g., 3)
    # or reduce days_back.
    for slug in slugs:
        try:
            path = export_slug_csv(
                slug,
                days_back=1000,
                bucket="1h",
                price_chunk_days=7,
            )
            print(f"Saved [{slug}]: {path}")
        except Exception as e:
            print(f"FAILED [{slug}]: {e}")
            failed.append((slug, str(e)))

    print("\n==== SUMMARY ====")
    if failed:
        print("The following slugs failed:")
        for slug, err in failed:
            print(f"- {slug}: {err}")
    else:
        print("All slugs exported successfully.")
