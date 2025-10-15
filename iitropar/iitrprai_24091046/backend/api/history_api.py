import os
import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter()

# Paths
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(_CURRENT_DIR, "..", ".."))

default_ai_ml_dir = os.path.join(project_root, "AI_ML")
DEFAULT_MODEL_DIR = os.path.join(default_ai_ml_dir, "models_store")

MODEL_DIR = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)
DATA_PATH = os.path.join(MODEL_DIR, "master_data.csv")

@router.get("/history/{stock}")
def stock_history(stock: str, days: int = 60):
    """
    Return the last `days` rows of OHLCV data for a given stock
    from master_data.csv. Defaults to last 60 rows.
    """

    symbol = stock.strip().upper()

    # 1) Load CSV
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="master_data.csv not found.")

    df = pd.read_csv(DATA_PATH)

    # 2) Validate columns
    required_cols = {"date", "symbol", "open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        raise HTTPException(
            status_code=500,
            detail=f"master_data.csv must contain columns: {sorted(required_cols)}"
        )

    # 3) Parse date and filter symbol
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df_sym = df[df["symbol"].str.upper() == symbol].sort_values("date")

    if df_sym.empty:
        raise HTTPException(status_code=404, detail=f"No rows for {symbol} in master_data.csv")

    # 4) Get last `days` rows
    df_last = df_sym.tail(days)

    # 5) Return JSON response
    return {
        "stock": symbol,
        "days": days,
        "history": [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "close": round(float(row["close"]), 2)
               
            }
            for _, row in df_last.iterrows()
        ]
    }

@router.get("/stocklist")
def stock_list():
    """
    Return a list of all unique stock symbols in master_data.csv.
    """

    # Load CSV
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="master_data.csv not found.")

    df = pd.read_csv(DATA_PATH)

    if "symbol" not in df.columns:
        raise HTTPException(status_code=422, detail="master_data.csv must contain 'symbol' column.")

    # Get unique symbols
    stocks = sorted(df["symbol"].dropna().unique())

    return {
        "count": len(stocks),
        "stocks": stocks
    }