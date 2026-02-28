#took this straight from my ML project i did in the fall.


import pandas as pd
import yfinance as yf
import numpy as np
import time
import random
import warnings
from pathlib import Path
from datetime import datetime


warnings.simplefilter(action="ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "Data"
DOWNLOAD_DIR = DATA_DIR / "Pulling"

def run_sp500_pipeline(start_date, end_date):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    symbols = pd.read_feather("/Users/lukeromes/Desktop/SP500Project/Data/FINALDATA.FEATHER")
    tickers = symbols["Ticker"].unique().astype(str).tolist()

    for i, ticker in enumerate(tickers):
        if ticker in ["IPG", "BRK.B", "BF.B"]:
            continue

        if (i + 1) % 50 == 0:
            time.sleep(random.uniform(20, 30))

        try:
            data = yf.download(
                ticker,
                start=start_date,
                #end=end_date,
                interval="1d",
                progress=True
            )

            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data["Ticker"] = ticker
                data.to_csv(DOWNLOAD_DIR / f"{ticker}.csv")

        except Exception:
            pass

    files = [f for f in DOWNLOAD_DIR.iterdir() if f.suffix == ".csv"]

    frames = []
    for f in files:
        df = pd.read_csv(f)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["Close"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values(["Ticker", "Date"])

    g = df.groupby("Ticker")

    df["Daily_Return"] = g["Close"].pct_change() * 100
    df["Day_Range"] = df["High"] - df["Low"]

    for w in [5, 12, 15, 20, 26, 30, 50]:
        df[f"SMA_{w}"] = g["Close"].transform(lambda x: x.rolling(w, 1).mean())

    for w in [5, 12, 26, 50]:
        df[f"EMA_{w}"] = g["Close"].transform(lambda x: x.ewm(span=w, adjust=False).mean())

    df["MACD"] = df["EMA_12"] - df["EMA_26"]

    def rsi(s, n=14):
        d = s.diff()
        g = d.where(d > 0, 0).rolling(n).mean()
        l = -d.where(d < 0, 0).rolling(n).mean()
        return 100 - (100 / (1 + g / l))

    df["RSI"] = g["Close"].transform(rsi)

    df["Bollinger_Band_Middle"] = df["SMA_20"]
    df["SD"] = g["Close"].transform(lambda x: x.rolling(20, 1).std())
    df["Bollinger_Band_Upper"] = df["Bollinger_Band_Middle"] + 2 * df["SD"]
    df["Bollinger_Band_Lower"] = df["Bollinger_Band_Middle"] - 2 * df["SD"]

    df["OBV"] = (
        g.apply(lambda x: (np.sign(x["Close"].diff()).fillna(0) * x["Volume"]).cumsum(), include_groups=False)
        .reset_index(level=0, drop=True)
    )

    try:
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=True)["Close"].reset_index()
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix.columns = ["Date", "VIX_Close"]
        vix["Date"] = pd.to_datetime(vix["Date"]).dt.tz_localize(None)
        df = df.merge(vix, on="Date", how="left")
    except Exception:
        df["VIX_Close"] = np.nan

    df["next_day_pct_change"] = (g["Close"].shift(-1) - df["Close"]) / df["Close"] * 100
    df["Movement"] = (df["next_day_pct_change"] > 0).astype(int)

    df["next_5_day_pct_change"] = (g["Close"].shift(-5) - df["Close"]) / df["Close"] * 100
    df["Movement_5_day"] = (df["next_5_day_pct_change"] > 0).astype(int)

    df["next_30_day_pct_change"] = (g["Close"].shift(-30) - df["Close"]) / df["Close"] * 100
    df["Movement_30_day"] = (df["next_30_day_pct_change"] > 0).astype(int)

    df["earnings_bool"] = 0
    df["Split_Indicator"] = 0
    df["Ticker"] = df["Ticker"].astype(str)
    df = df.reset_index(drop=True)

    df.to_feather("/Users/lukeromes/Desktop/NewsScrapingSentiment/FINALSP500Data.feather")

    return df
