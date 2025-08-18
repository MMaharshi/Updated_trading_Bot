# ------- train_model.py  ---------
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from xgboost import XGBClassifier
import pickle

# === CONFIGURATION ===
UNIVERSE = ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT",
            "ADA/USDT","DOGE/USDT","MATIC/USDT","LTC/USDT","DOT/USDT"]
TIMEFRAME        = "5m"
TRAIN_PERIOD_DAYS= 90
LABEL_LOOKAHEAD  = 6      # 6 candles = 30 min
LABEL_THRESHOLD  = 0.01   # +1%

# ----
exchange = ccxt.binance({"enableRateLimit":True})

def fetch_df(symbol, start_ts):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=start_ts, limit=5000)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def add_indicators(df):
    # Supertrend (ATR 14, multiplier 1.4)
    atr = AverageTrueRange(df['high'],df['low'],df['close'],window=14).average_true_range()
    hl2 = (df['high']+df['low'])/2
    upper = hl2 + 1.4*atr
    lower = hl2 - 1.4*atr
    st = []
    trend_up=True
    for i in range(len(df)):
        if i==0: st.append(lower.iloc[i])
        else:
            if df['close'].iloc[i] > upper.iloc[i-1]:
                trend_up=True
            elif df['close'].iloc[i] < lower.iloc[i-1]:
                trend_up=False
            st.append(lower.iloc[i] if trend_up else upper.iloc[i])
    df['supertrend'] = st

    # RSI + SMA
    df['rsi'] = RSIIndicator(df['close'], window=21).rsi()
    df['rsi_sma'] = df['rsi'].rolling(16).mean()
    return df

def build_features(df):
    df['cross_up'] = (df['close'].shift(1) <= df['supertrend'].shift(1)) & (df['close'] > df['supertrend'])
    df['bullish']  = df['close']>df['open']
    df['vol_up']   = df['volume']>df['volume'].shift(1)
    df['break_ph'] = df['close']>df['high'].shift(1)
    df['rsi_gt']   = (df['rsi']>df['rsi_sma'])
    return df

def create_label(df):
    future = df['close'].shift(-LABEL_LOOKAHEAD)
    pct = (future/df['close'])-1
    df['label'] = (pct > LABEL_THRESHOLD).astype(int)
    return df

# ---------- COLLECT DATA ----------
start_ts = int((datetime.utcnow() - timedelta(days=TRAIN_PERIOD_DAYS)).timestamp()*1000)
frames=[]

for sym in UNIVERSE:
    raw = fetch_df(sym,start_ts)
    raw = add_indicators(raw)
    raw = build_features(raw)
    raw = create_label(raw)
    raw['symbol'] = sym
    frames.append(raw)

full_df = pd.concat(frames, ignore_index=True).dropna()
features = ['cross_up','bullish','vol_up','break_ph','rsi_gt']
X = full_df[features].values
y = full_df['label'].values

# ---------- TRAIN ----------
model = XGBClassifier(use_label_encoder=False,eval_metric='logloss')
model.fit(X,y)
# ---------- SAVE ----------
with open("model.pkl","wb") as f:
    pickle.dump(model,f)

print("Training complete. Model saved to model.pkl")
