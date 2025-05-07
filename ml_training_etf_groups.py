import pandas as pd, numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from ta.momentum  import RSIIndicator, StochasticOscillator
from ta.trend     import MACD
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

CSV_DIR   = Path(r"C:\Users\ChanA\OneDrive\Documents\CS4774\etfs")
SYMBOLS   = ["AAAU", "AADR", "AAXJ", "ABEQ", "ACES", "ACIO"]
LAG_DAYS  = 15        
SEED      = 42
N_SPLITS  = 5

def engineer(df):
    df = df.sort_values("Date").reset_index(drop=True)
    df["ret_1"]  = df["Close"].pct_change()
    df["ret_5"]  = df["Close"].pct_change(5)
    df["ret_10"] = df["Close"].pct_change(10)

    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["oc_range"] = (df["Close"] - df["Open"]) / df["Open"]

    df["vol_5"]  = np.log1p(df["ret_1"]).rolling(5).std()
    df["vol_15"] = np.log1p(df["ret_1"]).rolling(15).std()

    for n in (5, 10, 20):
        df[f"sma_{n}"] = df["Close"].rolling(n).mean()
        df[f"ema_{n}"] = df["Close"].ewm(span=n, adjust=False).mean()

    rsi = RSIIndicator(df["Close"], window=14).rsi()
    df["rsi_14"] = rsi
    df["rsi_7"]  = rsi.rolling(7).mean()

    df["macd_diff"] = MACD(df["Close"]).macd_diff()

    bb = BollingerBands(df["Close"])
    df["bb_bw"] = bb.bollinger_hband() - bb.bollinger_lband()

    df["stoch_k"] = StochasticOscillator(df["High"], df["Low"], df["Close"]).stoch()
    return df.dropna().reset_index(drop=True)

frames = []
for sid, sym in enumerate(SYMBOLS):
    df = pd.read_csv(CSV_DIR / f"{sym}.csv", parse_dates=["Date"])
    df = engineer(df)
    df["symbol_id"] = sid
    frames.append(df)

data = (pd.concat(frames, ignore_index=True)
          .sort_values("Date")
          .reset_index(drop=True))

feat_cols = [c for c in data.columns if c not in
             ("Date", "Open", "High", "Low", "Adj Close", "Close", "Volume")]

X_rows, y_rows = [], []
for i in range(LAG_DAYS, len(data) - 1):          
    X_rows.append(data.loc[i-LAG_DAYS+1:i, feat_cols].values.flatten())
    y_rows.append(int(data.loc[i+1, "Close"] > data.loc[i, "Close"]))

X = np.asarray(X_rows, dtype=np.float32)
y = np.asarray(y_rows,  dtype=np.int8)

scaler = StandardScaler()
model  = XGBClassifier(
    n_estimators=1200,
    learning_rate=0.02,
    max_depth=3,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.5,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=-1,
)

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
scores = []
for fold, (tr, te) in enumerate(tscv.split(X), 1):
    X_tr = scaler.fit_transform(X[tr]); X_te = scaler.transform(X[te])
    model.fit(X_tr, y[tr])
    preds = (model.predict_proba(X_te)[:,1] > 0.5).astype(int)
    acc = accuracy_score(y[te], preds)
    scores.append(acc)
    print(f"fold {fold}: acc = {acc:.3f}")

print(f"\nmean accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

scaler.fit(X)
model.fit(scaler.transform(X), y)

last_prob = model.predict_proba(scaler.transform(X[-1:]))[0,1]
print(f"\nNext‑day GREEN probability: {last_prob:.1%}")