import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix

# attempting to train a RF classifier

folder = Path("etfs")
for CSV_PATH in folder.iterdir():
    WINDOW     = 5        # past 5 trading days ≈ one week
    TEST_RATIO = 0.2      # last 20% of samples become the test set
    SEED       = 42

    # Load the ETF and sort chronologically
    df = (
        pd.read_csv(CSV_PATH, parse_dates=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    feature_cols = ["Open", "High", "Low", "Close"]

    # 3. Transform into supervised‑learning format
    X, y_reg, y_cls, prev_close = [], [], [], []

    for idx in range(WINDOW, len(df) - 1):
        past_week        = df.loc[idx - WINDOW:idx - 1, feature_cols].values.flatten()
        next_day_ohlc    = df.loc[idx + 1, feature_cols].values
        today_close      = df.loc[idx, "Close"]

        X.append(past_week)
        y_reg.append(next_day_ohlc)
        y_cls.append(1 if next_day_ohlc[3] >= today_close else 0)  # green / red
        prev_close.append(today_close)

    X          = np.asarray(X)
    y_reg      = np.asarray(y_reg)
    y_cls      = np.asarray(y_cls)
    prev_close = np.asarray(prev_close)

    # 4. Time‑aware train / test split  (keep chronological order)
    
    cut        = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test           = X[:cut], X[cut:]
    y_train_reg, y_test_reg   = y_reg[:cut], y_reg[cut:]
    y_train_cls, y_test_cls   = y_cls[:cut], y_cls[cut:]
    prev_close_test           = prev_close[cut:]

    # 5. Fit a multi‑output Random‑Forest regressor
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=SEED
    )
    model.fit(X_train, y_train_reg)

    # 6. Evaluate regression & derived classification

    y_pred_reg = model.predict(X_test)
    mae        = mean_absolute_error(y_test_reg, y_pred_reg, multioutput="raw_values")

    predicted_cls      = (y_pred_reg[:, 3] >= prev_close_test).astype(int)
    acc                = accuracy_score(y_test_cls, predicted_cls)
    cm                 = confusion_matrix(y_test_cls, predicted_cls)

    print("Mean Absolute Error  [Open, High, Low, Close]:", mae.round(4))
    print(f"Trend accuracy (green/red): {acc:.3f}")
    print("Confusion matrix  (rows: true [red, green] | cols: pred [red, green])")
    print(cm)

    # preview = pd.DataFrame({
    #     "Prev_Close": prev_close_test[:10],
    #     "Pred_Close": y_pred_reg[:10, 3].round(2),
    #     "True_Close": y_test_reg[:10, 3],
    #     "Pred_Label": np.where(predicted_cls[:10] == 1, "green", "red"),
    #     "True_Label": np.where(y_test_cls[:10] == 1, "green", "red"),
    # })
    # print("\nSample predictions:\n", preview.to_string(index=False))
