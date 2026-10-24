# Bitcoin / Kalshi 15-Minute Lab

## Run on Windows

1. Install Python 3.10 or newer from python.org (check **Add Python to PATH**).
2. Extract this folder, open PowerShell in it.
3. Run `py -m pip install -r requirements.txt`
4. Run `py -m streamlit run app.py`
5. Your browser should open to a local dashboard, usually `http://localhost:8501`.

## What is live?
Coinbase Exchange BTC-USD ticker and historical one-minute candles are fetched via its public REST API. Click **Refresh live data** to update. This is not a continuously streaming WebSocket and may be delayed. Coinbase can be inaccessible from some networks.

## What is manual?
Kalshi's target, contract ask prices, fees, and settlement time. Enter the exact current market terms from Kalshi. The Coinbase spot feed is **not** necessarily Kalshi's settlement source.

## Backtest limits
The historical walk-forward test uses the first 1-minute candle close as a **proxy target**, predicts 5 minutes into a window, and uses the 15th minute close as a **proxy settlement**. It avoids using future returns to estimate volatility, but it is not a real Kalshi market backtest. It cannot establish profitable trading without actual historical market targets, executable contract prices, fee schedule, and settlement rules. No historical Kalshi prices are invented.

## CSV trade log format
`outcome,won,ask_cents,fee_cents,contracts`

Example:
`UP,1,55,2,1`
`DOWN,0,38,2,1`

All costs are in cents **per contract**; net payout is $1 per winning contract.
