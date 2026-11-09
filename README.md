# GEX Dashboard

This is the dashboard created in this video
https://youtu.be/3tWRSCsSQ1I

A Streamlit dashboard that estimates options **gamma exposure (GEX)** for an index, ETF or stock, using CBOE's free delayed quotes (about 15 minutes behind).

## Features

- Net GEX, the zero-gamma flip, the call wall and the put wall
- GEX by strike, by expiration, and a gamma profile across a range of prices
- Price chart (intraday or daily) with the key GEX levels drawn on it, plus moving averages, volume, MACD and RSI
- Open interest and volume by strike, with put/call ratios
- Filters for expirations and the strike window

## Run it

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501. Enter a ticker in the sidebar, for example `SPX`, `NDX`, `SPY`, `QQQ` or `AAPL`.

## How GEX is calculated

For each contract with open interest:

```
GEX = gamma × open interest × 100 × spot² × 0.01 × (+1 for calls, −1 for puts)
```

The result is dealers' dollar gamma for a 1% move. Gamma is recomputed with Black-Scholes from each contract's implied volatility, because CBOE rounds gamma to 4 decimals. The calculation uses a 4% rate and no dividends. SPX/NDX/RUT monthlies settle at the 9:30 ET open; everything else settles at 16:00 ET.

The zero-gamma flip is the price where total GEX, recomputed across a ±15% price range, crosses zero.

## Caveats

- **Estimates, not facts.** GEX assumes dealers are long calls and short puts. Real dealer positioning isn't public.
- **Open interest is as of the prior close**, because OCC updates it overnight. Volume is the latest session's.
- **The data source is unofficial.** CBOE's public JSON feed isn't a documented API, and its terms are geared to personal, non-commercial use. It can change or stop without notice. Check CBOE's terms before hosting this publicly or using it commercially.
- Not investment advice.
