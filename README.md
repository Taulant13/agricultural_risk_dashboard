# Agricultural Risk Dashboard

An interactive Plotly Dash application that isolates **Idiosyncratic Volatility (IVol)** in agricultural commodities and detects crisis regimes using a hybrid econometric and machine-learning pipeline.

The dashboard answers a simple question for each commodity:

> How much of today's price risk is *specific to this commodity*  weather, crop disease, logistics rather than just the broader market moving?

And then:

> Are we currently in a *crisis regime*, and what is the 30-day forward outlook for that specific risk?

---

## What the dashboard shows

- **Regime Analysis tab.** Total volatility vs. idiosyncratic volatility per commodity, on a single axis, with the chart background dynamically shaded green (calm) or red (crisis) based on detected market regime. Hover markers above the line highlight real-world events the model identified (e.g., Brazil coffee frost, US drought, COVID supply shocks).
- **Forecasting tab.** A 30-days forward forecast with two nested prediction-interval bands (50% inner, 80% outer) around the central forecast.
- **Sidebar controls.** Commodity selector and a date-range control (All / 20y / 10y / 5y) that zooms the regime analysis chart.

---

## Commodities and risk factors

**Agricultural basket (9 targets):** Corn, Soybean, Wheat, Cotton, Coffee, Oats, Sugar, Soybean Oil, Lumber.

**Risk factors (3 drivers):** S&P 500, WTI Crude Oil (energy / input costs), US Dollar Index.

---

## What was done

### Step 1 — Data collection and cleaning

- Downloaded daily historical prices for the 9 commodities and the 3 risk factors back to January 1986.
- Converted every asset to daily log returns.
- Merged all assets on Date so every row has a value for every asset.
- Applied data-quality fixes (see **Data** section below) and produced a single master dataset plus a correlation heatmap.

### Step 2 — Leave-one-out rolling regression

For each target commodity, the goal is to subtract out everything that can be explained by the broader market, leaving only what is specific to that commodity.

- Built a dynamic "rest of sector" index (RICI-weighted) that excludes the target itself, to avoid circular logic.
- Ran a 126-day rolling OLS regression:
- The residuals are the *idiosyncratic shocks*, the part of a commodity's return that the market, energy prices and the rest of the sector cannot explain.
- Computed **Idiosyncratic Volatility (IVol)** as the rolling standard deviation of those residuals.

### Step 3 — Regime detection with a Gaussian HMM

- Fit a two-state Gaussian Hidden Markov Model per commodity on the residual series.
- Labelled the higher-variance state as "Crisis."
- Exported both the posterior probability of crisis and the regime label for every day.
- Ran an event study: filtered for windows where `P(Crisis) ≥ 0.9` for at least 5 consecutive trading days, built a catalogue of real-world events (droughts, frosts, wars, supply chain shocks, policy changes), and matched HMM-detected crisis windows to events. Match rate: 261 of 264 windows.

### Step 4 — Forecasting with Prophet

- Trained Facebook Prophet on the IVol series.
- Generated a 30-business-day forward forecast per commodity.
- Produced **two prediction-interval bands** (80% and 50%), the outer band uses Prophet's built-in interval, the inner band is derived from Prophet's posterior samples.
- Persisted the forecasts to a pickle cache keyed on the input-data mtime, so the app starts in under a second after the first run instead of refitting nine models every time.

### Step 5 — Interactive dashboard

- Built with Plotly Dash and Dash Bootstrap Components.
- Two tabs wired to a shared sidebar. Figures are rebuilt from cached data, so commodity switching is effectively fast once the app is warm.
- Event annotations from the HMM event study are layered onto the regime chart as hoverable markers.

---

## Data

### Sources

| Asset | Source |
|---|---|
| 8 of 9 agricultural commodities | Macrotrends: manually downloaded CSVs |
| Soybean Oil extension | yfinance, ticker `ZL=F` |
| US Dollar Index | yfinance, ticker `DX-Y.NYB` |
| S&P 500 | yfinance, ticker `^GSPC` |
| WTI Crude Oil | FRED St. Louis Fed, series `DCOILWTICO` |

All Macrotrends files arrived in `MM/DD/YYYY` and were converted to `YYYY-MM-DD`. All rows prior to 1986-01-02 were dropped to establish a common start date across every asset.

### Known data issues

- **Lumber contract transition, May–June 2023.** CME replaced the legacy `LBS` contract with a physical `LBR` contract, leaving 25 flat-price days and a +44% jump at the rollover. These rows were corrected in the cleaning step.
- **Oats flat-price stretch, July 2009 – May 2011.** 450 days of identical prices from the vendor. Not fixed in the dataset; the Oats regime labels are therefore unreliable around this period.
- **Simultaneous flat run in Corn / Soybean / Wheat / Soybean Oil, October 2013.** 13 days of identical prices probably a vendor glitch, not fixed.
- **Cotton 1986 and Coffee 1986–88.** Multiple long flat stretches in the earliest years.
- **Negative WTI price, 20 April 2020.** Real market event where futures settled at around –$37 during the COVID demand collapse. Preserved in the dataset.
- **Data cutoff: 2026-03-06.** Nothing after that date is in the model. This is the single most important limitation see below.

---

## Limitations

- **Data cutoff.** The entire analysis is anchored to the last row of the input data (2026-03-06 by default, the dashboard displays the active cutoff next to the title). The 30-day forecast window is *relative to the cutoff*, not to the current date. Refreshing the input data and re-running the pipeline regenerates the forecasts.
- **Lumber post-2020 regime.** Roughly 40% of trading days after 2020 are flagged as crisis. This is not a bug it reflects genuine structural volatility and illiquidity in the post-pandemic lumber market.
- **Oats regime labels.** The unfixed 2009–2011 flat-price artifact makes Oats regime labels around that window unreliable.

---

## Reproduce

Tested with **Python 3.11**. The fastest path from clone to a running dashboard:

```bash
git clone <this-repo>
cd agricultural_risk_dashboard

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Then open [http://127.0.0.1:8050](http://127.0.0.1:8050) in a browser.

The repository ships with all processed data and a pre-built Prophet forecast cache (`data/processed/prophet_cache.pkl`), so no model refitting.

### Notebooks

The `notebooks/` directory contains the three analytical notebooks that produced the processed data:

- `phase1_data_preparation.ipynb`: data loading, cleaning, log returns, correlation heatmap
- `phase2_model_construction.ipynb`: rolling leave-one-out regression, residual extraction, IVol computation
- `phase3_regime_detection.ipynb`: Gaussian HMM fitting per commodity, event study, annotation

**You do not need to run these to use the dashboard**: all their outputs are already committed under `data/processed/`. They are included for transparency: to show the methodology end-to-end and the data-quality decisions behind the numbers the dashboard displays.

---

## Context

This was a semester project completed in my studies, in collaboration with an industry partner.
