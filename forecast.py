import logging
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

from prophet import Prophet  # noqa: E402

DATA_DIR = Path(__file__).parent / "data" / "processed"
TRADING_DAYS = 252
HORIZON_DAYS = 30
CI_OUTER = 0.80
CI_INNER = 0.50
HISTORY_YEARS = 3

COMMODITIES = [
    "Corn", "Soybean", "Wheat", "Cotton", "Coffee",
    "Oats", "Sugar", "Soybean_Oil", "Lumber",
]

_VOL_PATH = DATA_DIR / "volatility.csv"
_CACHE_PATH = DATA_DIR / "prophet_cache.pkl"

_vol = pd.read_csv(_VOL_PATH, parse_dates=["Date"])
_CACHE = {}


def _series(commodity):
    s = _vol[["Date", f"{commodity}_IVol"]].dropna()
    s = s[s[f"{commodity}_IVol"] > 0]
    return pd.DataFrame({"ds": s["Date"].values, "y": np.log(s[f"{commodity}_IVol"].values)})


def _history(commodity):
    h = _vol[["Date", f"{commodity}_IVol"]].dropna().copy()
    h["IVol"] = h[f"{commodity}_IVol"] * np.sqrt(TRADING_DAYS)
    return h[["Date", "IVol"]]


def _fit_and_forecast(commodity):
    series = _series(commodity)
    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_mode="additive",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=CI_OUTER,
    )
    model.fit(series)

    future = model.make_future_dataframe(periods=HORIZON_DAYS, freq="B")
    raw = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    samples = model.predictive_samples(future)["yhat"]
    inner_alpha = (1.0 - CI_INNER) / 2.0
    lower_inner = np.quantile(samples, inner_alpha, axis=1)
    upper_inner = np.quantile(samples, 1.0 - inner_alpha, axis=1)

    scale = np.sqrt(TRADING_DAYS)
    return pd.DataFrame(
        {
            "ds": raw["ds"].values,
            "yhat": np.exp(raw["yhat"].values) * scale,
            "yhat_lower_80": np.exp(raw["yhat_lower"].values) * scale,
            "yhat_upper_80": np.exp(raw["yhat_upper"].values) * scale,
            "yhat_lower_50": np.exp(lower_inner) * scale,
            "yhat_upper_50": np.exp(upper_inner) * scale,
        }
    )


def warmup_all():
    if _CACHE_PATH.exists() and _CACHE_PATH.stat().st_mtime >= _VOL_PATH.stat().st_mtime:
        with open(_CACHE_PATH, "rb") as f:
            _CACHE.update(pickle.load(f))
        print(f"loaded {len(_CACHE)} cached forecasts")
        return

    print("no valid cache, refitting...")
    t0 = time.time()
    for c in COMMODITIES:
        t = time.time()
        _CACHE[c] = {"forecast": _fit_and_forecast(c), "history": _history(c)}
        print(f"  {c}: {time.time() - t:.1f}s")
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(_CACHE, f)
    print(f"warmup done in {time.time() - t0:.1f}s")


def build_forecast_figure(commodity):
    history = _CACHE[commodity]["history"]
    forecast = _CACHE[commodity]["forecast"]

    last_hist = history["Date"].max()
    window_start = last_hist - pd.DateOffset(years=HISTORY_YEARS)

    hist_view = history[history["Date"] >= window_start]
    fc_view = forecast[forecast["ds"] >= window_start]
    future_mask = fc_view["ds"] > last_hist

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fc_view["ds"], y=fc_view["yhat_upper_80"],
            mode="lines", line=dict(width=0),
            hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_view["ds"], y=fc_view["yhat_lower_80"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(214, 39, 40, 0.10)",
            hoverinfo="skip", showlegend=True,
            name=f"{int(CI_OUTER * 100)}% CI",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_view["ds"], y=fc_view["yhat_upper_50"],
            mode="lines", line=dict(width=0),
            hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_view["ds"], y=fc_view["yhat_lower_50"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(214, 39, 40, 0.26)",
            hoverinfo="skip", showlegend=True,
            name=f"{int(CI_INNER * 100)}% CI",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=hist_view["Date"], y=hist_view["IVol"],
            mode="lines", name="Historical IVol",
            line=dict(color="#1f77b4", width=1.6),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>IVol: %{y:.1%}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=fc_view.loc[future_mask, "ds"],
            y=fc_view.loc[future_mask, "yhat"],
            mode="lines", name="Forecast (Prophet)",
            line=dict(color="#d62728", width=2.0),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Forecast: %{y:.1%}<extra></extra>",
        )
    )

    now_iso = last_hist.strftime("%Y-%m-%d")
    fig.add_shape(
        type="line", x0=now_iso, x1=now_iso, xref="x",
        y0=0, y1=1, yref="paper",
        line=dict(color="#888888", width=1, dash="dash"),
    )
    fig.add_annotation(
        x=now_iso, xref="x", y=1.0, yref="paper", yanchor="bottom",
        text="now", showarrow=False,
        font=dict(color="#888888", size=11),
    )

    fig.update_layout(
        title=f"{commodity}: {HORIZON_DAYS}-day IVol forecast (Prophet), last {HISTORY_YEARS}y history",
        xaxis_title="Date",
        yaxis_title="Annualized volatility",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=80, b=50),
        template="plotly_white",
        height=520,
    )
    return fig
