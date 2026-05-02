from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

DATA_DIR = Path(__file__).parent / "data" / "processed"
TRADING_DAYS = 252

_vol = pd.read_csv(DATA_DIR / "volatility.csv", parse_dates=["Date"])
_states = pd.read_csv(DATA_DIR / "hmm_viterbi_states.csv", parse_dates=["Date"])
_prob = pd.read_csv(DATA_DIR / "hmm_crisis_probabilities.csv", parse_dates=["Date"])
_annot = pd.read_csv(DATA_DIR / "crisis_periods_annotated.csv", parse_dates=["Start", "End"])

_DATE_RANGE_YEARS = {"5y": 5, "10y": 10, "20y": 20}


def _series(commodity):
    vol = _vol[["Date", f"{commodity}_IVol", f"{commodity}_TotalVol"]].rename(
        columns={f"{commodity}_IVol": "IVol", f"{commodity}_TotalVol": "TotalVol"}
    )
    states = _states[["Date", f"{commodity}_State"]].rename(
        columns={f"{commodity}_State": "State"}
    )
    prob = _prob[["Date", f"{commodity}_CrisisProb"]].rename(
        columns={f"{commodity}_CrisisProb": "P_Crisis"}
    )
    return vol.merge(states, on="Date").merge(prob, on="Date")


def _crisis_intervals(df):
    state = df["State"].astype(int).to_numpy()
    if state.size == 0:
        return []
    changes = np.diff(state, prepend=state[0] ^ 1)
    starts = np.where((changes != 0) & (state == 1))[0]
    ends = []
    for s in starts:
        k = s
        while k + 1 < state.size and state[k + 1] == 1:
            k += 1
        ends.append(k)
    dates = df["Date"].to_numpy()
    return [(pd.Timestamp(dates[s]), pd.Timestamp(dates[e])) for s, e in zip(starts, ends)]


def build_regime_figure(commodity, date_range="all"):
    df = _series(commodity).dropna(subset=["IVol", "TotalVol", "State"])
    if date_range in _DATE_RANGE_YEARS:
        cutoff = df["Date"].max() - pd.DateOffset(years=_DATE_RANGE_YEARS[date_range])
        df = df[df["Date"] >= cutoff].copy()

    scale = np.sqrt(TRADING_DAYS)
    ivol = df["IVol"] * scale
    total_vol = df["TotalVol"] * scale

    fig = go.Figure()

    if not df.empty:
        fig.add_vrect(
            x0=df["Date"].min(), x1=df["Date"].max(),
            fillcolor="#50c878", line_width=0, layer="below",
        )
    for start, end in _crisis_intervals(df):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="#ff4949", line_width=0, layer="below",
        )

    state_labels = np.where(df["State"].to_numpy() == 1, "Crisis", "Normal")
    p_crisis_pct = df["P_Crisis"].to_numpy() * 100
    ivol_meta = [
        f"Regime: {s}<br>P(Crisis): {p:.0f}%"
        for s, p in zip(state_labels, p_crisis_pct)
    ]

    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=total_vol,
            mode="lines", name="Total Vol",
            line=dict(color="#444444", width=1.2),
            hovertemplate="Total Vol: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=ivol,
            mode="lines", name="IVol",
            line=dict(color="#1f77b4", width=1.6),
            customdata=ivol_meta,
            hovertemplate="IVol: %{y:.1%}<br>%{customdata}<extra></extra>",
        )
    )

    annot = _annot[_annot["Commodity"] == commodity]
    if not annot.empty and not df.empty:
        annot = annot[annot["End"] >= df["Date"].min()]
    if not annot.empty:
        mid_dates = annot["Start"] + (annot["End"] - annot["Start"]) / 2
        y_max = float(max(ivol.max(), total_vol.max()))
        marker_y = [y_max * 1.03] * len(annot)
        text = [
            f"<b>{r.Event_Name}</b><br>{r.Category}, {r.Strength}<br>"
            f"{r.Start.date()} to {r.End.date()} ({int(r.Duration_Days)}d)"
            for r in annot.itertuples()
        ]
        fig.add_trace(
            go.Scatter(
                x=mid_dates, y=marker_y,
                mode="markers",
                marker=dict(symbol="diamond", size=7, color="#d62728",
                            line=dict(color="white", width=1)),
                name="Event",
                hovertemplate="%{text}<extra></extra>",
                text=text,
            )
        )

    fig.update_layout(
        title=f"{commodity}: IVol vs Total Vol with HMM regimes",
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
