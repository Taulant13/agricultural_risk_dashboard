from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, Input, Output, dcc, html

from forecast import build_forecast_figure, warmup_all
from regime import build_regime_figure

DATA_DIR = Path(__file__).parent / "data" / "processed"

COMMODITIES = [
    "Corn", "Soybean", "Wheat", "Cotton", "Coffee",
    "Oats", "Sugar", "Soybean_Oil", "Lumber",
]

DATE_RANGE_OPTIONS = [
    {"label": "All", "value": "all"},
    {"label": "Last 20y", "value": "20y"},
    {"label": "Last 10y", "value": "10y"},
    {"label": "Last 5y", "value": "5y"},
]

data_cutoff = (
    pd.read_csv(DATA_DIR / "volatility.csv", parse_dates=["Date"])["Date"]
    .max()
    .strftime("%Y-%m-%d")
)

print("warming prophet cache...")
warmup_all()

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Agricultural Risk Dashboard"

sidebar = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Controls", className="card-title"),
            html.Label("Commodity", className="fw-semibold"),
            dcc.Dropdown(
                id="commodity-dropdown",
                options=[{"label": c.replace("_", " "), "value": c} for c in COMMODITIES],
                value="Corn",
                clearable=False,
            ),
            html.Br(),
            html.Label("Date range", className="fw-semibold"),
            dcc.RadioItems(
                id="date-range-radio",
                options=DATE_RANGE_OPTIONS,
                value="all",
                labelClassName="me-3",
                inputClassName="me-1",
            ),
            html.Div(
                "Applies to the Regime Analysis tab. IVol uses a fixed 126-day window.",
                className="text-muted small mt-3",
            ),
        ]
    )
)

graph_config = {"scrollZoom": True, "displaylogo": False}

main_panel = dbc.Tabs(
    [
        dbc.Tab(dcc.Graph(id="regime-figure", config=graph_config), label="Regime Analysis"),
        dbc.Tab(dcc.Graph(id="forecast-figure", config=graph_config), label="Forecasting"),
    ]
)

app.layout = dbc.Container(
    [
        html.Div(
            [
                html.H1("Agricultural Risk Dashboard", className="mb-0"),
                html.Span(f"Data cutoff: {data_cutoff}", className="text-muted small"),
            ],
            className="d-flex justify-content-between align-items-baseline my-3",
        ),
        dbc.Row(
            [
                dbc.Col(sidebar, width=3),
                dbc.Col(main_panel, width=9),
            ]
        ),
    ],
    fluid=True,
)


@app.callback(
    Output("regime-figure", "figure"),
    Input("commodity-dropdown", "value"),
    Input("date-range-radio", "value"),
)
def update_regime_figure(commodity, date_range):
    return build_regime_figure(commodity, date_range=date_range)


@app.callback(
    Output("forecast-figure", "figure"),
    Input("commodity-dropdown", "value"),
)
def update_forecast_figure(commodity):
    return build_forecast_figure(commodity)


if __name__ == "__main__":
    app.run()
