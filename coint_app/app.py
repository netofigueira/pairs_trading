from flask import Flask, render_template, request
from pair_model import PairCointegration
import plotly.graph_objs as go
import plotly
import json

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    plot_divs = []
    error = None

    if request.method == "POST":
        ticker1 = request.form.get("ticker1")
        ticker2 = request.form.get("ticker2")

        try:
            pair = PairCointegration(ticker1, ticker2)
            pair.cointegration_model()
            pair._cointegration_test()

            # --- Gráfico 1: Resíduo Normalizado ---
            trace1 = go.Scatter(
                x=pair.data.index,
                y=pair.norm_resid.to_list(),
                name="Resíduo Normalizado",
                line=dict(color="royalblue")
            )
            layout1 = go.Layout(
                title="Resíduo Normalizado",
                shapes=[
                    dict(type='line', x0=pair.data.index[0], x1=pair.data.index[-1], y0=0, y1=0, line=dict(dash='dash', color='black')),
                    dict(type='line', x0=pair.data.index[0], x1=pair.data.index[-1], y0=2, y1=2, line=dict(dash='dash', color='gray')),
                    dict(type='line', x0=pair.data.index[0], x1=pair.data.index[-1], y0=-2, y1=-2, line=dict(dash='dash', color='gray'))
                ],
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(family='Arial', size=12, color='black'),
                margin=dict(l=50, r=50, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=True, gridcolor='lightgrey'),
                yaxis=dict(showgrid=True, gridcolor='lightgrey'),
            )
            fig1 = go.Figure(data=[trace1], layout=layout1)

            # --- Gráfico 2: Preços Normalizados ---
            p1_norm = pair.pair1_price_series / pair.pair1_price_series.iloc[0] * 100
            p2_norm = pair.pair2_price_series / pair.pair2_price_series.iloc[0] * 100
            trace2_1 = go.Scatter(x=p1_norm.index, y=p1_norm.to_list(), name=ticker1)
            trace2_2 = go.Scatter(x=p2_norm.index, y=p2_norm.to_list(), name=ticker2)
            fig2 = go.Figure(data=[trace2_1, trace2_2], layout=layout1)

            # --- Gráfico 3: RSI ---
            trace3_1 = go.Scatter(x=pair.pair1_rsi.index, y=pair.pair1_rsi.to_list(), name=f"RSI {ticker1}")
            trace3_2 = go.Scatter(x=pair.pair2_rsi.index, y=pair.pair2_rsi.to_list(), name=f"RSI {ticker2}")
            layout3 = go.Layout(
                title="RSI (Índice de Força Relativa)",
                shapes=[
                    dict(type='line', x0=pair.data.index[0], x1=pair.data.index[-1], y0=70, y1=70, line=dict(dash='dash', color='red')),
                    dict(type='line', x0=pair.data.index[0], x1=pair.data.index[-1], y0=30, y1=30, line=dict(dash='dash', color='green'))
                ],
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(family='Arial', size=12, color='black'),
                margin=dict(l=50, r=50, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=True, gridcolor='lightgrey'),
                yaxis=dict(showgrid=True, gridcolor='lightgrey'),
            )
            fig3 = go.Figure(data=[trace3_1, trace3_2], layout=layout3)

            plot_divs = [
                json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder),
                json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder),
                json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder),
            ]

        except Exception as e:
            error = str(e)

    return render_template("index.html", plot_divs=plot_divs, error=error)

if __name__ == "__main__":
    app.run(debug=True)