import pandas as pd
import plotly.express as px
import numpy as np


def loss_to_plot(current_losses, epochs, benchmark):
    loss_df = pd.DataFrame({
        "Epoch": list(np.array(current_losses[0]) + 1),
        "Loss": current_losses[1],
        "Loss shift": list(np.array(current_losses[1]) - 10),
        "Energy": current_losses[2],
        "Energy shift": list(np.array(current_losses[2]) - 10),
    })

    if benchmark:
        if benchmark[1]:
            lower_value = benchmark[0] - benchmark[1] - 0.2  # Padding
        else:
            lower_value = benchmark[0] - 0.2  # Padding
    else:
        lower_value = -3  # Standard

    current_fig = px.line(loss_df, x="Epoch", y=["Loss", "Energy"],
                          color_discrete_map={"Energy": "black"})
    current_fig_zoom = px.line(loss_df, x="Epoch", y=["Loss", "Energy"],
                               color_discrete_map={"Energy": "black"})

    if benchmark:
        if benchmark[1]:
            current_fig.add_hline(y=float(benchmark[0]),
                                  line=dict(dash="solid",
                                            color="black",
                                            width=1),
                                  label=dict(text="Benchmark \n (Data)", font=dict(size=10)))
            current_fig_zoom.add_hline(y=float(benchmark[0]),
                                       line=dict(dash="solid",
                                                 color="black",
                                                 width=1),
                                       label=dict(text="Benchmark \n (Data)", font=dict(size=10)))
        else:
            current_fig.add_hline(y=float(benchmark[0]),
                                  line=dict(dash="dot",
                                            color="black",
                                            width=1),
                                  label=dict(text="Benchmark \n (Estimate)", font=dict(size=10)))
            current_fig_zoom.add_hline(y=float(benchmark[0]),
                                       line=dict(dash="dot",
                                                 color="black",
                                                 width=1),
                                       label=dict(text="Benchmark \n (Estimate)", font=dict(size=10)))

    current_fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#ececec',
                   range=[1, epochs], dtick=epochs / 10, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor='#ececec',
                   range=[lower_value, 0], dtick=0.2, tickfont=dict(size=10)),
        yaxis_title=None,
        xaxis_title=dict(font=dict(size=12)),
        margin=dict(l=50, r=50, t=20, b=20),
        legend=dict(title="Metrics", orientation="h", font=dict(size=10)),
        showlegend=True,
        hovermode="x unified"
    )
    current_fig.update_traces(
        # mode="markers+lines",
        hovertemplate=None,
        line=dict(shape='spline', smoothing=1.3),
    )

    current_fig_zoom.update_layout(
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#ececec', tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor='#ececec', tickfont=dict(size=10)),
        yaxis_title=None,
        xaxis_title=dict(font=dict(size=12)),
        margin=dict(l=50, r=50, t=20, b=20),
        legend=dict(title="Metrics", orientation="h", font=dict(size=10)),
        showlegend=True,
        hovermode="x unified"
    )
    current_fig_zoom.update_traces(
        # mode="markers+lines",
        hovertemplate=None,
        line=dict(shape='spline', smoothing=1.3),
    )

    return current_fig, current_fig_zoom, current_losses


def predict_benchmark(k):
    prediction = (k - (-9.675808573724105)) / (-8.69340193226905)
    return prediction


def benchmarks_reader(n, k, p, graph_type):
    if graph_type == "Random regular":
        df = pd.read_csv("EO_benchmarks.csv")
        count_assign = dict(df['N/K'])
        res = dict((v, k) for k, v in count_assign.items())
        if str(k) not in df.columns:
            return predict_benchmark(k), None
        if str(n) not in res.keys():
            return predict_benchmark(k), None
        chosen = df[str(k)][res[str(n)]]
        if not pd.isnull(chosen):
            value, deviation = chosen.split('(')
            deviation = deviation.rstrip(')')
            deviation = "0." + "0" * (len(value) - len(deviation) - 2) + deviation  # Rough
            return -float(value), float(deviation)
        else:
            return predict_benchmark(k), None
    else:
        return None
