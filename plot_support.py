import pandas as pd
import plotly.express as px
import numpy as np

def adapt_to_plot(current_losses, epochs, adaptive_parameters):

    adapt_df = pd.DataFrame({
        "Epoch": list(np.array(current_losses[0]) + 1),
        "Beta": list(adaptive_parameters[0]),
        "Damping": list(adaptive_parameters[1]),
        "Tau": list(adaptive_parameters[2])
    })

    beta_fig = px.line(adapt_df, x="Epoch", y=["Beta"],
                          color_discrete_map={"Beta": "black"})
    damping_fig = px.line(adapt_df, x="Epoch", y=["Damping"],
                          color_discrete_map={"Damping": "black"})
    tau_fig = px.line(adapt_df, x="Epoch", y=["Tau"],
                          color_discrete_map={"Tau": "black"})

    figs = [beta_fig, damping_fig, tau_fig]
    for f in figs:
        f.update_layout(
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='#ececec',
                       range=[1, epochs], dtick=epochs / 10, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor='#ececec', tickfont=dict(size=10)),
            yaxis_title=None,
            xaxis_title=None,
            margin=dict(l=50, r=50, t=20, b=20),
            legend=dict(title="Adapt", orientation="h", font=dict(size=10)),
            showlegend=False,
            hovermode="x unified"
        )
        f.update_traces(
            # mode="markers+lines",
            hovertemplate=None,
            line=dict(shape='spline', smoothing=1.3),
        )

    return beta_fig, damping_fig, tau_fig, adaptive_parameters


def loss_to_plot(current_losses, epochs, benchmark):
    loss_df = pd.DataFrame({
        "Epoch": list(np.array(current_losses[0]) + 1),
        "Loss": current_losses[1],
        "Energy": current_losses[2],
        "Test energy": current_losses[3],
    })

    if benchmark:
        if benchmark[1]:
            lower_value = benchmark[0] - benchmark[1] - 0.2  # Padding
        else:
            lower_value = benchmark[0] - 0.2  # Padding
    else:
        lower_value = -3  # Standard

    current_fig = px.line(loss_df, x="Epoch", y=["Loss", "Energy", "Test energy"],
                          color_discrete_map={"Energy": "black"})
    current_fig_zoom = px.line(loss_df, x="Epoch", y=["Loss", "Energy", "Test energy"],
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
        xaxis_title=None,
        margin=dict(l=50, r=50, t=20, b=20),
        legend=dict(title="Metrics", orientation="h", font=dict(size=10)),
        showlegend=False,
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
