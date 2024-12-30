import torch
import pickle
from click import style
from torch_geometric.data import Data
import networkx as nx
import dash_bootstrap_components as dbc
from dash import html

import datetime
import pandas as pd
import plotly.express as px
import os
import numpy as np

def make_bp_data(G, K=1):
    ei = [];
    ej = []
    l = dict();
    u = 0
    for i in G.nodes():
        for j in G.neighbors(i):
            ei.append(i)
            ej.append(j)
            l[(i, j)] = u
            u += 1

    m1 = [];
    m2 = []
    for u in range(len(ei)):
        i = ei[u]
        j = ej[u]
        for k in G.neighbors(j):
            if k != i:
                m1.append(u)
                m2.append(l[(j, k)])

    n1 = [];
    n2 = []
    for i in G.nodes():
        for j in G.neighbors(i):
            n1.append(i)
            n2.append(l[(i, j)])

    edge_index = torch.tensor([ei, ej], dtype=torch.long)
    message_index = torch.tensor([m1, m2], dtype=torch.long)
    node_agg_index = torch.tensor([n1, n2], dtype=torch.long)
    data = Data(edge_index=edge_index)
    data.message_index = message_index
    data.node_agg_index = node_agg_index
    data.clamped = torch.tensor([0] * G.number_of_nodes()).reshape((G.number_of_nodes(), 1))
    data.prior = torch.tensor([0.0] * G.number_of_nodes()).reshape((G.number_of_nodes(), 1))
    data.x = torch.randn((G.number_of_nodes(), K))
    data.num_nodes = G.number_of_nodes()
    return data

def translate_aggr(word):
    if word == "Summation":
        return "sum"
    elif word == "Multiplication":
        return "mul"
    elif word == "Average":
        return "mean"
    elif word == "Minimum":
        return "min"
    elif word == "Maximum":
        return "max"
    else:
        raise ("Unknown aggregation")

def translate_nonl(word):
    if word == "Rectified linear unit":
        return "relu"
    elif word == "Hyperbolic tangent":
        return "tanh"
    else:
        raise ("Unknown non-linearity")

def apply_split(G, split_size, split_method):
    if split_size > 1:
        if split_method == "Greedy modularity":
            splits = nx.algorithms.community.greedy_modularity_communities(G, cutoff=split_size, best_n=split_size)
            split_graph = nx.disjoint_union_all([G.subgraph(split).copy() for split in splits])
            return split_graph
        else:
            raise ("Unknown split method")
    else:
        return G

def load_graph(n, k, p, graph_type, batch_size, split_size, split_method):
    if graph_type == "Random regular":
        data = [apply_split(nx.random_regular_graph(k, n),split_size,split_method)
                for _ in range(batch_size)]
    elif graph_type == "Fast binomial":
        data = [apply_split(nx.fast_gnp_random_graph(n, p),split_size,split_method)
                for _ in range(batch_size)]
    elif graph_type == "Erdos renyi":
        data = [apply_split(nx.erdos_renyi_graph(n, p),split_size,split_method)
                for _ in range(batch_size)]
    else:
        raise ("Unknown graph type")
    data = nx.disjoint_union_all(data)
    return data

def save_model(model_outputs, model_stats):
    model, model_losses, model_info = model_outputs

    model_name = "Model_" + str(datetime.datetime.now().strftime('%d_%m-%Y_%H-%M-%S'))
    model_dir = f'saved/{model_name}'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_stats_path = os.path.join(model_dir, 'model_stats.pkl')
    model_info_path = os.path.join(model_dir, 'model_parameters.pkl')
    model_losses_path = os.path.join(model_dir, 'model_losses.pkl')
    model_path = os.path.join(model_dir, 'model.pth')

    with open(model_stats_path, 'wb') as f:
        pickle.dump(model_stats, f)

    with open(model_info_path, 'wb') as f:
        pickle.dump(model_info, f)

    with open(model_losses_path, 'wb') as f:
        pickle.dump(model_losses, f)

    torch.save(model.state_dict(), model_path)

def loss_to_plot(current_losses, epochs, benchmark):

    loss_df = pd.DataFrame({
        "Epoch": list(np.array(current_losses[0])+1),
        "Loss": current_losses[1],
        "Energy": current_losses[2],
    })

    if benchmark:
        if benchmark[1]:
            lower_value = benchmark[0] - benchmark[1] - 0.2 # Padding
        else:
            lower_value = benchmark[0] - 0.2 # Padding
    else:
        lower_value = -3 # Standard

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
        xaxis=dict(showgrid=True,gridcolor='#ececec',
                   range=[1,epochs], dtick=epochs/10, tickfont = dict(size=10)),
        yaxis=dict(showgrid=True,gridcolor='#ececec',
                   range=[lower_value,0], dtick=0.2,tickfont = dict(size=10)),
        yaxis_title=None,
        xaxis_title=dict(font=dict(size=12)),
        margin=dict(l=50, r=50, t=20, b=20),
        legend=dict(title="Metrics", orientation="h", font=dict(size=10)),
        showlegend=True,
        hovermode="x unified"
        )
    current_fig.update_traces(
        #mode="markers+lines",
        hovertemplate=None,
        line=dict(shape='spline', smoothing=1.3),
    )

    current_fig_zoom.update_layout(
        plot_bgcolor='white',
        xaxis=dict(showgrid=True,gridcolor='#ececec', tickfont = dict(size=10)),
        yaxis=dict(showgrid=True,gridcolor='#ececec', tickfont = dict(size=10)),
        yaxis_title=None,
        xaxis_title=dict(font=dict(size=12)),
        margin=dict(l=50, r=50, t=20, b=20),
        legend=dict(title="Metrics",orientation="h", font=dict(size=10)),
        showlegend=True,
        hovermode="x unified"
        )
    current_fig_zoom.update_traces(
        #mode="markers+lines",
        hovertemplate=None,
        line=dict(shape='spline', smoothing=1.3),
    )


    return current_fig, current_fig_zoom, current_losses

def general_table(inputs):
    titles = list(inputs.keys())
    model_count = len(list(inputs.values())[0])

    table_header = [
        html.Thead(html.Tr([html.Th(title) for title in titles]))
    ]

    rows = []
    for r in range(model_count):
        pieces = [list(inputs[t])[r] for t in titles]
        rows.append(html.Tr([html.Td(item) for item in pieces]))

    table_body = [html.Tbody(rows)]

    table = dbc.Table(table_header + table_body,
                      striped=False, bordered=True, hover=True, size="sm",
                      style={"font-size": "12px", "text-align": "center", "vertical-align": "middle"})

    return table

def load_parameters(file):
    file_path_params = 'saved/' + str(file) + '/model_parameters.pkl'
    with open(file_path_params, 'rb') as f:
        file_params = pickle.load(f)

    file_path_losses = 'saved/' + str(file) + '/model_losses.pkl'
    with open(file_path_losses, 'rb') as f:
        file_losses = pickle.load(f)

    return file_params, file_losses

def get_files():
    folder_path = 'saved'
    
    file_list = os.listdir(folder_path)

    files = []
    files_best_energy = []
    files_final_energy = []
    files_benchmark_energy = []
    files_training_time = []
    load_buttons = []
    for file in file_list:
        if file[0] != '.':
            files.append(file)
            stats_path = 'saved/' + str(file) + '/model_stats.pkl'
            with open(stats_path, 'rb') as f:
                model_stats = pickle.load(f)
            files_best_energy.append(model_stats["Best energy"])
            files_final_energy.append(model_stats["Current energy"])
            files_benchmark_energy.append(model_stats["Benchmark"])
            files_training_time.append(model_stats["Training time"])

            button_group = html.Div([dbc.ButtonGroup(
                [
                    dbc.Button("Load parameters", outline=True, color="primary",
                               id={"type": "Load parameters", "index": str(file)}, style={"font-size": "12px"}),
                    dbc.Button("Load results", outline=True, color="primary",
                               id={"type": "Load results", "index": str(file)}, style={"font-size": "12px"}),
                    dbc.Button("Load model", outline=False, color="primary",
                               id={"type": "Load model", "index": str(file)}, style={"font-size": "12px"})
                ], id={"type": "Button group", "index": str(file)}, size="sm"
            ), dbc.Button([html.I(className="bi bi-x-circle me-2"), "Delete"], outline=False, color="danger", size="sm",
                               id={"type": "Delete model", "index": str(file)}, style={"font-size": "12px", "margin-left":"10px"})],
            )

            load_buttons.append(button_group)

    data = {
            "Model name": files,
            "Final energy": files_final_energy,
            "Best energy": files_best_energy,
            "Benchmark energy": files_benchmark_energy,
            "Training time": files_training_time,
            "Options": load_buttons
             }

    """
    table = dbc.Table.from_dataframe(pd.DataFrame(data),
                             striped=False, bordered=True, hover=True, size="sm",
                             style={"font-size": "12px"})
    """

    table = general_table(data)

    return table

def predict_benchmark(k):
    prediction = (k - (-9.675808573724105) ) / (-8.69340193226905)
    return prediction

def benchmarks_reader(n,k,p,graph_type):
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
            deviation = "0." + "0"*(len(value)-len(deviation)-2) + deviation # Rough
            return -float(value), float(deviation)
        else:
            return predict_benchmark(k), None
    else:
        return None

def time_convert(time_diff):
    hours, remainder = divmod(time_diff, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    return formatted_time