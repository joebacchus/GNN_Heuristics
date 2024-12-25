import torch
from torch_geometric.data import Data
import networkx as nx
import dash_bootstrap_components as dbc
# from dash import dash_table

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


def load_graph(n, k, p, graph_type):
    if graph_type == "Random regular":
        data = nx.random_regular_graph(k, n)
    elif graph_type == "Fast binomial":
        data = nx.fast_gnp_random_graph(n, p)
    elif graph_type == "Erdos renyi":
        data = nx.erdos_renyi_graph(n, p)
    else:
        raise ("Unknown graph type")
    return data

def save_model(model_outputs):
    model, model_losses, model_info = model_outputs

    model_name = "Model_" + str(datetime.datetime.now().strftime('%d_%m-%Y_%H-%M-%S'))
    model_dir = f'saved/{model_name}'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_info_path = os.path.join(model_dir, 'model_info.npy')
    model_losses_path = os.path.join(model_dir, 'model_losses.npy')
    model_path = os.path.join(model_dir, 'model.pth')

    with open(model_info_path, 'wb') as f:
        np.save(f, model_info)

    with open(model_losses_path, 'wb') as f:
        np.save(f, model_losses)

    torch.save(model.state_dict(), model_path)

"""
def load_model(model_name):
    model_losses = np.load(
        f"models/{model_name}/model_losses.npy", allow_pickle=True)
    model_info = np.load(
        f"models/{model_name}/model_info.npy", allow_pickle=True)
    model = th.load(
        f"models/{model_name}/model", weights_only=False)

    model_selections = {"Model": model,
                        "Model Info": model_info,
                        "Model Losses": model_losses,
                        }

    return model_selections
"""

def loss_to_plot(current_losses, epoch, epochs):

    loss_df = pd.DataFrame({
        "Epoch": current_losses[0],
        "Loss": current_losses[1],
        "Energy": current_losses[2],
    })

    current_fig = px.line(loss_df, x="Epoch", y=["Loss", "Energy"])
    current_fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(showgrid=True,gridcolor='lightgray',
                   range=[0,epochs], dtick=epochs/10),
        yaxis=dict(showgrid=True,gridcolor='lightgray',
                   range=[-2,2]),
        yaxis_title=None,
        margin=dict(l=40, r=80, t=40, b=20),
        legend=dict(title="Metrics"),
        )
    current_fig.update_traces(
        line=dict(shape='spline', smoothing=1.3),
    )

    current_fig_zoom = px.line(loss_df, x="Epoch", y=["Loss", "Energy"])
    current_fig_zoom.update_layout(
        plot_bgcolor='white',
        xaxis=dict(showgrid=True,gridcolor='lightgray'),
        yaxis=dict(showgrid=True,gridcolor='lightgray'),
        yaxis_title=None,
        margin=dict(l=40, r=80, t=40, b=20),
        legend=dict(title="Metrics")
        )
    current_fig_zoom.update_traces(
        line=dict(shape='spline', smoothing=1.3),
    )

    return current_fig, current_fig_zoom

def get_files():
    folder_path = 'saved'
    file_list = os.listdir(folder_path)
    files = [file for file in file_list if file[0] != '.']

    data = pd.DataFrame(
        {
            "File Name": files,
        }
    )
    table = dbc.Table.from_dataframe(data,
                             striped=False, bordered=True, hover=True, size="sm",
                             style={"font-size": "14px"})

    """
        table = dash_table.DataTable(
        id='Table',
        columns=[
            {'name': col, 'id': col} for col in data.columns
        ],
        data=data.to_dict('records'),
        sort_action='native',
        style_table={'height': '300px', 'overflowY': 'auto'},
    )
    """

    return table