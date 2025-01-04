import torch
import pickle
import networkx as nx
import dash_bootstrap_components as dbc
from dash import html
from algorithms_support import apply_split

import datetime
import os


def time_convert(time_diff):
    hours, remainder = divmod(time_diff, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    return formatted_time


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


def load_graph(n, k, p, graph_type, batch_size, split_size, split_method):
    if graph_type == "Random regular":
        data = [apply_split(nx.random_regular_graph(k, n), split_size, split_method)
                for _ in range(batch_size)]
    elif graph_type == "Fast binomial":
        data = [apply_split(nx.fast_gnp_random_graph(n, p), split_size, split_method)
                for _ in range(batch_size)]
    elif graph_type == "Erdos renyi":
        data = [apply_split(nx.erdos_renyi_graph(n, p), split_size, split_method)
                for _ in range(batch_size)]
    else:
        raise ("Unknown graph type")
    data = nx.disjoint_union_all(data)
    return data


def load_parameters(file):
    file_path_params = 'saved/' + str(file) + '/model_parameters.pkl'
    with open(file_path_params, 'rb') as f:
        file_params = pickle.load(f)

    file_path_adapt = 'saved/' + str(file) + '/model_adapt.pkl'
    with open(file_path_adapt, 'rb') as f:
        file_adapt = pickle.load(f)

    file_path_losses = 'saved/' + str(file) + '/model_losses.pkl'
    with open(file_path_losses, 'rb') as f:
        file_losses = pickle.load(f)

    return file_params, file_losses, file_adapt


def save_model(model_outputs, model_stats):
    model, model_losses, model_adapt, model_info = model_outputs

    model_name = "Model_" + str(datetime.datetime.now().strftime('%d_%m-%Y_%H-%M-%S'))
    model_dir = f'saved/{model_name}'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_stats_path = os.path.join(model_dir, 'model_stats.pkl')
    model_info_path = os.path.join(model_dir, 'model_parameters.pkl')
    model_losses_path = os.path.join(model_dir, 'model_losses.pkl')
    model_adapt_path = os.path.join(model_dir, 'model_adapt.pkl')
    model_path = os.path.join(model_dir, 'model.pth')

    with open(model_stats_path, 'wb') as f:
        pickle.dump(model_stats, f)

    with open(model_info_path, 'wb') as f:
        pickle.dump(model_info, f)

    with open(model_losses_path, 'wb') as f:
        pickle.dump(model_losses, f)

    with open(model_adapt_path, 'wb') as f:
        pickle.dump(model_adapt, f)

    torch.save(model.state_dict(), model_path)

def load_model(model_name):
    model_dir = f'saved/{str(model_name)}/model'
    model = torch.load(model_dir, weights_only=False)
    return model

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
                          id={"type": "Delete model", "index": str(file)},
                          style={"font-size": "12px", "margin-left": "10px"})],
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
