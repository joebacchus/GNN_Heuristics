import os

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

from dash import Dash, Input, Output, State, callback, ctx, ALL, no_update
import dash_bootstrap_components as dbc
from commands import run_heuristica
from support import get_files, load_parameters
from components import load_layout, iconify, init_globals
import components

from dash import DiskcacheManager
import diskcache

cache = diskcache.Cache("./cache")
cache.clear()
background_callback_manager = DiskcacheManager(cache)

init_parameters = {"Node count": 512,
                   "K-parameter": 3,
                   "P-parameter": 0.1,
                   "Graph type": "Random regular",

                   "Model type": "Belief propagation",
                   "Decimation": False,
                   "Beta": 2,
                   "Damping": 0.9,

                   "Tau": 1,
                   "Train parameters": True,
                   "Spectral initialisation": True,
                   "Learning rate": 0.001,
                   "Split method": "Greedy modularity",
                   "Split size": 1,
                   "Batch size": 100,
                   "Epochs": 300,
                   "Optimizer": "Adam",
                   "Non linearity": "Hyperbolic tangent",
                   "Aggregation": "Summation",
                   "GNN layers": 8,
                   "GNN size": 16,
                   "GNN repeats": 1,
                   }

init_globals(init_parameters)

app = Dash(external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
           background_callback_manager=background_callback_manager)

app.layout = load_layout(init_parameters, init_parameters, [[0], [0], [0]], "Ready")

for s in components.switches:
    @callback(
        Output(s, "children"),
        Output(s, "color"),
        Input(s, "n_clicks")
    )
    def toggle_switch(n_clicks):
        if n_clicks % 2 == 0:
            return "Off", "danger"
        else:
            return "On", "success"

for p in components.pieces:
    if p in components.switches:
        detector = "children"
    else:
        detector = "value"


    @callback(
        Input(p, detector),
        Input(p, "id"),
        Input("Status", "children"),
    )
    def update_parameters(detector, id, status):
        if detector == "Off":
            detector = False
        elif detector == "On":
            detector = True
        components.global_parameters[id] = detector


@callback(
    Output('url-refresh', 'href'),
    # Output("Files", "children"),
    State("Cancelled", "children"),
    Input({'type': 'Load parameters', 'index': ALL}, 'n_clicks'),
    Input({'type': 'Load results', 'index': ALL}, 'n_clicks'),
    Input({'type': 'Load model', 'index': ALL}, 'n_clicks'),
    Input({'type': 'Delete model', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def table_options(cancelled_status, param_clicks, result_clicks, model_clicks, delete_clicks):
    trigger = ctx.triggered_id

    if trigger and cancelled_status == "False":
        file_id = trigger['index']

        if trigger['type'] == 'Load parameters':
            components.global_parameters, _ = load_parameters(file_id)
            app.layout = load_layout(components.global_parameters.copy(), components.global_parameters.copy(),
                                     [[0], [0], [0]], "Loaded")
            return "/"

        elif trigger['type'] == 'Load results':
            graph_parameters, file_losses = load_parameters(file_id)
            app.layout = load_layout(components.global_parameters.copy(), graph_parameters.copy(), file_losses,
                                     "Loaded")
            return "/"

        elif trigger['type'] == 'Delete model':
            delete_path = "saved/" + file_id
            for filename in os.listdir(delete_path):
                file_path = delete_path + "/" + filename
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(delete_path)
            app.layout = load_layout(components.global_parameters.copy(), components.global_parameters.copy(),
                                     components.current_losses, "Ready")
            return "/"

    return no_update


@callback(
    Output("Run model", "children"),
    Output("Run model", "color"),
    Output("Run model", "disabled"),
    Output("Status icon", "color"),
    Output("Status icon", "children"),
    Input("Status", "children")
)
def status_change(status):
    if status == "Ready":
        return "Start", "primary", False, "primary", iconify(status)
    elif status == "Starting":
        return "Starting", "primary", True, "primary", iconify(status)
    elif status == "Training":
        return "Stop", "danger", False, "primary", iconify(status)
    elif status == "Finished":
        return "Start", "primary", False, "success", iconify(status)
    elif status == "Loaded":
        return "Start", "primary", False, "success", iconify(status)
    elif status == "Stopped":
        return "Start", "primary", False, "danger", iconify(status)
    else:
        return "Unknown", "danger", True, "danger", iconify(status)


@callback(
    Output("Status", "children", allow_duplicate=True),
    Output("Run model", "n_clicks", allow_duplicate=True),
    Output("Files", "children"),
    Output("Cancelled", "children"),
    State("Status", "children"),
    Input("Run model", "n_clicks"),
    interval=50,
    background=True,
    progress=[
        Output("Progress", "value"),
        Output("Progress", "label"),
        Output("Progress", "color"),
        Output("Progress", "striped"),
        Output("Progress", "animated"),
        Output("Status", "children"),
        Output("Loss", "figure"),
        Output("Loss zoom", "figure"),

        Output("Benchmark", "children"),
        Output("Current energy", "children"),
        Output("Best energy", "children"),

        Output("Current time", "children"),
        Output("Estimated time", "children")
    ],
    prevent_initial_call=True,
    cancel=[Input("Cancelled", "children")],
)
def run_process(set_progress, status_value, run_clicks):
    if run_clicks >= 2:
        return "Stopped", 0, get_files(), "True"
    if run_clicks == 1 and (status_value == "Ready" or status_value == "Stopped" or
                            status_value == "Finished" or status_value == "Loaded"):
        cache.clear()
        run_heuristica(set_progress, components.global_parameters)
        return "Finished", 0, get_files(), "True"


@callback(
    Output("Cancelled", "children", allow_duplicate=True),
    Input("Cancelled", "children"),
    prevent_initial_call=True
)
def toggle_back(cancelled_status):
    if cancelled_status == "True":
        return "False"
    else:
        return no_update


if __name__ == '__main__':
    app.run(debug=True, dev_tools_ui=True)
