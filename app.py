import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from commands import run_heuristica
from support import loss_to_plot, get_files, benchmarks_reader
######################################################

from dash import DiskcacheManager
import diskcache

cache = diskcache.Cache("./cache")
cache.clear()
background_callback_manager = DiskcacheManager(cache)

###################################################### SIDEBAR
font_size = "12px"

parameters = {"Node count": 128,
              "K-parameter": 3,
              "P-parameter": 0.1,
              "Graph type": "Random regular",

              "Model type": "Belief propagation",
              "Decimation": False,
              "Beta": 1,
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
              "Non linearity": "Rectified linear unit",
              "Aggregation": "Summation",
              "GNN layers": 8,
              "GNN size": 16,
              "GNN repeats": 1,
              }

def section(name, parts):
    output = html.Div(className="section", children=[
        html.H6(name),
        html.Div(parts)
    ])
    return output

pieces = []
switches = []

def popper(name, description):
    return dbc.Popover(
        [dbc.PopoverHeader(name),
         dbc.PopoverBody(description)],
        target=name + " label",
        trigger="legacy",
        body=True
    )

def component(name, type, placeholder, *options):
    description = "I am a description of the current chosen parameter."
    pieces.append(name)
    if type == "number":
        if options[2] == None:
            stepper = "any"
        else:
            stepper = options[2]
        output = html.Div(children=[
            dbc.InputGroup([
                dbc.Input(
                    type=type,
                    value=placeholder,
                    id=name,
                    min=options[0],
                    max=options[1],
                    step=stepper,
                    style={"font-size":font_size}
                    ),
                dbc.InputGroupText(name, id=name+" label", style={"font-size":font_size}),
                popper(name, description)
            ], size="sm")
        ], className="numerical_input", style={"padding": "1px"})

    elif type == "select":
        reformed = [{"label": o, "value": o} for o in options[0]]
        output = html.Div(children=[
            dbc.InputGroup([
                dbc.Select(
                    options=reformed,
                    value=placeholder,
                    id=name,
                    style={"font-size": font_size}
                    #clearable=False,
                    ),
                dbc.InputGroupText(name, id=name+" label", style={"font-size":font_size}),
                popper(name, description)
            ], size="sm", style={"padding": "1px"})
        ])
    elif type == "switch":
        switches.append(name)
        output = html.Div(children=[
            dbc.InputGroup([
                dbc.Button(
                    id=name,
                    n_clicks=int(placeholder),
                    style={'flex': '1', 'width': '100%', "font-size":font_size}
                ),
                dbc.InputGroupText(name, id=name+" label", style={"font-size":font_size}),
                popper(name, description)
            ], size="sm", style={"padding": "1px"})
        ])
    return output

def run_component(name):
    output = html.Div(children=[
        dbc.InputGroup([
            dbc.Button("Start",
                id=name,
                n_clicks=0,
                style={'flex': '1', 'width': '100%', "font-size":font_size}
            )
        ], size="sm", style={"padding": "1px"})
    ])
    return output

sections_1 = [
    component("Node count", "number", parameters["Node count"],0,None,1),
    component("K-parameter", "number", parameters["K-parameter"],0,None,1),
    component("P-parameter", "number", parameters["P-parameter"],0,1,None),
    component("Graph type", "select", parameters["Graph type"],
              ["Random regular","Fast binomial","Erdos renyi"])
]

sections_2 = [
    component("Model type", "select", parameters["Model type"],
              ["Pure", "Mean-field", "Belief propagation"]),
    component("Decimation", "switch", parameters["Decimation"]),
    component("Beta", "number", parameters["Beta"],None,None,None),
    component("Damping", "number", parameters["Damping"],0,1,None)
]

sections_3 = [
    component("Tau", "number", parameters["Damping"],None,None,None),
    component("Train parameters", "switch", parameters["Train parameters"]),
    component("Spectral initialisation", "switch", parameters["Spectral initialisation"]),
    component("Learning rate", "number", parameters["Learning rate"],0,None,None),
    component("Epochs", "number", parameters["Epochs"],0,None,1),
    component("Split method", "select", parameters["Split method"],
              ["Greedy modularity"]),
    component("Split size", "number", parameters["Split size"],0,None,1),
    component("Batch size", "number", parameters["Batch size"],0,None,1),
    component("Optimizer", "select", parameters["Optimizer"],
                  ["Adam"]),
    component("Non linearity", "select", parameters["Non linearity"],
              ["Rectified linear unit", "Hyperbolic tangent"]),
    component("Aggregation", "select", parameters["Aggregation"],
              ["Summation","Multiplication","Average","Minimum","Maximum"]),
    component("GNN layers", "number", parameters["GNN layers"],0,None,1),
    component("GNN size", "number", parameters["GNN size"],0,None,1),
    component("GNN repeats", "number", parameters["GNN repeats"],0,None,1)
]

section_run = [
    run_component("Run model")
]

sidebar = [
    section("Graph features", sections_1),
    section("Model features", sections_2),
    section("GNN features", sections_3),
    section( "Run model", section_run)
]

###################################################### GRAPHBAR
init_losses = [[0],[0],[0]]
init_benchmark = benchmarks_reader(parameters["Node count"],
                                   parameters["K-parameter"],
                                   parameters["P-parameter"],
                                   parameters["Graph type"])
init_fig, init_fig_zoom = loss_to_plot(
    init_losses, 0, parameters["Epochs"], init_benchmark)

graphbar = dcc.Graph(className="graph",
        id='Loss',
        figure=init_fig,
        config={'displayModeBar': False},
        style={'height': '300px'}
    )
graphbar_zoom = dcc.Graph(className="graph",
        id='Loss zoom',
        figure=init_fig_zoom,
        config={'displayModeBar': False},
        style={'height': '300px'}
    )

###################################################### FILEBAR

section_load = html.Div(
    children=get_files(), id="Files",
    style={'height': '192px', 'overflowY': 'auto'}
)

filebar = html.Div([
    section("Saved models", section_load),
])

###################################################### STATBAR

statbar = html.Div([
    dbc.Badge("Current energy", color="white", text_color="primary",
              style={"width": "auto", "margin-right": "5px", "height": "20px"},
              id="Current energy label"),
    dbc.Badge("Unknown", color="white", text_color="black",
              style={"width": "100px", "margin-right": "5px", "height": "20px"},
              id="Current energy", className="border me-1"),

    dbc.Badge("Best energy", color="white", text_color="primary",
                  style={"width": "auto", "margin-right": "5px", "height": "20px"},
                  id="Best energy label"),
    dbc.Badge("Unknown", color="white", text_color="black",
              style={"width": "100px", "margin-right": "5px", "height": "20px"},
              id="Best energy", className="border me-1"),

    dbc.Badge("Benchmark energy", color="white", text_color="primary",
                  style={"width": "auto", "margin-right": "5px", "height": "20px"},
                  id="Benchmark label"),
    dbc.Badge("Unknown", color="white", text_color="black",
              style={"width": "100px", "margin-right": "5px", "height": "20px"},
              id="Benchmark", className="border me-1"),

    dbc.Badge("Current time", color="white", text_color="primary",
              style={"width": "auto", "margin-right": "5px", "height": "20px"},
              id="Current time label"),
    dbc.Badge("Unknown", color="light", text_color="black",
              style={"width": "100px", "margin-right": "5px", "height": "20px"},
              id="Current time"),

    dbc.Badge("Estimated time", color="white", text_color="primary",
              style={"width": "auto", "margin-right": "5px", "height": "20px"},
              id="Estimated time label"),
    dbc.Badge("Unknown", color="light", text_color="black",
              style={"width": "100px", "margin-right": "5px", "height": "20px"},
              id="Estimated time")

])

###################################################### INFOBAR

infobar = html.Div([
    dbc.Badge("Ready", color="info", style={"width": "100px", "margin-right": "5px", "height": "20px"}, id="Status"),
    dbc.Progress(id="Progress", label=f" ", value=0, color="primary", animated=False, striped=False,
                 style={'flex': '1', 'width': '100%',"height": "20px", "padding": "2px"}),
], className="info")

###################################################### SEEBAR

seebar = html.Div([
    statbar,
    graphbar,
    graphbar_zoom,
    infobar,
    filebar
])

######################################################

app = Dash(external_stylesheets=[dbc.themes.FLATLY],
           background_callback_manager=background_callback_manager)
app.layout = html.Div([
    dbc.Row([dbc.Col(sidebar, width=2), dbc.Col(seebar, width=10)])
])

for s in switches:
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

for p in pieces:
    if p in switches:
        detector = "children"
    else:
        detector = "value"
    @callback(
        Input(p, detector),
        Input(p, "id"),
        Input("Status", "children")
    )
    def update_parameters(detector, id, status):
        if detector == "Off":
            detector = False
        elif detector == "On":
            detector = True
        parameters[id] = detector

@callback(
    Output("Run model", "children"),
    Output("Run model", "color"),
    Output("Run model", "disabled"),
    Output("Status", "color"),
    Input("Status", "children")
)
def status_change(status):
    if status == "Ready":
        return "Start", "primary", False, "info"
    elif status == "Starting":
        return "Starting", "info", True, "info"
    elif status == "Training":
        return "Stop", "danger", False, "info"
    elif status == "Stopping":
        return "Stopping", "warning", True, "warning"
    elif status == "Finished":
        return "Start", "primary", False, "success"
    elif status == "Stopped":
        return "Start", "primary", False, "danger"
    else:
        return "Unknown", "danger", True, "danger"

@callback(
    Output("Status", "children", allow_duplicate=True),
    Output("Run model", "n_clicks", allow_duplicate=True),
    Output("Files", "children"),
    State("Status", "children"),
    Input("Run model", "n_clicks"),
    interval= 50,
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
    #cancel=[Input("Run model", "n_clicks")]
)
def run_process(set_progress, status_value, run_clicks):
    if run_clicks >= 2:
        return "Stopped", 0, get_files()
    if run_clicks == 1 and (status_value == "Ready" or status_value == "Stopped" or status_value == "Finished"):
        cache.clear()
        run_heuristica(set_progress, parameters)
        return "Finished", 0, get_files()

if __name__ == '__main__':
    app.run(debug=True, dev_tools_ui=False)

"""
 dbc.Popover(
                    [dbc.PopoverHeader(name),
                    dbc.PopoverBody("The number of epochs determines how many steps to iterate over.")],
                    target=name+" label",
                    trigger="click",
                    body=True
                )
"""