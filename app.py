import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from commands import run_heuristica
from support import loss_to_plot, get_files
######################################################

from dash import DiskcacheManager
import diskcache

cache = diskcache.Cache("./cache")
cache.clear()
background_callback_manager = DiskcacheManager(cache)

###################################################### SIDEBAR

parameters = {"Node count": 32,
              "K-parameter": 3,
              "P-parameter": 0.1,
              "Graph type": "Random regular",

              "Model type": "Mean-field",
              "Decimation": False,
              "Beta": 2,
              "Damping": 0.9,

              "Tau": 1,
              "Spectral initialisation": True,
              "Learning rate": 0.001,
              "Epochs": 100,
              "Optimizer": "Adam",
              "Non linearity": "Rectified linear unit",
              "Aggregation": "Summation",
              "GNN layers": 8,
              "GNN size": 16,
              "GNN repeats": 16,
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
                    step=stepper
                    ),
                dbc.InputGroupText(name, id=name+" label"),
                popper(name, description)
            ], size="sm")
        ], className="numerical_input", style={"padding": "2px"})

    elif type == "select":
        reformed = [{"label": o, "value": o} for o in options[0]]
        output = html.Div(children=[
            dbc.InputGroup([
                dbc.Select(
                    options=reformed,
                    value=placeholder,
                    id=name,
                    #clearable=False,
                    ),
                dbc.InputGroupText(name, id=name+" label"),
                popper(name, description)
            ], size="sm", style={"padding": "2px"})
        ])
    elif type == "switch":
        switches.append(name)
        output = html.Div(children=[
            dbc.InputGroup([
                dbc.Button(
                    id=name,
                    n_clicks=int(placeholder),
                    style={'flex': '1', 'width': '100%'}
                ),
                dbc.InputGroupText(name, id=name+" label"),
                popper(name, description)
            ], size="sm", style={"padding": "2px"})
        ])
    return output

def run_component(name):
    output = html.Div(children=[
        dbc.InputGroup([
            dbc.Button("Start",
                id=name,
                n_clicks=0,
                style={'flex': '1', 'width': '100%'}
            )
        ], size="sm", style={"padding": "2px"})
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
    component("Spectral initialisation", "switch", parameters["Spectral initialisation"]),
    component("Learning rate", "number", parameters["Learning rate"],0,None,None),
    component("Epochs", "number", parameters["Epochs"],0,None,1),
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
init_fig, init_fig_zoom = loss_to_plot(init_losses, 0, parameters["Epochs"])
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

###################################################### INFOBAR

infobar = html.Div([
    dbc.Badge("Ready", color="info", style={"width": "100px", "margin-right": "5px", "height": "20px"}, id="Status"),
    dbc.Progress(id="Progress", label=f" ", value=0, color="primary", animated=False, striped=False,
                 style={'flex': '1', 'width': '100%',"height": "20px", "padding": "2px"})
], className="info")

###################################################### SEEBAR

seebar = html.Div([
    graphbar,
    graphbar_zoom,
    infobar,
    filebar
])

######################################################

app = Dash(external_stylesheets=[dbc.themes.FLATLY],
           background_callback_manager=background_callback_manager)
app.layout = html.Div([
    dbc.Row([dbc.Col(sidebar, width=3), dbc.Col(seebar, width=9)])
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
        return "Starting", "warning", True, "warning"
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
        Output("Loss zoom", "figure")
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