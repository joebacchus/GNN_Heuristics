from dash import dcc, html
import dash_bootstrap_components as dbc
from support import get_files
from plot_support import loss_to_plot, benchmarks_reader


def init_globals(init_parameters):
    global global_parameters, pieces, switches, current_losses
    global_parameters = init_parameters.copy()
    pieces = []
    switches = []
    current_losses = [[0], [0], [0]]


def popper(name, description):
    return dbc.Popover(
        [dbc.PopoverHeader(name),
         dbc.PopoverBody(description)],
        target=name + " label",
        trigger="legacy",
        body=True
    )


def section(name, id, parts):
    if id:
        id_select = name + " " + id
    else:
        id_select = name
    output = html.Div(className="section", children=[
        html.H6(name),
        html.Div(parts)
    ], id=id_select)
    return output


def component(name, id, type, placeholder, *options):
    global pieces, switches
    if id:
        id_select = name + " " + id
    else:
        id_select = name
    font_size = "12px"
    line_height = "17px"
    description = "I am a description of the current chosen parameter."
    pieces.append(id_select)
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
                    id=id_select,
                    min=options[0],
                    max=options[1],
                    step=stepper,
                    style={"font-size": font_size, "line-height": line_height}
                ),
                dbc.InputGroupText(name, id=id_select + " label",
                                   style={"font-size": font_size, "line-height": line_height}),
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
                    id=id_select,
                    style={"font-size": font_size, "line-height": line_height}
                    # clearable=False,
                ),
                dbc.InputGroupText(name, id=id_select + " label",
                                   style={"font-size": font_size, "line-height": line_height}),
                popper(name, description)
            ], size="sm", style={"padding": "1px"})
        ])
    elif type == "switch":
        switches.append(id_select)
        output = html.Div(children=[
            dbc.InputGroup([
                dbc.Button(
                    id=id_select,
                    n_clicks=int(placeholder),
                    style={'flex': '1', 'width': '100%',
                           "font-size": font_size, "line-height": line_height}
                ),
                dbc.InputGroupText(name, id=id_select + " label",
                                   style={"font-size": font_size, "line-height": line_height}),
                popper(name, description)
            ], size="sm", style={"padding": "1px"})
        ])
    return output


def run_component(name, id):
    if id:
        id_select = name + " " + id
    else:
        id_select = name
    font_size = "12px"
    line_height = "17px"
    output = html.Div(children=[
        dbc.InputGroup([
            dbc.Button("Start",
                       id=id_select,
                       n_clicks=0,
                       style={'flex': '1', 'width': '100%',
                              "font-size": font_size, "line-height": line_height}
                       )
        ], size="sm", style={"padding": "1px"})
    ])
    return output


def iconify(status):
    if status == "Ready":
        return [html.I(style={"display": "none"}), status]
    elif status == "Starting":
        return [html.I(style={"display": "none"}), status]
    elif status == "Training":
        return [html.I(style={"display": "none"}), status]
    elif status == "Finished":
        return [html.I(className="bi bi-check-circle-fill me-2"), status]
    elif status == "Loaded":
        return [html.I(className="bi bi-check-circle-fill me-2"), status]
    elif status == "Stopped":
        return [html.I(className="bi bi-exclamation-triangle-fill me-2"), status]
    else:
        return [html.I(style={"display": "none"}), status]


def sections_train(alt, parameters):
    if alt == None:
        run_name = "Train model"
    elif alt == "test":
        run_name = "Test model"
    else:
        raise "Unknown alternative type"
    sections_1 = [
        component("Node count", alt, "number", parameters["Node count"], 0, None, 1),
        component("K-parameter", alt, "number", parameters["K-parameter"], 0, None, 1),
        component("P-parameter", alt, "number", parameters["P-parameter"], 0, 1, None),
        component("Graph type", alt, "select", parameters["Graph type"],
                  ["Random regular", "Fast binomial", "Erdos renyi"])
    ]

    sections_2 = [
        component("Model type", alt, "select", parameters["Model type"],
                  ["Pure", "Mean-field", "Belief propagation"]),
        component("Decimation", alt, "switch", parameters["Decimation"]),
        component("Beta", alt, "number", parameters["Beta"], None, None, None),
        component("Damping", alt, "number", parameters["Damping"], 0, 1, None)
    ]

    sections_3 = [
        component("Tau", alt, "number", parameters["Damping"], None, None, None),
        component("Train parameters", alt, "switch", parameters["Train parameters"]),
        component("Spectral initialisation", alt, "switch",
                  parameters["Spectral initialisation"]),
        component("Learning rate", alt, "number", parameters["Learning rate"], 0, None, None),
        component("Epochs", alt, "number", parameters["Epochs"], 0, None, 1),
        component("Split method", alt, "select", parameters["Split method"],
                  ["Greedy modularity"]),
        component("Split size", alt, "number", parameters["Split size"], 0, None, 1),
        component("Batch size", alt, "number", parameters["Batch size"], 0, None, 1),
        component("Optimizer", alt, "select", parameters["Optimizer"],
                  ["Adam"]),
        component("Non linearity", alt, "select", parameters["Non linearity"],
                  ["Rectified linear unit", "Hyperbolic tangent"]),
        component("Aggregation", alt, "select", parameters["Aggregation"],
                  ["Summation", "Multiplication", "Average", "Minimum", "Maximum"]),
        component("GNN layers", alt, "number", parameters["GNN layers"], 0, None, 1),
        component("GNN size", alt, "number", parameters["GNN size"], 0, None, 1),
        component("GNN repeats", alt, "number", parameters["GNN repeats"], 0, None, 1)
    ]

    section_run = [
        run_component("Run model", alt)
    ]

    sidebar = [
        section("Graph features", alt, sections_1),
        section("Model features", alt, sections_2),
        section("GNN features", alt, sections_3),
        section(run_name, alt, section_run)
    ]

    return sidebar


def load_layout(parameters, graph_parameters, losses, status_set):
    global pieces, switches, current_losses

    ###################################################### TABS

    sidebar_tabs = dbc.Tabs([
        dbc.Tab(sections_train(None, parameters), label="Train", className="tab",
                label_style={"color": "black", "font-weight": "bolder"},
                tab_style={"border-color": "#ececec"}),
        dbc.Tab(sections_train("test", parameters), label="Test", className="tab",
                label_style={"color": "black", "font-weight": "bolder"},
                tab_style={"border-color": "#ececec"}),
    ], style={"margin-bottom": "5px", "font-size": "14px"}, className="tab")

    ###################################################### GRAPHBAR

    benchmark = benchmarks_reader(graph_parameters["Node count"],
                                  graph_parameters["K-parameter"],
                                  graph_parameters["P-parameter"],
                                  graph_parameters["Graph type"])

    fig, fig_zoom, current_losses = loss_to_plot(losses, graph_parameters["Epochs"], benchmark)

    graphbar = dcc.Graph(className="graph",
                         id='Loss',
                         figure=fig,
                         config={'displayModeBar': False},
                         style={'height': '300px'}
                         )
    graphbar_zoom = dcc.Graph(className="graph",
                              id='Loss zoom',
                              figure=fig_zoom,
                              config={'displayModeBar': False},
                              style={'height': '300px'}
                              )

    ###################################################### FILEBAR

    section_load = html.Div(
        children=get_files(), id="Files",
        style={'height': '192px', 'overflowY': 'auto'}
    )

    filebar = html.Div([
        section("Saved models", None, section_load),
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
        dbc.Badge("False", id="Cancelled", style={"display": "none"}),
        dbc.Badge(status_set, color="primary",
                  style={"width": "100px", "margin-right": "5px", "height": "20px", "display": "none"},
                  id="Status"),
        dbc.Badge(iconify(status_set), color="primary",
                  style={"width": "100px", "margin-right": "5px", "height": "20px"},
                  id="Status icon"),
        dbc.Progress(id="Progress", label=f" ", value=0, color="primary", animated=False, striped=False,
                     style={'flex': '1', 'width': '100%', "height": "20px", "padding": "2px"}),
    ], className="info")

    ##################################################### NAVBAR

    navbar =  dbc.Badge("Heuristica", color="primary", className="nav") ###

    ###################################################### SEEBAR

    seebar = html.Div([
        statbar,
        graphbar,
        graphbar_zoom,
        infobar,
        filebar
    ])

    ######################################################

    out = html.Div([
        dcc.Location(id='url-refresh', refresh=True),
        #dbc.Row([navbar]),
        dbc.Row([dbc.Col(sidebar_tabs, width=2), dbc.Col(seebar, width=10)])
    ])

    return out
