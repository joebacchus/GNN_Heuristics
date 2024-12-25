from algorithms import *
from support import *
from tqdm import tqdm
import time

from support import loss_to_plot, save_model, get_files

def run_heuristica_dud(set_progress,parameters):
    set_progress([100, f" ", "warning", True, True, "Starting"])
    time.sleep(1)
    for i in tqdm(range(100)):
        time.sleep(0.1)
        progressed = int(100*i/100)
        set_progress([progressed, f"{progressed} %", "info", False, False, "Training"])
    set_progress([progressed, f"100 %", "success", False, False, "Training"])
    return None, None, None

def run_heuristica(set_progress,mod_par):

    n = int(mod_par['Node count'])
    k = int(mod_par['K-parameter'])
    p = float(mod_par['P-parameter'])
    graph_type = str(mod_par['Graph type'])

    model_type = str(mod_par['Model type'])
    # decimation = bool(mod_par['Decimation'])
    beta = float(mod_par['Beta'])
    damping = float(mod_par['Damping'])
    # anneal_boost = bool(mod_par['Anneal']) ###

    tau = float(mod_par['Tau'])
    spectral_cut_switch = bool(mod_par['Spectral initialisation'])
    learning_rate = float(mod_par['Learning rate'])
    epochs = int(mod_par['Epochs'])
    optimizer = str(mod_par['Optimizer'])
    non_linearity = str(translate_nonl(mod_par['Non linearity']))
    aggregation = str(translate_aggr(mod_par['Aggregation']))
    num_layers = int(mod_par['GNN layers'])
    hidden_size = int(mod_par['GNN size'])
    repeat_layers = int(mod_par['GNN repeats'])

    current_losses = [[0],[0],[0]]
    current_fig, current_fig_zoom = loss_to_plot(current_losses, 0, epochs)

    set_progress([100, f" ", "warning", True, True, "Starting", current_fig, current_fig_zoom])

    G = load_graph(n, k, p, graph_type)
    data = make_bp_data(G, K=1)
    if spectral_cut_switch:
        g = spectral_cut(G)
        data.x = torch.tensor(g).reshape(data.x.shape)
    data.clamped[0] = 1
    data.prior[0] = 10.0 * np.sign(data.x[0][0])

    if model_type == "Pure":
        model = GNN(hidden_size=hidden_size, num_layers=num_layers,
                    non_linearity=non_linearity, aggregation=aggregation, K=1,)
    elif model_type == "Mean-field":
        model = MFGNN(hidden_size=hidden_size, num_layers=num_layers,
                      non_linearity=non_linearity, aggregation=aggregation, K=1)
    elif model_type == "Belief propagation":
        model = BPGNN(hidden_size=hidden_size, num_layers=num_layers,
                      non_linearity=non_linearity,aggregation=aggregation, K=1)
    else:
        raise ("Unknown model")

    model.d = torch.nn.Parameter(torch.tensor(torch.math.exp(beta)))
    model.df = torch.nn.Parameter(torch.tensor(damping))

    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ("Unknown optimizer")

    results = []
    best_model = model
    best_loss = 100

    for epoch in range(epochs):

        model.train() # Informs that the model is training
        optimizer.zero_grad() # Reset gradients

        # Attached to a sequence of torch encoded functions that define loss
        loss = model.energy(data, tau=tau, num_its = repeat_layers, d=model.d, df = model.df)

        loss.backward() # Calculate gradients

        optimizer.step() # Adjust weights parameters

        # Forward pass to evaluate new values
        x = model.forward(data, num_its = repeat_layers, d=model.d, df=model.df).detach()

        data.x[:,-1] = x[:,0] # update data x

        # Sign is equivalent to argmax for 2-coloring
        g = np.sign(x.numpy().flatten()) # Projecting the results for real energy calculation

        results.append([epoch, float(loss), energy(G, g)].copy())
        current_losses = np.array(results).T
        current_fig, current_fig_zoom = loss_to_plot(current_losses, epoch, epochs)

        # detached values no longer require gradient
        if loss.detach().numpy() < best_loss:
            best_loss = loss.detach().numpy()
            best_model = model

        progressed = int(epoch/epochs*100) #######
        set_progress([progressed, f"{progressed} %", "info", False, False, "Training", current_fig, current_fig_zoom])
        #data.x = torch.randn((G.number_of_nodes(),1))

    set_progress([progressed, f"100 %", "success", False, False, "Training", current_fig, current_fig_zoom])

    output = [best_model, current_losses, mod_par]
    save_model(output)