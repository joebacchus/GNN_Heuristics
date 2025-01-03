from algorithms import *
from support import *
import time

from plot_support import loss_to_plot, benchmarks_reader
from algorithms_support import make_bp_data


def run_heuristica(set_progress, mod_par):
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
    train_parameters = float(mod_par['Train parameters'])
    spectral_cut_switch = bool(mod_par['Spectral initialisation'])
    learning_rate = float(mod_par['Learning rate'])
    split_method = str(mod_par['Split method'])
    split_size = int(mod_par['Split size'])
    batch_size = int(mod_par['Batch size'])
    epochs = int(mod_par['Epochs'])
    optimizer = str(mod_par['Optimizer'])
    non_linearity = str(translate_nonl(mod_par['Non linearity']))
    aggregation = str(translate_aggr(mod_par['Aggregation']))
    num_layers = int(mod_par['GNN layers'])
    hidden_size = int(mod_par['GNN size'])
    repeat_layers = int(mod_par['GNN repeats'])

    benchmark = benchmarks_reader(n, k, p, graph_type)
    current_losses = [[0], [0], [0]]
    current_fig, current_fig_zoom, _ = loss_to_plot(current_losses, epochs, benchmark)

    energy_stats = {"Current energy": "Unknown",
                    "Best energy": "Unknown",
                    "Benchmark": "Unknown"}

    if benchmark:
        energy_stats["Benchmark"] = float(np.round(float(benchmark[0]), 6))
    else:
        energy_stats["Benchmark"] = "Unknown"

    set_progress([100, f" ", "primary", True, True, "Starting", current_fig, current_fig_zoom,
                  energy_stats["Benchmark"], energy_stats["Current energy"], energy_stats["Best energy"],
                  time_convert(0), "Unknown"])

    G = load_graph(n, k, p, graph_type, batch_size, split_size, split_method)
    data = make_bp_data(G, K=1)
    if spectral_cut_switch:
        g = spectral_cut(G)
        data.x = torch.tensor(g).reshape(data.x.shape)
    data.clamped[0] = 1
    data.prior[0] = 10.0 * np.sign(data.x[0][0])

    if model_type == "Pure":
        model = GNN(hidden_size=hidden_size, num_layers=num_layers,
                    non_linearity=non_linearity, aggregation=aggregation, K=1, )
    elif model_type == "Mean-field":
        model = MFGNN(hidden_size=hidden_size, num_layers=num_layers,
                      non_linearity=non_linearity, aggregation=aggregation, K=1)
    elif model_type == "Belief propagation":
        model = BPGNN(hidden_size=hidden_size, num_layers=num_layers,
                      non_linearity=non_linearity, aggregation=aggregation, K=1)
    else:
        raise ("Unknown model")

    if train_parameters:
        model.d = torch.nn.Parameter(torch.tensor(torch.math.exp(beta)))
        model.df = torch.nn.Parameter(torch.tensor(damping))
        model.tau = torch.nn.Parameter(torch.tensor(tau))
        out_d = model.d;
        out_df = model.df;
        out_tau = model.tau
    else:
        out_d = torch.tensor(torch.math.exp(beta));
        out_df = torch.tensor(damping);
        out_tau = torch.tensor(tau)

    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ("Unknown optimizer")

    results = []
    best_model = model
    best_loss = 100
    best_energy = 100

    start_time = time.time()

    for epoch in range(epochs):
        estimate_start = time.time()
        current_time = time_convert(time.time() - start_time)
        model.train()  # Informs that the model is training
        optimizer.zero_grad()  # Reset gradients

        # Attached to a sequence of torch encoded functions that define loss
        loss = model.energy(data, tau=out_tau, num_its=repeat_layers, d=out_d, df=out_df)

        loss.backward()  # Calculate gradients

        optimizer.step()  # Adjust weights parameters

        # Forward pass to evaluate new values
        x = model.forward(data, num_its=repeat_layers, d=out_d, df=out_df).detach()

        data.x[:, -1] = x[:, 0]  # update data x

        # Sign is equivalent to argmax for 2-coloring
        g = np.sign(x.numpy().flatten())  # Projecting the results for real energy calculation

        iteration_energy = energy(G, g)
        results.append([epoch, float(loss), iteration_energy].copy())
        current_losses = np.array(results).T
        current_fig, current_fig_zoom, _ = loss_to_plot(current_losses, epochs, benchmark)

        energy_stats["Current energy"] = round(float(iteration_energy), 6)
        # detached values no longer require gradient
        if loss.detach().numpy() < best_loss:
            best_loss = loss.detach().numpy()
            best_model = model

        if iteration_energy < best_energy:
            energy_stats["Best energy"] = round(float(iteration_energy), 6)
            best_energy = iteration_energy

        progressed = int(epoch / epochs * 100)  #######
        estimated_time = time_convert((time.time() - start_time) + (time.time() - estimate_start) * (epochs - epoch))
        set_progress([progressed, f"{progressed} %", "primary", False, False, "Training", current_fig, current_fig_zoom,
                      energy_stats["Benchmark"], energy_stats["Current energy"], energy_stats["Best energy"],
                      current_time, estimated_time])
        # data.x = torch.randn((G.number_of_nodes(),1))

        """
        for i,param in enumerate(model.parameters()):
            print(f"Parameter {i}",param.name, param.data, param.size())
        """

    run_time = time_convert(time.time() - start_time)
    energy_stats["Training time"] = run_time

    set_progress([100, f"100 %", "success", False, False, "Training", current_fig, current_fig_zoom,
                  energy_stats["Benchmark"], energy_stats["Current energy"], energy_stats["Best energy"],
                  run_time, run_time])

    output = [best_model, current_losses, mod_par]
    save_model(output, energy_stats)
