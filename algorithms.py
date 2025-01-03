from support import *

from torch_geometric.nn import GraphConv
import scipy.sparse
import numpy as np


def spectral_cut(G):
    L = nx.laplacian_matrix(G, nodelist=range(G.number_of_nodes()))
    x = np.real(scipy.sparse.linalg.eigs(L * 1.0, k=1, which='LM')[1]).flatten()
    g = np.sign(x).astype(int)
    return g  # hard (-1,1) outputs


def spectral_split(G, rank):
    L = nx.laplacian_matrix(G, nodelist=range(G.number_of_nodes()))
    x = np.real(scipy.sparse.linalg.eigs(L * 1.0, k=rank, which='LM')[1]).flatten()
    g = np.sign(x).astype(int)


def energy(G, g):
    return np.sum([g[i] * g[j] for i, j in G.edges()]) / G.number_of_nodes()


def mf(x, edges, d, df):  # Un-normalised input
    p = x.sigmoid()
    tmp = (p / d + (1 - p) * d).log() - (p * d + (1 - p) / d).log()
    lp = torch.zeros(x.shape).scatter_reduce(
        0, edges[0].unsqueeze(1).expand(-1, x.size(1)), tmp[edges[1]], "sum")
    return ((1 - df) * lp) + (df * x)


def isolated_run(iterations, x, edges, d, df, G):
    loss = []
    for _ in range(iterations):
        loss.append(energy(G, np.sign(x)))
        x = mf(x, edges, d, df)
    loss.append(energy(G, np.sign(x)))
    return np.sign(x), np.array(loss)


def ann_isolated_run(iterations, x, edges, d, df, G):
    loss = []
    for b in np.exp(np.linspace(0, d, iterations)):
        loss.append(energy(G, np.sign(x)))
        x = mf(x, edges, b, df)
    loss.append(energy(G, np.sign(x)))
    return np.sign(x), np.array(loss)


def clamp(x, cl, pr):  # freezes the cl=1 terms
    return (1 - cl) * x + cl * torch.cat([pr] * x.shape[1], dim=1)


def decimate(model, node_order, data1, verbose=True):
    data = data1.clone()
    data.x = model.forward(data, num_its, d=model.d, df=model.df)
    for i in node_order:
        data.clamped[i] = 1
        data.prior[i] = torch.sign(data.x.detach()[i]) * 10
        data.x = model.forward(data, num_its=8, d=model.d, df=model.df)
        if verbose:
            print(model.sign_energy(data, num_its=8, d=model.d, df=model.df))
    return model.energy(data, tau=1., num_its=8, d=model.d, df=model.df), data.x.detach()


class GNN(torch.nn.Module):

    def __init__(self, hidden_size, num_layers, non_linearity, aggregation, K=1):
        super().__init__()
        self.K = K  # Number of colors so K=1 is for 2-coloring
        self.input = GraphConv(K, hidden_size, aggr=aggregation)
        self.layers = torch.nn.ModuleList([])
        for l in range(num_layers):
            self.layers.append(GraphConv(hidden_size, hidden_size, aggr=aggregation))

        if non_linearity == 'relu':
            self.nonlin = torch.nn.functional.tanh
        elif non_linearity == 'tanh':
            self.nonlin = torch.nn.functional.relu
        else:
            raise ("Unknown non-linearity")

        self.agg = torch.nn.Linear(hidden_size + K, hidden_size + K)
        self.output = torch.nn.Linear(hidden_size + K, 1)
        torch.nn.init.eye_(self.agg.weight)
        torch.nn.init.zeros_(self.agg.bias)
        torch.nn.init.eye_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, data, num_its, d, df):
        x, edge_index = data.x, data.edge_index
        h = self.input(x, edge_index)
        for s in range(num_its):
            for l in range(len(self.layers)):  # Going through every layer
                h = self.nonlin(self.layers[l](h, edge_index))  # Assigning input to each with non lin after
                x, h = torch.split(self.agg(torch.cat([x, h], dim=1)),
                                   [self.K, h.shape[1]], dim=1)
        return self.output(torch.cat([x, h], dim=1))

    def sample(self, data, tau, num_its, d, df):
        x = self.forward(data, num_its=num_its, d=d, df=df)
        y = torch.stack([x, torch.zeros(x.shape)])
        return torch.nn.functional.gumbel_softmax(y, tau=tau, dim=0)[0]  # Like soft argmax/sign

    def energy(self, data, num_its, d, df, tau):
        y = 2 * self.sample(data, tau=tau, num_its=num_its, d=d, df=df) - 1
        return torch.sum(y[data.edge_index[0]] * y[data.edge_index[1]]) / (2 * (data.num_nodes))

    def set_zero(self):
        for l in range(len(self.layers)):
            torch.nn.init.zeros_(self.layers[l].lin_root.weight)
            torch.nn.init.zeros_(self.layers[l].lin_rel.weight)
            torch.nn.init.zeros_(self.layers[l].lin_rel.bias)
        torch.nn.init.eye_(self.agg.weight)
        torch.nn.init.zeros_(self.agg.bias)
        torch.nn.init.eye_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)


class MFGNN(GNN):

    def forward(self, data, num_its, d, df):
        x, edge_index = data.x, data.edge_index
        cl, pr = data.clamped, data.prior
        x = clamp(x, cl, pr)
        h = self.input(x, edge_index)
        for s in range(num_its):
            for l in range(len(self.layers)):
                x = clamp(mf(x, edge_index, d=d, df=df), cl, pr)  # separate mp line of input
                h = self.nonlin(self.layers[l](h, edge_index))  # gnn line of input
                x, h = torch.split(self.agg(torch.cat([x, h], dim=1)),  # combine lines
                                   [self.K, h.shape[1]], dim=1)
                x = clamp(x, cl, pr)
        return clamp(self.output(torch.cat([x, h], dim=1)), cl, pr)


class BPGNN(GNN):

    def forward(self, data, num_its, d, df):
        m_ind = data.message_index
        pr = data.prior[data.edge_index[1]]
        cl = data.clamped[data.edge_index[1]]
        mx = clamp(data.x[data.edge_index[1]], pr, cl)
        h = self.input(mx, m_ind)
        mx = clamp(mf(mx, m_ind, d=d, df=df), cl, pr)
        for s in range(num_its):
            for l in range(len(self.layers)):
                mx = clamp(mf(mx, m_ind, d=d, df=df), cl, pr)  # Still meanfield but on different indices
                h = self.nonlin(self.layers[l](h, m_ind))  # Different edge index input
                mx, h = torch.split(self.agg(torch.cat([mx, h], dim=1)),
                                    [self.K, h.shape[1]], dim=1)
                mx = clamp(mx, cl, pr)
        p = clamp(self.output(torch.cat([mx, h], dim=1)), cl, pr).sigmoid()
        tmp = (p / d + (1 - p) * d).log() - (p * d + (1 - p) / d).log()
        N = data.node_agg_index
        x = torch.zeros(data.num_nodes, 1).scatter_reduce(
            0, N[0].unsqueeze(1).expand(-1, 1), tmp[N[1]], "sum")
        # x = torch.zeros(data.x.shape).scatter_reduce(0, N[0].unsqueeze(1).expand(-1, data.x.shape[1]), tmp[N[1]], "sum")
        return clamp(x, data.clamped, data.prior)
