from torch_geometric.data import Data
import torch
import networkx as nx


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
