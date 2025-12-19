from torch_geometric.data import Data
import torch

# import SGS
from training.gnn_models.Sage.mip_bipartite_simple_class import SimpleNet

from gurobipy import GRB
import gurobipy as gp
import networkx as nx
import pandas as pd
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os

def get_graph_data(i):
    df = pd.read_csv(f"datagen/node_data/graph_data_{i}.csv", index_col=0)

    num_nodes_var = len(df)
    num_nodes_con = 1

    feat_var = [[row["degree"], 1] for _, row in df.iterrows()]

    budget = int(round(len(df) * 0.1))
    feat_con = [[budget, len(df)]]
    feat_rhs = [[budget]]

    y_real = [row["mean"] for _, row in df.iterrows()]
    y = [0 if val <= 0 else 1 for val in y_real]

    edge_list_var = [[j, 0] for j in range(num_nodes_var)]
    edge_list_con = [[0, j] for j in range(num_nodes_var)]
    edge_features_var = [[1] for _ in range(num_nodes_var)]
    edge_features_con = [[1] for _ in range(num_nodes_var)]

    data = Data(
        var_node_features=torch.tensor(feat_var, dtype=torch.float),
        con_node_features=torch.tensor(feat_con, dtype=torch.float),
        rhs=torch.tensor(feat_rhs, dtype=torch.float),
        edge_index_var=torch.tensor(edge_list_var).t().contiguous(),
        edge_index_con=torch.tensor(edge_list_con).t().contiguous(),
        edge_features_var=torch.tensor(edge_features_var, dtype=torch.float),
        edge_features_con=torch.tensor(edge_features_con, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.long),
        y_real=torch.tensor(y_real, dtype=torch.float),
        num_nodes_var=num_nodes_var,
        num_nodes_con=num_nodes_con,
        index=torch.tensor([0], dtype=torch.long)
    )
    return data

def get_prediction(i):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleNet(64, aggr="mean", num_layers=4).to(device)

    # Note: change model name if needed
    model.load_state_dict(torch.load("training/training_results/SGS_0.0_0_0", map_location=device))

    data = get_graph_data(i)
    model.eval()

    data = data.to(device)
    out = model(data).exp()[:, 1].cpu().detach().numpy()

    return out


def build_graph(filename):
    df = pd.read_csv(filename)
    G = nx.Graph()
    df = df.values.tolist()
    G.add_edges_from(df)
    return G


def get_warmstart(i, m, G):
    # get prediction
    prediction_list = get_prediction(i)

    # build predicted solution
    predicted_solution = []
    for j in prediction_list:
        if j >= 0.9:
            predicted_solution.append(1)
        else:
            predicted_solution.append(0)

    # repair predicted solution to be feasible
    budget = int(round(len(G.nodes) * 0.1))
    degrees = dict(G.degree)
    while sum(predicted_solution) < budget:
        predicted_solution[max([j for j, k in enumerate(predicted_solution) if k == 0], key=lambda l: degrees[l])] = 1

    # fix vars
    for k in range(len(predicted_solution)):
        m.addConstr(m.getVars()[:len(G.nodes)][k] == predicted_solution[k])

    # solve lp relaxation
    m.update()
    m.optimize()

    # get primal and dual vectors
    predicted_primal = []
    for v in m.getVars():
        predicted_primal.append(v.X)

    predicted_dual = []
    for c in m.getConstrs():
        predicted_dual.append(c.Pi)

    return predicted_primal, predicted_dual, m.ObjVal


def build_subgraph_model(G, budget):
    # params
    env = gp.Env(empty=True)
    env.start()
    m = gp.Model(env=env)
    m.setParam('OutputFlag', 0)
    m.Params.Threads = 8
    m.Params.Method = 3
    m.Params.ConcurrentMethod = 3

    # BVC-AND

    # vars
    node_vars = m.addVars(sorted(G.nodes), vtype=GRB.BINARY)
    neighbor_vars = m.addVars(G.edges, vtype=GRB.CONTINUOUS)

    # constrs
    m.addConstr(gp.quicksum(node_vars) == budget)
    m.addConstrs((node_vars[u] + node_vars[v] - 1 <= neighbor_vars[u, v] for u, v in G.edges))

    # objective
    degrees = dict(G.degree)
    obj = gp.quicksum(node_vars[v] * degrees[v] for v in G.nodes) - gp.quicksum(neighbor_vars)
    m.setObjective(obj, GRB.MAXIMIZE)
    m.update()

    return m


def get_integer_sol(G, nodes):
    num_edges = len(G.edges)

    for v in nodes:
        G.remove_node(v)

    return num_edges - len(G.edges)


def build_model(i, prediction=None, ER=None):
    # init
    G = build_graph(f"datagen/graphs/graph_{i}.csv")
    budget = int(round(len(G.nodes) * 0.1))

    # params
    env = gp.Env(empty=True)
    env.start()
    m = gp.Model(env=env)
    m.setParam('OutputFlag', 0)
    m.Params.Threads = 8
    m.Params.Method = 3
    m.Params.ConcurrentMethod = 3

    # BVC-AND

    # vars
    node_vars = m.addVars(sorted(G.nodes), vtype=GRB.BINARY)
    neighbor_vars = m.addVars(G.edges, vtype=GRB.CONTINUOUS)

    # constrs
    m.addConstr(gp.quicksum(node_vars) == budget)
    m.addConstrs(node_vars[u] + node_vars[v] - 1 <= neighbor_vars[u, v] for u, v in G.edges)

    # obj
    degrees = dict(G.degree)
    m.setObjective(gp.quicksum(node_vars[v] * degrees[v] for v in G.nodes) - gp.quicksum(neighbor_vars), GRB.MAXIMIZE)

    # relax
    m.update()
    m_relaxed = m.relax()
    m_relaxed.update()

    if prediction:
        # get warmstart
        predicted_primal, predicted_dual, predicted_objective = get_warmstart(i, m_relaxed.copy(), G)

        # set warmstart vectors
        vars = m_relaxed.getVars()
        for v in range(len(vars)):
            vars[v].PStart = predicted_primal[v]

        constrs = m_relaxed.getConstrs()
        for c in range(len(constrs)):
            constrs[c].DStart = predicted_dual[c]

        m_relaxed.Params.LPWarmStart = 2

        # solve warmstart relaxation
        m_relaxed.setParam("LogFile", f"results/solve_time_graph_{i}_pred.log")
        m_relaxed.update()
        m_relaxed.optimize()

        if ER:
            # check fractionality at root
            all_relaxed_vars_values = [v.X for v in m_relaxed.getVars()[:len(G.nodes)]]
            fractional_nodes_values = [v.X for v in m_relaxed.getVars()[:len(G.nodes)] if v.X != 1 and v.X != 0]
            fractional_nodes_ids = [v.index for v in m_relaxed.getVars()[:len(G.nodes)] if v.X != 1 and v.X != 0]

            # if root is not integral
            if len(fractional_nodes_values) > 0:

                # current variables at 1
                integer_sol = set([v.index for v in m_relaxed.getVars()[:len(G.nodes)] if v.X == 1])

                # extract sub-graph of fraction solution and neighbors
                sub_graph_nodes = set(fractional_nodes_ids)
                for v in fractional_nodes_ids:
                    sub_graph_nodes.update(G.neighbors(v))

                sub_graph = G.subgraph(sub_graph_nodes).copy()

                # get integer variables that will be fixed
                ones = []
                zeros = []
                for v in sub_graph.nodes:
                    if all_relaxed_vars_values[v] == 1:
                        ones.append(v)
                    elif all_relaxed_vars_values[v] == 0:
                        zeros.append(v)

                # index mapping
                mapping = {old: new for new, old in enumerate(sub_graph.nodes())}
                sub_graph = nx.relabel_nodes(sub_graph, mapping)

                ones = [mapping[v] for v in ones]
                zeros = [mapping[v] for v in zeros]

                # build sub MIP
                m = build_subgraph_model(sub_graph, sum(fractional_nodes_values) + len(ones))

                # fix vars
                vars = m.getVars()
                for v in ones:
                    m.addConstr(vars[v] == 1)
                for v in zeros:
                    m.addConstr(vars[v] == 0)
                m.update()

                # exactly solve sub MIP
                m.update()
                m.optimize()

                # get integer solution
                inv = {new: old for old, new in mapping.items()}
                integer_subsol = set([inv[v.index] for v in m.getVars()[:len(sub_graph.nodes)] if v.X == 1])
                integer_sol.update(integer_subsol)

                opt_val = get_integer_sol(G.copy(), integer_sol)

                with open(f"results/objective_value_ER_{i}.log", "w") as f:
                    f.write(f"Best objective {opt_val}")

    else:
        # default solve
        m_relaxed.setParam("LogFile", f"results/solve_time_graph_{i}.log")
        m_relaxed.update()
        m_relaxed.optimize()


if __name__ == "__main__":
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    num_processes = 8
    num_graphs = 200
    with mp.Pool(processes=num_processes) as pool:
        pool.map(build_model, range(num_graphs))
        pool.map(partial(build_model, prediction=True, ER=True), range(num_graphs))
