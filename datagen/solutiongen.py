from gurobipy import GRB
import gurobipy as gp
import networkx as nx
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import multiprocessing as mp

def build_graph(filename):
    df = pd.read_csv(filename)
    G = nx.Graph()
    df = df.values.tolist()
    G.add_edges_from(df)
    return G


def build_model(i):

    # instance
    G = build_graph(f"graphs/graph_{i}.csv")
    budget = int(round(len(G.nodes) * 0.1))

    # params
    env = gp.Env(empty=True)
    env.start()
    m = gp.Model(env=env)
    m.setParam('OutputFlag', 0)
    m.Params.Threads = 8
    m.Params.Method = 3

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

    # pool params
    m.Params.PoolSearchMode = 2
    m.Params.PoolSolutions = 10

    m.update()
    m.optimize()

    # save solution data
    for j in range(m.SolCount):
        m.setParam(gp.GRB.Param.SolutionNumber, j)
        solution = [val.Xn for val in m.getVars()][:len(G.nodes)]
        pd.DataFrame(solution).T.to_csv(f"results/graph_{i}_{j}.csv",
                                        index=False, header=False)

if __name__ == "__main__":
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    num_processes = 9
    num_graphs = 1000

    with mp.Pool(processes=num_processes) as pool:
        pool.map(build_model, range(num_graphs))