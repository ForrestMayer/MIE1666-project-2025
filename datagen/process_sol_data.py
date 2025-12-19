import pandas as pd
import networkx as nx
import os
import multiprocessing as mp

def build_graph(filename):
    df = pd.read_csv(filename)
    G = nx.Graph()
    df = df.values.tolist()
    G.add_edges_from(df)
    return G

def get_node_data(i):
    cols = []
    # for each of the near-optimal solutions
    for j in range(10):
        df = pd.read_csv(f"results/graph_{i}_{j}.csv", header=None)
        cols.append(df.T.astype(int).squeeze())
    df = pd.concat(cols, axis=1)

    # calc bias
    df["mean"] = df.mean(axis=1)

    # add degree for GNN features
    G = build_graph(f"graphs/graph_{i}.csv")
    degrees = dict(G.degree)
    df["degree"] = [degrees[node] for node in sorted(G.nodes)]

    df[["mean", "degree"]].to_csv(f"node_data/graph_data_{i}.csv")

if __name__ == "__main__":
    output_dir = "node_data"
    os.makedirs(output_dir, exist_ok=True)

    num_processes = 36
    num_graphs = 1000

    with mp.Pool(processes=num_processes) as pool:
        pool.map(get_node_data, range(num_graphs))