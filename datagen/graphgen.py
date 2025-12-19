import multiprocessing as mp
import networkx as nx
import os
import numpy as np

def generate_graph(i):
    nodes = 50000
    # edge, node ratio of morPOP graph = 6.54
    edges = 6.54*nodes
    density = 2*edges/(nodes*(nodes-1))
    G = nx.erdos_renyi_graph(nodes, density)

    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for j in range(len(components)-1):
            G.add_edge(np.random.choice(list(components[j])),
                       np.random.choice(list(components[j+1])))

    filename = f"graphs/graph_{i}.csv"
    nx.write_edgelist(G, filename, delimiter=",", data=False)

if __name__ == "__main__":

    output_dir = "graphs"
    os.makedirs(output_dir, exist_ok=True)

    num_processes = 36
    num_graphs = 1000

    with mp.Pool(processes=num_processes) as pool:
        pool.map(generate_graph, range(num_graphs))