# %%
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

G = nx.karate_club_graph()
A = nx.to_scipy_sparse_array(G)

def random_walk(A, n_steps=1000, start_node=None, restart_node=None, t_start=None, t_end=None, restart_prob=0.0):
    """Generate a random walk sequence on a network with probabilistic restart period.

    Args:
        A: scipy sparse matrix representing adjacency matrix
        n_steps: number of steps to walk
        start_node: starting node index (if None, randomly chosen)
        restart_node: node to restart from during restart period (if None, no restart)
        t_start: start time of restart period (if None, no restart)
        t_end: end time of restart period (if None, no restart)
        restart_prob: probability of restarting during restart period (between 0 and 1)

    Returns:
        np.ndarray: Sequence of node indices visited during walk
    """
    n_nodes = A.shape[0]

    # Convert to transition probability matrix
    D = np.array(A.sum(axis=1)).flatten()
    D[D == 0] = 1  # Avoid division by zero for isolated nodes

    # Choose random starting node if not specified
    if start_node is None:
        start_node = np.random.randint(0, n_nodes)

    if restart_node is None:
        restart_node = np.random.randint(0, n_nodes)

    # Initialize walk
    walk = np.zeros(n_steps, dtype=int)
    walk[0] = start_node

    # Perform random walk
    current_node = start_node
    restarted_last_step = False  # Track if we restarted in previous step

    for i in range(1, n_steps):
        # Check if we're in restart period
        in_restart_period = (restart_node is not None and
                           t_start is not None and
                           t_end is not None and
                           t_start <= i <= t_end)

        # Determine if we should restart based on probability
        # Don't restart if we just restarted or if this is the first step
        should_restart = (in_restart_period and
                         np.random.random() < restart_prob and
                         not restarted_last_step and
                         i > 1)

        if should_restart:
            current_node = restart_node
            restarted_last_step = True
        else:
            # Get row slice for current node
            start_idx = A.indptr[current_node]
            end_idx = A.indptr[current_node + 1]

            # Get neighbors and compute probabilities
            neighbors = A.indices[start_idx:end_idx]

            # Choose next node based on transition probabilities
            current_node = np.random.choice(neighbors)
            restarted_last_step = False

        walk[i] = current_node

    return walk

# Generate random walk data
n_walkers = 1500 * 3  # Number of parallel random walks
n_steps = 50

walks = []
products = []
n_nodes = A.shape[0]
for _ in range(n_walkers):
    dt = 20
    t_start = np.random.uniform(0, n_steps - dt)
    t_end = t_start + dt
    restart_node = np.random.randint(0, n_nodes)
    seq = random_walk(A, n_steps=n_steps,
               t_start=t_start,
               t_end=t_end,
               restart_prob=0.9,
               restart_node=restart_node)
    walks.append(seq)
    products.append(restart_node)

walks = np.array(walks)

# Convert walks and products to pandas DataFrame
import pandas as pd

# Create separate columns for each timestamp
data_table = pd.DataFrame(walks, columns=[f't{i}' for i in range(walks.shape[1])])
data_table['target'] = products

train_data_size = 3000
train_data = data_table.iloc[:train_data_size]
test_data = data_table.iloc[train_data_size:]

train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)
# %%
