import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import argparse

import read_dataset
import read_dataset_tensor
from Algorithms import Algorithms
from Algorithms_Accelerated import AcceleratedAlgorithms
from GraphTopology import GraphType

# ==== Argument Parser for Flexibility ====
parser = argparse.ArgumentParser()
parser.add_argument("-ds", "--dataset", default="mnist", choices=["mnist", "cifar10", "tensor_mnist", "tensor_cifar10"])
parser.add_argument("-n", "--num_nodes", type=int, default=10)
parser.add_argument("-K", "--K", type=int, default=5)
parser.add_argument("-s", "--stepsize", type=float, default=0.1)
parser.add_argument("-r", "--rank", type=int, default=50, help="Rank for tensor decomposition (only for tensor datasets)")
parser.add_argument("-a", "--alpha", type=float, default=0.002, help="Step size for accelerated DSA")
parser.add_argument("-b", "--beta", type=float, default=0.9, help="Momentum for accelerated DSA")
parser.add_argument("-i", "--iterations", type=int, default=10000)
parser.add_argument("-f", "--step_flag", type=int, default=0, help="0: constant, 1: 1/t^0.2, 2: 1/sqrt(t)")
parser.add_argument("--limit", type=int, default=None, help="Max samples to load for training")

args = parser.parse_args()

# ==== Parameters ====
dataset = args.dataset
K = args.K
step_size = args.stepsize
alpha = args.alpha
beta = args.beta
iterations = args.iterations
flag = args.step_flag
num_nodes = args.num_nodes
rank = args.rank
if dataset.startswith("tensor_"):
    if dataset.endswith("mnist"):
        rank_dims = [rank, rank]
    elif dataset.endswith("cifar10"):
        rank_dims = [rank, rank, 3]
    else:
        raise ValueError("Unsupported tensor dataset.")
else:
    rank_dims = None

graph_type = "erdos-renyi"
p = 0.5

# ==== Load Dataset ====
if dataset.startswith("tensor_"):
    base = dataset.split("_", 1)[1]
    if dataset.endswith("mnist"):
        rank_dims = [rank, rank, rank]   # ✅ Fix: match tensor_mnist 3D shape
    elif dataset.endswith("cifar10"):
        rank_dims = [rank, rank, 3]      # already correct
    else:
        raise ValueError("Unsupported tensor dataset.")
    data = read_dataset_tensor.read_data_tensor(base, ranks=rank_dims)
    Cy = (1 / data.shape[1]) * np.dot(data, data.T)
    eigvals, eigvecs = np.linalg.eigh(Cy)
    X_gt = eigvecs[:, -K:]  # ← proper top-K eigenvectors
else:
    data = read_dataset.read_data(dataset, limit=args.limit)
    ev_path = f"Datasets/true_eigenvectors/EV_{dataset}.pickle"
    with open(ev_path, 'rb') as f:
        X_gt = pickle.load(f)[:, :K]



# ==== Graph Construction ====
graphW = GraphType(graph_type, num_nodes, p)
W = graphW.createGraph()
WW = np.kron(W, np.identity(K))

# ==== Initialization ====
np.random.seed(42)
X_init = np.random.rand(data.shape[0], K)
X_init, _ = np.linalg.qr(X_init)


algo_std = Algorithms(data, iterations, K, num_nodes, X_init, X_gt)
algo_accel = AcceleratedAlgorithms(data, iterations, K, num_nodes, X_init, X_gt)

# ==== Run Algorithms ====
print("Running Centralized Sanger (FAST-PCA)...")
t0 = time.time()
angle_sanger = algo_std.centralized_sanger(step_size, flag)
print("✅ Centralized Sanger done in", round(time.time() - t0, 2), "sec")

print("Running Standard DSA...")
t0 = time.time()
angle_dsa = algo_std.DSA(WW, step_size, flag)
print("✅ Standard DSA done in", round(time.time() - t0, 2), "sec")

print("Running Accelerated DSA...")
t0 = time.time()
angle_accel_dsa = algo_accel.accelerated_DSA(WW, alpha, beta, flag)
print("✅ Accelerated DSA done in", round(time.time() - t0, 2), "sec")

# ==== Plotting ====
plt.figure(figsize=(8, 6))
plt.plot(angle_sanger, label="Centralized Sanger (FAST-PCA)", linestyle='--')
plt.plot(angle_dsa, label="Standard DSA", linestyle='-.')
plt.plot(angle_accel_dsa, label="Accelerated DSA", linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Cosine Angle Difference")
plt.title(f"PCA Algorithm Comparison ({dataset.upper()})")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ==== Save Plot & Results ====
plot_path = f"results/{dataset}_core_algorithms_comparison.png"
plt.savefig(plot_path)
plt.show()
print("📈 Plot saved to:", plot_path)

results = {
    'centralized_sanger': angle_sanger,
    'standard_dsa': angle_dsa,
    'accelerated_dsa': angle_accel_dsa
}
results_path = f"results/{dataset}_core_algorithms_results.pickle"
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print("📦 Results saved to:", results_path)
