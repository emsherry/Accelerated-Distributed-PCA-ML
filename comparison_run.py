import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from Algorithms import Algorithms
from Algorithms_Accelerated import AcceleratedAlgorithms
from GraphTopology import GraphType
import read_dataset

# ==== Parameters ====
iterations = 10000
K = 5
num_nodes = 10
dataset = 'cifar10'
step_size = 0.1
flag = 0                # 0: constant, 1: 1/t^0.2, 2: 1/sqrt(t)
graph_type = 'erdos-renyi'
connectivity_p = 0.5
alpha = 0.002           # learning rate for accelerated DSA
beta = 0.9              # momentum for accelerated DSA

# ==== Load Dataset ====
data = read_dataset.read_data(dataset)
with open(f"Datasets/true_eigenvectors/EV_{dataset}.pickle", 'rb') as f:
    X_gt = pickle.load(f)[:, :K]

# ==== Generate Graph ====
graphW = GraphType(graph_type, num_nodes, connectivity_p)
W = graphW.createGraph()
WW = np.kron(W, np.identity(K))

# ==== Initialize ====
np.random.seed(42)
X_init = np.random.rand(data.shape[0], K)
X_init, _ = np.linalg.qr(X_init)

algo_std = Algorithms(data, iterations, K, num_nodes, X_init, X_gt)
algo_accel = AcceleratedAlgorithms(data, iterations, K, num_nodes, X_init, X_gt)

# ==== Run Algorithms ====
print("Running Centralized Sanger (FAST-PCA)...")
t0 = time.time()
angle_sanger = algo_std.centralized_sanger(step_size, flag)
t1 = time.time()
print("✅ Centralized Sanger done in", round(t1 - t0, 2), "sec")

print("Running Standard DSA...")
t0 = time.time()
angle_dsa = algo_std.DSA(WW, step_size, flag)
t1 = time.time()
print("✅ Standard DSA done in", round(t1 - t0, 2), "sec")

print("Running Accelerated DSA...")
t0 = time.time()
angle_accel_dsa = algo_accel.accelerated_DSA(WW, alpha=alpha, beta=beta, step_flag=flag)
t1 = time.time()
print("✅ Accelerated DSA done in", round(t1 - t0, 2), "sec")

# ==== Plot ==== 
plt.figure(figsize=(8, 6))
plt.plot(angle_sanger, label="Centralized Sanger (FAST-PCA)", linestyle='--')
plt.plot(angle_dsa, label="Standard DSA", linestyle='-.')
plt.plot(angle_accel_dsa, label="Accelerated DSA", linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Cosine Angle Difference")
plt.title("PCA Algorithm Comparison (Core Methods)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = f"results/{dataset}_core_algorithms_comparison.png"
plt.savefig(plot_path)
plt.show()
print("📈 Plot saved to:", plot_path)

# ==== Save Metrics for Research Paper or Table ====
results_dict = {
    'centralized_sanger': angle_sanger,
    'standard_dsa': angle_dsa,
    'accelerated_dsa': angle_accel_dsa
}

results_file = f"results/{dataset}_core_algorithms_results.pickle"
with open(results_file, 'wb') as f:
    pickle.dump(results_dict, f)
print("📦 Results saved to:", results_file)
