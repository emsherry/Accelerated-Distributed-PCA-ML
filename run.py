import numpy as np
from Algorithms import Algorithms
from Algorithms_Accelerated import AcceleratedAlgorithms
from GraphTopology import GraphType
import pickle
import argparse
import read_dataset
import matplotlib.pyplot as plt
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument("-K", "--K", help="number of eigenvectors to be estimated, default number is 5", type=int, default=5)
parser.add_argument("-n", "--num_nodes", help="number of nodes in the network, default number is 10", type=int, default=10)
parser.add_argument("-s", "--stepsize", help="step size (or learning rate) for DSA and centralized GHA algorithms, default value is 0.1", type=float, default=0.1)
parser.add_argument("-ds", "--dataset", help="dataset used for the experiment, default is MNIST",
                    choices=['mnist', 'cifar10'], type=str, default="mnist")
args = parser.parse_args()

# ==== Initialization ====
iterations = 10000
K = args.K
num_nodes = args.num_nodes
step_size = args.stepsize
step_sizeg = step_size
step_sizep = 1
flag = 0
gtype = 'erdos-renyi'
p = 0.5

# ==== Generate Graph ====
graphW = GraphType(gtype, num_nodes, p)
W = graphW.createGraph()
WW = np.kron(W, np.identity(K))

# ==== Load Data ====
dataset = args.dataset
data = read_dataset.read_data(dataset)

# ==== Load Ground Truth ====
with open(f"Datasets/true_eigenvectors/EV_{dataset}.pickle", 'rb') as f:
    X1 = pickle.load(f)
X_gt = X1[:, 0:K]

# ==== Initialize ====
np.random.seed(1)
X_init = np.random.rand(data.shape[0], K)
X_init, r = np.linalg.qr(X_init)

test_accel = AcceleratedAlgorithms(data, iterations, K, num_nodes, X_init, X_gt)

test_run = Algorithms(data, iterations, K, num_nodes, initial_est=X_init, ground_truth=X_gt)

# ==== Run All Algorithms ====
print("Running centralized Sanger...")
start = time.time()
angle_sanger = test_run.centralized_sanger(step_size, flag)
print("✅ Centralized Sanger done in", round(time.time() - start, 2), "sec")

print("Running Orthogonal Iteration...")
start = time.time()
angle_oi = test_run.OI()
print("✅ OI done in", round(time.time() - start, 2), "sec")

print("Running DSA...")
start = time.time()
angle_dsa = test_run.DSA(WW, step_size, flag)
print("✅ DSA done in", round(time.time() - start, 2), "sec")

print("Running Sequential Power Method...")
start = time.time()
angle_seqdistpm = test_run.seqdistPM(W, 50)
print("✅ PM done in", round(time.time() - start, 2), "sec")

print("Running Distributed Projected GD...")
start = time.time()
angle_dpgd = test_run.distProjGD(WW, step_sizep, flag)
print("✅ DPGD done in", round(time.time() - start, 2), "sec")

print("Running Accelerated DSA...")
start = time.time()
angle_accel_dsa = test_accel.accelerated_DSA(WW, alpha=0.002, beta=0.9, step_flag=0)
print("✅ Accelerated DSA done in", round(time.time() - start, 2), "sec")


# ==== Save Results ====
results = {
    "angle_dsa": angle_dsa,
    "angle_accel_dsa": angle_accel_dsa,
    "angle_sanger": angle_sanger,
    "angle_oi": angle_oi,
    "angle_seqdistpm": angle_seqdistpm,
    "angle_dpgd": angle_dpgd,
}


filename = f'results/{dataset}_K{K}_stepsize{step_size}_flag{flag}_graph{gtype}_n{num_nodes}.pickle'
with open(filename, 'wb') as f:
    pickle.dump(results, f)
print("📦 Results saved to:", filename)

# ==== Plot for Visual Proof ====
plt.plot(angle_dsa, label="DSA")
plt.plot(angle_sanger, label="Centralized Sanger")
plt.plot(angle_oi, label="Orthogonal Iteration")
plt.plot(angle_seqdistpm, label="Sequential PM")
plt.plot(angle_dpgd, label="Distributed PGD")
plt.plot(angle_accel_dsa, label="Accelerated DSA")

plt.xlabel("Iteration")
plt.ylabel("Cosine Angle Difference")
plt.title(f"PCA Algorithm Comparison ({dataset.upper()})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'results/{dataset}_comparison_with_accel_plot.png')

plt.show()
print("📈 Plot saved and displayed.")
