import numpy as np
import math

class AcceleratedAlgorithms:
    def __init__(self, data, iterations, K, num_nodes, initial_est, ground_truth):
        self.data = data
        self.num_itr = iterations
        self.K = K
        self.n = num_nodes
        self.X_init = initial_est
        self.X_gt = ground_truth
        
    def safe_normalize(self, M):
        norms = np.linalg.norm(M, axis=0)
        norms[norms < 1e-8] = 1e-8  # Avoid divide-by-zero and small values
        return M / norms

    
    def accelerated_DSA(self, WW, alpha=0.002, beta=0.9, step_flag=0):
        """
        Momentum-based Accelerated Distributed Sanger's Algorithm (DSA)
        """
        print("Running Accelerated DSA...")
        angle_accel_dsa = self.dist_subspace(self.X_gt, self.X_init)
        N = self.data.shape[1]
        Cy_cell = np.zeros((self.n,), dtype=object)
        s = math.floor(N / self.n)
        # Precompute covariance matrices for each node
        for i in range(self.n):
            Yi = self.data[:, i * s:(i + 1) * s]
            Cy_cell[i] = (1 / s) * np.dot(Yi, Yi.transpose())
        # Initialize weights and momentum terms
        X_dsa = np.tile(self.X_init.T, (self.n, 1))
        V = np.zeros_like(X_dsa)
        for itr in range(self.num_itr):
            if step_flag == 0:
                alpha0 = alpha
            elif step_flag == 1:
                alpha0 = alpha / (itr + 1) ** 0.2
            elif step_flag == 2:
                alpha0 = alpha / math.sqrt(itr + 1)
            grad = self.sanger_dist_update(Cy_cell, X_dsa)
            # Apply momentum update
            V = beta * V + grad
            X_dsa = np.dot(WW, X_dsa) + alpha0 * V
            
            # Apply orthonormalization to each node's estimate
            for i in range(self.n):
                X1 = X_dsa[i * self.K:(i + 1) * self.K, :]
                X2 = X1.T  # (d x K)
                X2, _ = np.linalg.qr(X2)  # Orthonormalize
                X_dsa[i * self.K:(i + 1) * self.K, :] = X2.T  # (K x d)
            
            err = 0
            for i in range(self.n):
                X1 = X_dsa[i * self.K:(i + 1) * self.K, :]
                err += self.dist_subspace(self.X_gt, X1.T)
            angle_accel_dsa = np.append(angle_accel_dsa, err / self.n)
            if itr % 1000 == 0:
                print(f"Iteration {itr}: avg error = {angle_accel_dsa[-1]:.4f}")

        return angle_accel_dsa

    def sanger_dist_update(self, Cell, X):
        """
        Distributed Sanger update with exact equations.
        X: stacked estimates from all nodes (n*K x d)
        Cell: local covariance matrices for each node
        """
        grad = np.zeros(X.shape)
        for i in range(Cell.shape[0]):
            # Extract node i's estimate (consensus result)
            X1 = X[i * self.K:(i + 1) * self.K, :]  # (K x d)
            X2 = X1.T  # (d x K)
            
            # Exact Sanger equation: grad = C @ X - X @ triu(X.T @ C @ X)
            T = np.dot(np.dot(X1, Cell[i]), X2)  # (K x K)
            T = np.triu(T)
            g = np.dot(Cell[i], X2) - np.dot(X2, T)  # (d x K)
            grad[i * self.K:(i + 1) * self.K, :] = g.T  # (K x d)
        return grad
    
    
    def dist_subspace(self, X, Y):
        X = self.safe_normalize(X)
        Y = self.safe_normalize(Y)
        M = np.matmul(X.T, Y)
        sine_angle = 1 - np.diag(M) ** 2
        dist = np.sum(sine_angle) / X.shape[1]
        return dist
