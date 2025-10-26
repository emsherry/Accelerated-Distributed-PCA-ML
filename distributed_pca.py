"""
Distributed PCA Algorithms with Exact Equations

This module implements distributed PCA algorithms including:
1. DSA (Distributed Sanger Algorithm)
2. M-DSA (Momentum-Accelerated Distributed Sanger Algorithm)

All implementations follow the exact mathematical equations specified.
"""

import numpy as np
import math
from typing import Tuple, List, Dict


class DistributedPCA:
    """
    Distributed PCA algorithms with exact equation implementations.
    """
    
    def __init__(self, data: np.ndarray, iterations: int, K: int, num_nodes: int, 
                 initial_est: np.ndarray, ground_truth: np.ndarray):
        """
        Initialize distributed PCA algorithms.
        
        Args:
            data: Input data matrix (d x N)
            iterations: Number of iterations
            K: Number of principal components to estimate
            num_nodes: Number of nodes in the network
            initial_est: Initial subspace estimate (d x K)
            ground_truth: True eigenvectors (d x K) for error calculation
        """
        self.data = data
        self.num_itr = iterations
        self.K = K
        self.n = num_nodes
        self.X_init = initial_est
        self.X_gt = ground_truth
        self.d, self.N = data.shape
        
        # Compute local covariance matrices for each node
        self.C_locals = self._compute_local_covariances()
        
    def _compute_local_covariances(self) -> List[np.ndarray]:
        """Compute local covariance matrices for each node."""
        C_locals = []
        s = math.floor(self.N / self.n)
        
        for i in range(self.n):
            # Each node gets s samples
            start_idx = i * s
            end_idx = (i + 1) * s
            Yi = self.data[:, start_idx:end_idx]
            Ci = (1 / s) * np.dot(Yi, Yi.T)
            C_locals.append(Ci)
            
        return C_locals
    
    def _get_local_data(self, node_idx: int) -> np.ndarray:
        """Get local data for a specific node."""
        s = math.floor(self.N / self.n)
        start_idx = node_idx * s
        end_idx = (node_idx + 1) * s
        return self.data[:, start_idx:end_idx]
    
    def _compute_mini_batch_gradient(self, node_idx: int, X_hat_i_t: np.ndarray, 
                                   batch_size: int) -> np.ndarray:
        """
        Compute mini-batch gradient for node i using sampled data.
        
        Args:
            node_idx: Index of the node
            X_hat_i_t: Current consensus estimate for node i (d x K)
            batch_size: Size of mini-batch
            
        Returns:
            Mini-batch gradient (d x K)
        """
        # Get local data for this node
        X_i = self._get_local_data(node_idx)  # (d x N_i)
        N_i = X_i.shape[1]
        
        if batch_size is None or batch_size >= N_i:
            # Use full batch (precomputed covariance)
            X_hat_i_t_T_Ci_X_hat_i_t = X_hat_i_t.T @ self.C_locals[node_idx] @ X_hat_i_t
            grad_i_t = self.C_locals[node_idx] @ X_hat_i_t - X_hat_i_t @ np.triu(X_hat_i_t_T_Ci_X_hat_i_t)
        else:
            # Sample mini-batch with replacement
            batch_indices = np.random.choice(N_i, size=batch_size, replace=True)
            X_batch = X_i[:, batch_indices]  # (d x B)
            
            # Compute mini-batch covariance
            C_batch = (1.0 / batch_size) * X_batch @ X_batch.T
            
            # Compute gradient using mini-batch covariance
            X_hat_i_t_T_Cbatch_X_hat_i_t = X_hat_i_t.T @ C_batch @ X_hat_i_t
            grad_i_t = C_batch @ X_hat_i_t - X_hat_i_t @ np.triu(X_hat_i_t_T_Cbatch_X_hat_i_t)
        
        return grad_i_t
    
    def safe_normalize(self, M: np.ndarray) -> np.ndarray:
        """Safely normalize matrix columns to avoid division by zero."""
        norms = np.linalg.norm(M, axis=0)
        norms[norms < 1e-8] = 1e-8
        return M / norms
    
    def dist_subspace(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculate cosine angle distance between subspaces.
        
        Args:
            X: First subspace (d x K)
            Y: Second subspace (d x K)
            
        Returns:
            Distance metric (lower is better)
        """
        X = self.safe_normalize(X)
        Y = self.safe_normalize(Y)
        M = np.matmul(X.T, Y)
        sine_angle = 1 - np.diag(M)**2
        dist = np.sum(sine_angle) / X.shape[1]
        return dist
    
    def DSA(self, W_mixing: np.ndarray, alpha: float = 0.01, step_flag: int = 0) -> List[float]:
        """
        Distributed Sanger Algorithm (DSA) with exact equations.
        
        Args:
            W_mixing: Mixing matrix (n x n) for consensus
            alpha: Step size
            step_flag: 0=constant, 1=1/t^0.2, 2=1/sqrt(t)
            
        Returns:
            Error history
        """
        print("Running Distributed Sanger Algorithm (DSA)...")
        
        # Initialize: each node starts with the same initial estimate
        X_all_nodes = [self.X_init.copy() for _ in range(self.n)]
        
        # Initialize error tracking
        errors = []
        initial_error = np.mean([self.dist_subspace(self.X_gt, X_i) for X_i in X_all_nodes])
        errors.append(initial_error)
        
        # Main iteration loop
        for itr in range(self.num_itr):
            # Calculate step size
            if step_flag == 0:
                alpha0 = alpha
            elif step_flag == 1:
                alpha0 = alpha / (itr + 1)**0.2
            elif step_flag == 2:
                alpha0 = alpha / math.sqrt(itr + 1)
            else:
                alpha0 = alpha
            
            # Store previous iteration's estimates
            X_all_nodes_prev = [X_i.copy() for X_i in X_all_nodes]
            
            # Update each node
            for i in range(self.n):
                # --- Start DSA Update for node i ---
                
                # 1. Consensus Step (Combine)
                X_hat_i_t = np.zeros_like(X_all_nodes_prev[i])
                for j in range(self.n):
                    if W_mixing[i, j] > 0:
                        X_hat_i_t += W_mixing[i, j] * X_all_nodes_prev[j]
                
                # 2. Local Sanger Direction (using consensus estimate)
                X_hat_i_t_T_Ci_X_hat_i_t = X_hat_i_t.T @ self.C_locals[i] @ X_hat_i_t
                H_i_t = self.C_locals[i] @ X_hat_i_t - X_hat_i_t @ np.triu(X_hat_i_t_T_Ci_X_hat_i_t)
                
                # 3. Position Update (Update)
                X_i_t_plus_1 = X_hat_i_t + alpha0 * H_i_t
                
                # 4. Local Orthonormalization
                X_i_t_plus_1, _ = np.linalg.qr(X_i_t_plus_1)
                
                # Store updated estimate
                X_all_nodes[i] = X_i_t_plus_1
                # --- End DSA Update for node i ---
            
            # Calculate average error across all nodes
            avg_error = np.mean([self.dist_subspace(self.X_gt, X_i) for X_i in X_all_nodes])
            errors.append(avg_error)
            
            if itr % 1000 == 0:
                print(f"  Iteration {itr}: avg error = {avg_error:.6f}")
        
        print(f"  Final error: {errors[-1]:.6f}")
        return errors
    
    def M_DSA(self, W_mixing: np.ndarray, alpha: float = 0.01, beta: float = 0.9, 
              step_flag: int = 0, batch_size: int = None) -> List[float]:
        """
        Momentum-Accelerated Distributed Sanger Algorithm (M-DSA) with exact equations.
        
        Args:
            W_mixing: Mixing matrix (n x n) for consensus
            alpha: Step size
            beta: Momentum parameter
            step_flag: 0=constant, 1=1/t^0.2, 2=1/sqrt(t)
            batch_size: Mini-batch size for gradient computation (None for full batch)
            
        Returns:
            Error history
        """
        print("Running Momentum-Accelerated Distributed Sanger Algorithm (M-DSA)...")
        
        # Initialize: each node starts with the same initial estimate
        X_all_nodes = [self.X_init.copy() for _ in range(self.n)]
        
        # Initialize local velocities for each node
        V_all_nodes = [np.zeros_like(self.X_init) for _ in range(self.n)]
        
        # Initialize error tracking
        errors = []
        initial_error = np.mean([self.dist_subspace(self.X_gt, X_i) for X_i in X_all_nodes])
        errors.append(initial_error)
        
        # Main iteration loop
        for itr in range(self.num_itr):
            # Calculate step size
            if step_flag == 0:
                alpha0 = alpha
            elif step_flag == 1:
                alpha0 = alpha / (itr + 1)**0.2
            elif step_flag == 2:
                alpha0 = alpha / math.sqrt(itr + 1)
            else:
                alpha0 = alpha
            
            # Store previous iteration's estimates
            X_all_nodes_prev = [X_i.copy() for X_i in X_all_nodes]
            V_all_nodes_prev = [V_i.copy() for V_i in V_all_nodes]
            
            # Update each node
            for i in range(self.n):
                # --- Start M-DSA Update for node i ---
                
                # 1. Consensus Step (Combine for Position)
                X_hat_i_t = np.zeros_like(X_all_nodes_prev[i])
                for j in range(self.n):
                    if W_mixing[i, j] > 0:
                        X_hat_i_t += W_mixing[i, j] * X_all_nodes_prev[j]
                
                # 2. Local Sanger Direction (Gradient using consensus estimate)
                # Use mini-batching if specified
                grad_i_t = self._compute_mini_batch_gradient(i, X_hat_i_t, batch_size)
                
                # 3. Local Velocity Update
                V_i_t_plus_1 = beta * V_all_nodes_prev[i] + grad_i_t
                
                # 4. Local Position Update (Update using velocity)
                X_i_t_plus_1 = X_hat_i_t + alpha0 * V_i_t_plus_1
                
                # 5. Local Orthonormalization
                X_i_t_plus_1, _ = np.linalg.qr(X_i_t_plus_1)
                
                # Store updated estimates
                X_all_nodes[i] = X_i_t_plus_1
                V_all_nodes[i] = V_i_t_plus_1
                # --- End M-DSA Update for node i ---
            
            # Calculate average error across all nodes
            avg_error = np.mean([self.dist_subspace(self.X_gt, X_i) for X_i in X_all_nodes])
            errors.append(avg_error)
            
            if itr % 1000 == 0:
                print(f"  Iteration {itr}: avg error = {avg_error:.6f}")
        
        print(f"  Final error: {errors[-1]:.6f}")
        return errors
    
    def get_final_estimates(self) -> List[np.ndarray]:
        """Get the final subspace estimates from all nodes."""
        return [X_i.copy() for X_i in self.X_all_nodes] if hasattr(self, 'X_all_nodes') else None


def test_distributed_algorithms():
    """Test function to verify the distributed algorithms work correctly."""
    print("Testing Distributed PCA Algorithms...")
    
    # Generate synthetic data
    d, N, K, n = 20, 1000, 5, 4
    print(f"Data: {d}D, {N} samples, {K} components, {n} nodes")
    
    # Create synthetic data
    np.random.seed(42)
    A = np.random.rand(d, d)
    U, Sigma, V = np.linalg.svd(A)
    eigvals = np.sqrt(np.linspace(1, 0.1, d))
    A_hat = U @ np.diag(eigvals) @ V.T
    Z = np.random.randn(d, N)
    data = A_hat @ Z
    
    # Compute ground truth
    C = (1 / N) * np.dot(data, data.T)
    eigvals_gt, eigvecs_gt = np.linalg.eigh(C)
    eigvals_gt = np.flip(eigvals_gt)
    eigvecs_gt = np.fliplr(eigvecs_gt)
    X_gt = eigvecs_gt[:, :K]
    
    # Initialize random starting point
    np.random.seed(42)
    X_init = np.random.rand(d, K)
    X_init, _ = np.linalg.qr(X_init)
    
    # Create mixing matrix (Erdős-Rényi graph)
    W_mixing = np.array([
        [0.6, 0.2, 0.0, 0.2],
        [0.2, 0.6, 0.2, 0.0],
        [0.0, 0.2, 0.6, 0.2],
        [0.2, 0.0, 0.2, 0.6]
    ])
    
    print(f"Mixing matrix:\n{W_mixing}")
    print(f"Row sums: {W_mixing.sum(axis=1)}")
    
    # Initialize algorithm
    dist_pca = DistributedPCA(data, iterations=500, K=K, num_nodes=n, 
                             initial_est=X_init, ground_truth=X_gt)
    
    # Test DSA
    errors_dsa = dist_pca.DSA(W_mixing, alpha=0.01, step_flag=0)
    
    # Test M-DSA
    errors_mdsa = dist_pca.M_DSA(W_mixing, alpha=0.01, beta=0.9, step_flag=0)
    
    # Results
    print(f"\nResults Summary:")
    print(f"  DSA final error:     {errors_dsa[-1]:.6f}")
    print(f"  M-DSA final error:   {errors_mdsa[-1]:.6f}")
    
    improvement = (errors_dsa[-1] - errors_mdsa[-1]) / errors_dsa[-1] * 100
    print(f"  M-DSA improvement:   {improvement:.2f}%")
    
    return {
        'dsa': errors_dsa,
        'mdsa': errors_mdsa
    }


if __name__ == "__main__":
    test_distributed_algorithms()
