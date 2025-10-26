"""
Centralized PCA Algorithms with Exact Equations

This module implements centralized PCA algorithms including:
1. Standard Sanger (GHA - Generalized Hebbian Algorithm)
2. Momentum-Sanger (Heavy Ball method)

All implementations follow the exact mathematical equations specified.
"""

import numpy as np
import math
from typing import Tuple, List


class CentralizedPCA:
    """
    Centralized PCA algorithms with exact equation implementations.
    """
    
    def __init__(self, data: np.ndarray, iterations: int, K: int, ground_truth: np.ndarray = None):
        """
        Initialize centralized PCA algorithms.
        
        Args:
            data: Input data matrix (d x N)
            iterations: Number of iterations
            K: Number of principal components to estimate
            ground_truth: True eigenvectors (d x K) for error calculation
        """
        self.data = data
        self.num_itr = iterations
        self.K = K
        self.X_gt = ground_truth
        self.d, self.N = data.shape
        
        # Compute covariance matrix
        self.C = (1 / self.N) * np.dot(self.data, self.data.T)
        
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
    
    def sanger_update_step(self, W: np.ndarray, C: np.ndarray, alpha: float) -> np.ndarray:
        """
        Standard Sanger (GHA) update step with exact equation.
        
        Args:
            W: Current subspace estimate (d x K)
            C: Covariance matrix (d x d)
            alpha: Step size
            
        Returns:
            Updated subspace estimate (d x K)
        """
        # Exact Sanger equation: W_new = W + alpha * (C @ W - W @ triu(W.T @ C @ W))
        WT_C_W = W.T @ C @ W
        H = C @ W - W @ np.triu(WT_C_W)
        W_new = W + alpha * H
        return W_new
    
    def run_sanger_pca(self, alpha: float = 0.01, step_flag: int = 0) -> Tuple[np.ndarray, List[float]]:
        """
        Run Standard Sanger (GHA) algorithm with exact equations.
        
        Args:
            alpha: Step size
            step_flag: 0=constant, 1=1/t^0.2, 2=1/sqrt(t)
            
        Returns:
            Final subspace estimate and error history
        """
        print("Running Standard Sanger (GHA) PCA...")
        
        # Initialize W with random values and QR decomposition
        np.random.seed(42)
        W = np.random.rand(self.d, self.K)
        W, _ = np.linalg.qr(W)
        
        # Initialize error tracking
        errors = []
        if self.X_gt is not None:
            errors.append(self.dist_subspace(self.X_gt, W))
        
        # Main iteration loop
        for itr in range(self.num_itr):
            # Calculate step size
            if step_flag == 0:
                alpha0 = alpha  # constant step-size
            elif step_flag == 1:
                alpha0 = alpha / (itr + 1)**0.2  # decreasing step-size
            elif step_flag == 2:
                alpha0 = alpha / math.sqrt(itr + 1)
            else:
                alpha0 = alpha
            
            # Sanger update step
            W_new = self.sanger_update_step(W, self.C, alpha0)
            
            # Orthonormalization after update step
            W, _ = np.linalg.qr(W_new)
            
            # Calculate and store error
            if self.X_gt is not None:
                error = self.dist_subspace(self.X_gt, W)
                errors.append(error)
                
                if itr % 1000 == 0:
                    print(f"  Iteration {itr}: error = {error:.6f}")
        
        print(f"  Final error: {errors[-1]:.6f}")
        return W, errors
    
    def momentum_sanger_update_step(self, W: np.ndarray, V: np.ndarray, C: np.ndarray, 
                                  alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Momentum-Sanger (Heavy Ball) update step with exact equation.
        
        Args:
            W: Current subspace estimate (d x K)
            V: Current velocity (d x K)
            C: Covariance matrix (d x d)
            alpha: Step size
            beta: Momentum parameter
            
        Returns:
            Updated subspace estimate and velocity (d x K each)
        """
        # Exact momentum equation: V_new = beta * V + grad, W_new = W + alpha * V_new
        WT_C_W = W.T @ C @ W
        grad = C @ W - W @ np.triu(WT_C_W)
        V_new = beta * V + grad
        W_new = W + alpha * V_new
        return W_new, V_new
    
    def run_momentum_sanger_pca(self, alpha: float = 0.01, beta: float = 0.9, 
                               step_flag: int = 0) -> Tuple[np.ndarray, List[float]]:
        """
        Run Momentum-Sanger (Heavy Ball) algorithm with exact equations.
        
        Args:
            alpha: Step size
            beta: Momentum parameter
            step_flag: 0=constant, 1=1/t^0.2, 2=1/sqrt(t)
            
        Returns:
            Final subspace estimate and error history
        """
        print("Running Momentum-Sanger (Heavy Ball) PCA...")
        
        # Initialize W with random values and QR decomposition
        np.random.seed(42)
        W = np.random.rand(self.d, self.K)
        W, _ = np.linalg.qr(W)
        
        # Initialize velocity to zeros
        V = np.zeros_like(W)
        
        # Initialize error tracking
        errors = []
        if self.X_gt is not None:
            errors.append(self.dist_subspace(self.X_gt, W))
        
        # Main iteration loop
        for itr in range(self.num_itr):
            # Calculate step size
            if step_flag == 0:
                alpha0 = alpha  # constant step-size
            elif step_flag == 1:
                alpha0 = alpha / (itr + 1)**0.2  # decreasing step-size
            elif step_flag == 2:
                alpha0 = alpha / math.sqrt(itr + 1)
            else:
                alpha0 = alpha
            
            # Momentum-Sanger update step
            W_new, V = self.momentum_sanger_update_step(W, V, self.C, alpha0, beta)
            
            # Orthonormalization after W update
            W, _ = np.linalg.qr(W_new)
            
            # Calculate and store error
            if self.X_gt is not None:
                error = self.dist_subspace(self.X_gt, W)
                errors.append(error)
                
                if itr % 1000 == 0:
                    print(f"  Iteration {itr}: error = {error:.6f}")
        
        print(f"  Final error: {errors[-1]:.6f}")
        return W, errors
    
    def run_orthogonal_iteration(self) -> Tuple[np.ndarray, List[float]]:
        """
        Run Orthogonal Iteration (Power Method) for comparison.
        
        Returns:
            Final subspace estimate and error history
        """
        print("Running Orthogonal Iteration (Power Method)...")
        
        # Initialize W with random values and QR decomposition
        np.random.seed(42)
        W = np.random.rand(self.d, self.K)
        W, _ = np.linalg.qr(W)
        
        # Initialize error tracking
        errors = []
        if self.X_gt is not None:
            errors.append(self.dist_subspace(self.X_gt, W))
        
        # Main iteration loop
        for itr in range(self.num_itr):
            # Power method update: W = C @ W
            W = self.C @ W
            W, _ = np.linalg.qr(W)
            
            # Calculate and store error
            if self.X_gt is not None:
                error = self.dist_subspace(self.X_gt, W)
                errors.append(error)
                
                if itr % 1000 == 0:
                    print(f"  Iteration {itr}: error = {error:.6f}")
        
        print(f"  Final error: {errors[-1]:.6f}")
        return W, errors


def test_centralized_algorithms():
    """Test function to verify the centralized algorithms work correctly."""
    print("Testing Centralized PCA Algorithms...")
    
    # Generate synthetic data
    d, N, K = 20, 1000, 5
    np.random.seed(42)
    
    # Create synthetic data with known structure
    A = np.random.rand(d, d)
    U, Sigma, V = np.linalg.svd(A)
    
    # Create eigenvalues with gap
    eigvals = np.sqrt(np.linspace(1, 0.1, d))
    A_hat = U @ np.diag(eigvals) @ V.T
    
    # Generate data
    Z = np.random.randn(d, N)
    data = A_hat @ Z
    
    # Compute ground truth
    C = (1 / N) * np.dot(data, data.T)
    eigvals_gt, eigvecs_gt = np.linalg.eigh(C)
    eigvals_gt = np.flip(eigvals_gt)
    eigvecs_gt = np.fliplr(eigvecs_gt)
    X_gt = eigvecs_gt[:, :K]
    
    # Initialize algorithm
    pca = CentralizedPCA(data, iterations=1000, K=K, ground_truth=X_gt)
    
    # Test Standard Sanger
    W_sanger, errors_sanger = pca.run_sanger_pca(alpha=0.01, step_flag=0)
    
    # Test Momentum-Sanger
    W_momentum, errors_momentum = pca.run_momentum_sanger_pca(alpha=0.01, beta=0.9, step_flag=0)
    
    # Test Orthogonal Iteration
    W_oi, errors_oi = pca.run_orthogonal_iteration()
    
    # Compare results
    print("\nResults Summary:")
    print(f"Standard Sanger final error:     {errors_sanger[-1]:.6f}")
    print(f"Momentum-Sanger final error:     {errors_momentum[-1]:.6f}")
    print(f"Orthogonal Iteration final error: {errors_oi[-1]:.6f}")
    
    improvement = (errors_sanger[-1] - errors_momentum[-1]) / errors_sanger[-1] * 100
    print(f"Momentum improvement:            {improvement:.2f}%")
    
    return {
        'sanger': (W_sanger, errors_sanger),
        'momentum': (W_momentum, errors_momentum),
        'orthogonal': (W_oi, errors_oi)
    }


if __name__ == "__main__":
    test_centralized_algorithms()
