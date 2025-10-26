#!/usr/bin/env python3
"""
Main Experiment Script for Momentum-Accelerated Distributed PCA

This script runs comprehensive experiments comparing:
1. Centralized Sanger (GHA)
2. Centralized Momentum-Sanger (Heavy Ball)
3. Distributed Sanger Algorithm (DSA)
4. Momentum-Accelerated Distributed Sanger Algorithm (M-DSA)

Supports both synthetic and real datasets (MNIST, CIFAR-10).
"""

import argparse
import numpy as np
import os
import pickle
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# Import our custom modules
from centralized_pca import CentralizedPCA
from distributed_pca import DistributedPCA
from Data import Data
from GraphTopology import GraphType
import read_dataset


def parse_arguments():
    """Parse command-line arguments for the experiment."""
    parser = argparse.ArgumentParser(description='Momentum-Accelerated Distributed PCA Experiments')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='synthetic', 
                       choices=['synthetic', 'mnist', 'cifar10'],
                       help='Dataset to use for experiments')
    parser.add_argument('--K', type=int, default=5,
                       help='Number of principal components to estimate')
    parser.add_argument('--max_iters', type=int, default=1000,
                       help='Maximum number of iterations')
    parser.add_argument('--num_nodes', type=int, default=4,
                       help='Number of nodes in the distributed network')
    parser.add_argument('--connectivity', type=float, default=0.5,
                       help='Connectivity probability for Erdos-Renyi graph')
    
    # Algorithm parameters
    parser.add_argument('--alpha', type=float, default=0.01,
                       help='Step size for all algorithms')
    parser.add_argument('--alpha_dsa', type=float, default=None,
                       help='Step size specifically for DSA (overrides --alpha)')
    parser.add_argument('--alpha_mdsa', type=float, default=None,
                       help='Step size specifically for M-DSA (overrides --alpha)')
    parser.add_argument('--beta', type=float, default=0.9,
                       help='Momentum parameter for momentum-based algorithms')
    
    # Synthetic data parameters
    parser.add_argument('--eigengap', type=float, default=0.6,
                       help='Eigenvalue gap for synthetic data')
    parser.add_argument('--d', type=int, default=50,
                       help='Dimension for synthetic data')
    parser.add_argument('--N', type=int, default=2000,
                       help='Number of samples for synthetic data')
    
    # Real data parameters
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of samples for real datasets')
    
    # Output parameters
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for results (default: auto-generated)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate and save plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress information')
    
    return parser.parse_args()


def load_data(args) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load data based on the specified dataset type.
    
    Returns:
        data: Input data matrix (d x N)
        X_gt: Ground truth eigenvectors (d x K)
        data_info: Dictionary with data information
    """
    if args.verbose:
        print(f"Loading {args.dataset} dataset...")
    
    if args.dataset == 'synthetic':
        # Generate synthetic data
        data_gen = Data(args.d, args.N, args.eigengap, args.K)
        data = data_gen.generateSynthetic()
        X_gt = data_gen.computeTrueEV(data)
        
        data_info = {
            'type': 'synthetic',
            'd': args.d,
            'N': args.N,
            'eigengap': args.eigengap,
            'K': args.K
        }
        
    elif args.dataset in ['mnist', 'cifar10']:
        # Load real dataset
        data = read_dataset.read_data(args.dataset, limit=args.limit)
        
        # Load ground truth eigenvectors
        ev_path = f"Datasets/true_eigenvectors/EV_{args.dataset}.pickle"
        with open(ev_path, 'rb') as f:
            X_gt_full = pickle.load(f)
        X_gt = X_gt_full[:, :args.K]
        
        data_info = {
            'type': args.dataset,
            'd': data.shape[0],
            'N': data.shape[1],
            'K': args.K,
            'limit': args.limit
        }
        
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    if args.verbose:
        print(f"  Data shape: {data.shape}")
        print(f"  Ground truth shape: {X_gt.shape}")
        print(f"  Data info: {data_info}")
    
    return data, X_gt, data_info


def prepare_distributed_setup(data: np.ndarray, num_nodes: int, connectivity: float) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Prepare distributed setup with local covariance matrices and mixing matrix.
    
    Args:
        data: Input data matrix (d x N)
        num_nodes: Number of nodes
        connectivity: Connectivity probability for graph
        
    Returns:
        C_locals: List of local covariance matrices
        W_mixing: Mixing matrix for consensus
    """
    if num_nodes == 1:
        # Centralized case
        C = (1 / data.shape[1]) * np.dot(data, data.T)
        C_locals = [C]
        W_mixing = np.array([[1.0]])
    else:
        # Distributed case
        d, N = data.shape
        s = N // num_nodes  # samples per node
        
        C_locals = []
        for i in range(num_nodes):
            start_idx = i * s
            end_idx = (i + 1) * s
            Yi = data[:, start_idx:end_idx]
            Ci = (1 / s) * np.dot(Yi, Yi.T)
            C_locals.append(Ci)
        
        # Generate mixing matrix
        graph = GraphType('erdos-renyi', num_nodes, connectivity)
        W_mixing = graph.createGraph()
    
    return C_locals, W_mixing


def run_centralized_sanger_wrapper(data: np.ndarray, X_gt: np.ndarray, 
                                 max_iters: int, alpha: float, verbose: bool = False) -> List[float]:
    """Wrapper for centralized Sanger algorithm."""
    if verbose:
        print("  Running Centralized Sanger...")
    
    pca = CentralizedPCA(data, max_iters, X_gt.shape[1], X_gt)
    _, errors = pca.run_sanger_pca(alpha=alpha, step_flag=0)
    return errors


def run_centralized_momentum_wrapper(data: np.ndarray, X_gt: np.ndarray, 
                                   max_iters: int, alpha: float, beta: float, 
                                   verbose: bool = False) -> List[float]:
    """Wrapper for centralized momentum Sanger algorithm."""
    if verbose:
        print("  Running Centralized Momentum-Sanger...")
    
    pca = CentralizedPCA(data, max_iters, X_gt.shape[1], X_gt)
    _, errors = pca.run_momentum_sanger_pca(alpha=alpha, beta=beta, step_flag=0)
    return errors


def run_dsa_wrapper(data: np.ndarray, X_gt: np.ndarray, C_locals: List[np.ndarray], 
                   W_mixing: np.ndarray, max_iters: int, alpha: float, 
                   verbose: bool = False) -> List[float]:
    """Wrapper for Distributed Sanger Algorithm (DSA)."""
    if verbose:
        print("  Running Distributed Sanger Algorithm (DSA)...")
    
    # Initialize random starting point
    np.random.seed(42)
    X_init = np.random.rand(data.shape[0], X_gt.shape[1])
    X_init, _ = np.linalg.qr(X_init)
    
    dist_pca = DistributedPCA(data, max_iters, X_gt.shape[1], len(C_locals), 
                             X_init, X_gt)
    errors = dist_pca.DSA(W_mixing, alpha=alpha, step_flag=0)
    return errors


def run_mdsa_wrapper(data: np.ndarray, X_gt: np.ndarray, C_locals: List[np.ndarray], 
                    W_mixing: np.ndarray, max_iters: int, alpha: float, beta: float, 
                    verbose: bool = False) -> List[float]:
    """Wrapper for Momentum-Accelerated Distributed Sanger Algorithm (M-DSA)."""
    if verbose:
        print("  Running Momentum-Accelerated Distributed Sanger Algorithm (M-DSA)...")
    
    # Initialize random starting point
    np.random.seed(42)
    X_init = np.random.rand(data.shape[0], X_gt.shape[1])
    X_init, _ = np.linalg.qr(X_init)
    
    dist_pca = DistributedPCA(data, max_iters, X_gt.shape[1], len(C_locals), 
                             X_init, X_gt)
    errors = dist_pca.M_DSA(W_mixing, alpha=alpha, beta=beta, step_flag=0)
    return errors


def run_experiments(args, data: np.ndarray, X_gt: np.ndarray, 
                   C_locals: List[np.ndarray], W_mixing: np.ndarray) -> Dict[str, List[float]]:
    """
    Run all algorithms and return results.
    
    Returns:
        results: Dictionary mapping algorithm names to error lists
    """
    results = {}
    
    # Define algorithms to run
    algorithms_to_run = {
        'CentralizedSanger': {
            'func': run_centralized_sanger_wrapper,
            'params': {
                'data': data,
                'X_gt': X_gt,
                'max_iters': args.max_iters,
                'alpha': args.alpha,
                'verbose': args.verbose
            }
        },
        'CentralizedMomentum': {
            'func': run_centralized_momentum_wrapper,
            'params': {
                'data': data,
                'X_gt': X_gt,
                'max_iters': args.max_iters,
                'alpha': args.alpha,
                'beta': args.beta,
                'verbose': args.verbose
            }
        },
        'DSA': {
            'func': run_dsa_wrapper,
            'params': {
                'data': data,
                'X_gt': X_gt,
                'C_locals': C_locals,
                'W_mixing': W_mixing,
                'max_iters': args.max_iters,
                'alpha': args.alpha_dsa if args.alpha_dsa is not None else args.alpha,
                'verbose': args.verbose
            }
        },
        'M-DSA': {
            'func': run_mdsa_wrapper,
            'params': {
                'data': data,
                'X_gt': X_gt,
                'C_locals': C_locals,
                'W_mixing': W_mixing,
                'max_iters': args.max_iters,
                'alpha': args.alpha_mdsa if args.alpha_mdsa is not None else args.alpha,
                'beta': args.beta,
                'verbose': args.verbose
            }
        }
    }
    
    # Run each algorithm
    for algo_name, algo_config in algorithms_to_run.items():
        if args.verbose:
            print(f"\nRunning {algo_name}...")
        
        start_time = time.time()
        try:
            errors = algo_config['func'](**algo_config['params'])
            results[algo_name] = errors
            elapsed_time = time.time() - start_time
            
            if args.verbose:
                print(f"  Completed in {elapsed_time:.2f} seconds")
                print(f"  Final error: {errors[-1]:.6f}")
                
        except Exception as e:
            print(f"  ERROR running {algo_name}: {e}")
            results[algo_name] = []
    
    return results


def save_results(args, data_info: Dict[str, Any], C_locals: List[np.ndarray], 
                W_mixing: np.ndarray, results: Dict[str, List[float]]) -> str:
    """
    Save experiment results to file.
    
    Returns:
        output_file: Path to the saved file
    """
    # Generate output filename if not provided
    if args.output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"results/momentum_experiment_{args.dataset}_K{args.K}_n{args.num_nodes}_{timestamp}.npz"
    else:
        output_file = args.output_file
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Prepare data for saving
    save_data = {
        'args': vars(args),  # Convert args to dictionary
        'data_info': data_info,
        'W_mixing': W_mixing,
        'num_nodes': len(C_locals)
    }
    
    # Add results
    for algo_name, errors in results.items():
        save_data[f'errors_{algo_name}'] = np.array(errors)
    
    # Save using numpy
    np.savez(output_file, **save_data)
    
    if args.verbose:
        print(f"\nResults saved to: {output_file}")
    
    return output_file


def plot_results(results_file: str, save_plot: bool = True) -> str:
    """
    Generate and save plots comparing the algorithms.
    
    Args:
        results_file: Path to the saved results file
        save_plot: Whether to save the plot to file
        
    Returns:
        plot_file: Path to the saved plot file
    """
    # Load results
    data = np.load(results_file, allow_pickle=True)
    
    # Extract algorithm results
    algorithms = []
    error_lists = []
    
    for key in data.keys():
        if key.startswith('errors_'):
            algo_name = key[7:]  # Remove 'errors_' prefix
            algorithms.append(algo_name)
            error_lists.append(data[key])
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    for i, (algo, errors) in enumerate(zip(algorithms, error_lists)):
        if len(errors) > 0:
            plt.semilogy(errors, label=algo, color=colors[i % len(colors)], 
                        linestyle=linestyles[i % len(linestyles)], linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Cosine Angle Distance (log scale)')
    plt.title(f'Algorithm Comparison - {data["data_info"].item()["type"].upper()} Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    if save_plot:
        plot_file = results_file.replace('.npz', '_plot.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")
    
    plt.show()
    
    return plot_file if save_plot else None


def main():
    """Main experiment function."""
    # Parse arguments
    args = parse_arguments()
    
    if args.verbose:
        print("=" * 80)
        print("MOMENTUM-ACCELERATED DISTRIBUTED PCA EXPERIMENT")
        print("=" * 80)
        print(f"Arguments: {vars(args)}")
    
    # Load data
    data, X_gt, data_info = load_data(args)
    
    # Prepare distributed setup
    C_locals, W_mixing = prepare_distributed_setup(data, args.num_nodes, args.connectivity)
    
    if args.verbose:
        print(f"\nDistributed setup:")
        print(f"  Number of nodes: {len(C_locals)}")
        print(f"  Mixing matrix shape: {W_mixing.shape}")
        print(f"  Mixing matrix:\n{W_mixing}")
    
    # Run experiments
    results = run_experiments(args, data, X_gt, C_locals, W_mixing)
    
    # Save results
    output_file = save_results(args, data_info, C_locals, W_mixing, results)
    
    # Generate plots if requested
    if args.plot:
        plot_results(output_file, save_plot=True)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Components: {args.K}")
    print(f"Nodes: {args.num_nodes}")
    print(f"Iterations: {args.max_iters}")
    print(f"Results file: {output_file}")
    
    print(f"\nFinal Errors:")
    for algo_name, errors in results.items():
        if len(errors) > 0:
            print(f"  {algo_name}: {errors[-1]:.6f}")
        else:
            print(f"  {algo_name}: FAILED")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
