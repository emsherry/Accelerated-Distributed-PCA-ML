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
    
    # Hyperparameter sweep parameters
    parser.add_argument('--alpha_list', type=float, nargs='+', default=None,
                       help='List of alpha values for hyperparameter sweep')
    parser.add_argument('--beta_list', type=float, nargs='+', default=None,
                       help='List of beta values for hyperparameter sweep')
    parser.add_argument('--target_dataset', type=str, default=None,
                       choices=['synthetic', 'mnist', 'cifar10'],
                       help='Target dataset for hyperparameter sweep')
    parser.add_argument('--target_K', type=int, default=None,
                       help='Target K for hyperparameter sweep')
    parser.add_argument('--target_nodes', type=int, default=None,
                       help='Target number of nodes for hyperparameter sweep')
    parser.add_argument('--sweep_mode', action='store_true',
                       help='Enable hyperparameter sweep mode')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Mini-batch size for M-DSA gradient computation (None for full batch)')
    
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
                    verbose: bool = False, batch_size: int = None) -> List[float]:
    """Wrapper for Momentum-Accelerated Distributed Sanger Algorithm (M-DSA)."""
    if verbose:
        print("  Running Momentum-Accelerated Distributed Sanger Algorithm (M-DSA)...")
    
    # Initialize random starting point
    np.random.seed(42)
    X_init = np.random.rand(data.shape[0], X_gt.shape[1])
    X_init, _ = np.linalg.qr(X_init)
    
    dist_pca = DistributedPCA(data, max_iters, X_gt.shape[1], len(C_locals), 
                             X_init, X_gt)
    errors = dist_pca.M_DSA(W_mixing, alpha=alpha, beta=beta, step_flag=0, batch_size=batch_size)
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
                'verbose': args.verbose,
                'batch_size': args.batch_size
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


def run_hyperparameter_sweep(args):
    """Run hyperparameter sweep for M-DSA on target configuration."""
    print("=" * 80)
    print("HYPERPARAMETER SWEEP MODE")
    print("=" * 80)
    
    # Set default values for sweep if not provided
    if args.alpha_list is None:
        args.alpha_list = [0.005, 0.01, 0.02, 0.05]
    if args.beta_list is None:
        args.beta_list = [0.8, 0.9, 0.95, 0.99]
    if args.target_dataset is None:
        args.target_dataset = 'mnist'
    if args.target_K is None:
        args.target_K = 5
    if args.target_nodes is None:
        args.target_nodes = 4
    
    print(f"Target configuration: {args.target_dataset}, K={args.target_K}, nodes={args.target_nodes}")
    print(f"Alpha values: {args.alpha_list}")
    print(f"Beta values: {args.beta_list}")
    
    # Load data for target configuration
    if args.target_dataset == 'synthetic':
        data_gen = Data(args.d, args.N, args.eigengap, args.target_K)
        data = data_gen.generateSynthetic()
        X_gt = data_gen.computeTrueEV(data)
        data_info = {
            'type': 'synthetic',
            'd': args.d,
            'N': args.N,
            'eigengap': args.eigengap,
            'K': args.target_K
        }
    elif args.target_dataset in ['mnist', 'cifar10']:
        data = read_dataset.read_data(args.target_dataset, limit=args.limit)
        ev_path = f"Datasets/true_eigenvectors/EV_{args.target_dataset}.pickle"
        with open(ev_path, 'rb') as f:
            X_gt_full = pickle.load(f)
        X_gt = X_gt_full[:, :args.target_K]
        data_info = {
            'type': args.target_dataset,
            'd': data.shape[0],
            'N': data.shape[1],
            'K': args.target_K,
            'limit': args.limit
        }
    
    # Prepare distributed setup
    C_locals, W_mixing = prepare_distributed_setup(data, args.target_nodes, args.connectivity)
    
    # Store all sweep results
    sweep_results = {
        'target_config': {
            'dataset': args.target_dataset,
            'K': args.target_K,
            'num_nodes': args.target_nodes,
            'data_info': data_info
        },
        'hyperparameters': {
            'alpha_list': args.alpha_list,
            'beta_list': args.beta_list
        },
        'results': {}
    }
    
    # Run baseline DSA with a few alpha values for comparison
    print(f"\nRunning baseline DSA experiments...")
    baseline_alphas = [0.005, 0.01, 0.02]
    for alpha in baseline_alphas:
        print(f"  DSA with alpha={alpha}")
        errors = run_dsa_wrapper(data, X_gt, C_locals, W_mixing, 
                               args.max_iters, alpha, verbose=False)
        sweep_results['results'][f'DSA_alpha{alpha}'] = {
            'algorithm': 'DSA',
            'alpha': alpha,
            'beta': None,
            'errors': errors
        }
    
    # Run M-DSA sweep
    print(f"\nRunning M-DSA hyperparameter sweep...")
    total_combinations = len(args.alpha_list) * len(args.beta_list)
    current_combination = 0
    
    for alpha in args.alpha_list:
        for beta in args.beta_list:
            current_combination += 1
            print(f"  M-DSA {current_combination}/{total_combinations}: alpha={alpha}, beta={beta}")
            
            errors = run_mdsa_wrapper(data, X_gt, C_locals, W_mixing,
                                    args.max_iters, alpha, beta, verbose=False, batch_size=args.batch_size)
            
            sweep_results['results'][f'M-DSA_alpha{alpha}_beta{beta}'] = {
                'algorithm': 'M-DSA',
                'alpha': alpha,
                'beta': beta,
                'errors': errors
            }
    
    # Save sweep results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sweep_filename = f"results/hyperparameter_sweep_{args.target_dataset}_K{args.target_K}_n{args.target_nodes}_{timestamp}.npz"
    
    # Convert results to numpy arrays for saving
    save_data = {
        'target_config': sweep_results['target_config'],
        'hyperparameters': sweep_results['hyperparameters']
    }
    
    for key, result in sweep_results['results'].items():
        save_data[f'errors_{key}'] = np.array(result['errors'])
        save_data[f'alpha_{key}'] = result['alpha']
        if result['beta'] is not None:
            save_data[f'beta_{key}'] = result['beta']
    
    np.savez(sweep_filename, **save_data)
    
    print(f"\nSweep results saved to: {sweep_filename}")
    
    # Generate sweep plots
    plot_hyperparameter_sweep(sweep_results, sweep_filename)
    
    return sweep_results, sweep_filename


def plot_hyperparameter_sweep(sweep_results, filename):
    """Generate plots for hyperparameter sweep results."""
    print("Generating hyperparameter sweep plots...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Hyperparameter Sweep: {sweep_results["target_config"]["dataset"].upper()} '
                f'K={sweep_results["target_config"]["K"]} '
                f'n={sweep_results["target_config"]["num_nodes"]}', fontsize=16)
    
    # Plot 1: All M-DSA curves with different alpha/beta combinations
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(sweep_results['results'])))
    
    for i, (key, result) in enumerate(sweep_results['results'].items()):
        if result['algorithm'] == 'M-DSA':
            label = f"α={result['alpha']}, β={result['beta']}"
            ax1.semilogy(result['errors'], label=label, color=colors[i], linewidth=2)
    
    ax1.set_title('M-DSA Hyperparameter Sweep')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cosine Angle Distance (log scale)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Best M-DSA vs DSA baselines
    ax2 = axes[0, 1]
    
    # Find best M-DSA
    best_mdsa_key = None
    best_mdsa_error = float('inf')
    for key, result in sweep_results['results'].items():
        if result['algorithm'] == 'M-DSA' and len(result['errors']) > 0:
            if result['errors'][-1] < best_mdsa_error:
                best_mdsa_error = result['errors'][-1]
                best_mdsa_key = key
    
    # Plot DSA baselines
    for key, result in sweep_results['results'].items():
        if result['algorithm'] == 'DSA':
            label = f"DSA α={result['alpha']}"
            ax2.semilogy(result['errors'], label=label, linestyle='--', linewidth=2)
    
    # Plot best M-DSA
    if best_mdsa_key:
        best_result = sweep_results['results'][best_mdsa_key]
        label = f"Best M-DSA α={best_result['alpha']}, β={best_result['beta']}"
        ax2.semilogy(best_result['errors'], label=label, color='red', linewidth=3)
    
    ax2.set_title('Best M-DSA vs DSA Baselines')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cosine Angle Distance (log scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final error heatmap for alpha vs beta
    ax3 = axes[1, 0]
    
    # Create heatmap data
    alpha_list = sweep_results['hyperparameters']['alpha_list']
    beta_list = sweep_results['hyperparameters']['beta_list']
    heatmap_data = np.zeros((len(beta_list), len(alpha_list)))
    
    for i, beta in enumerate(beta_list):
        for j, alpha in enumerate(alpha_list):
            key = f'M-DSA_alpha{alpha}_beta{beta}'
            if key in sweep_results['results']:
                errors = sweep_results['results'][key]['errors']
                if len(errors) > 0:
                    heatmap_data[i, j] = errors[-1]
                else:
                    heatmap_data[i, j] = np.nan
            else:
                heatmap_data[i, j] = np.nan
    
    im = ax3.imshow(heatmap_data, cmap='viridis', aspect='auto')
    ax3.set_xticks(range(len(alpha_list)))
    ax3.set_xticklabels([f'{a:.3f}' for a in alpha_list])
    ax3.set_yticks(range(len(beta_list)))
    ax3.set_yticklabels([f'{b:.2f}' for b in beta_list])
    ax3.set_xlabel('Alpha (α)')
    ax3.set_ylabel('Beta (β)')
    ax3.set_title('Final Error Heatmap (M-DSA)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Final Error')
    
    # Add text annotations
    for i in range(len(beta_list)):
        for j in range(len(alpha_list)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax3.text(j, i, f'{heatmap_data[i, j]:.3f}',
                              ha="center", va="center", color="white", fontsize=8)
    
    # Plot 4: Improvement over best DSA
    ax4 = axes[1, 1]
    
    # Find best DSA
    best_dsa_error = float('inf')
    for key, result in sweep_results['results'].items():
        if result['algorithm'] == 'DSA' and len(result['errors']) > 0:
            if result['errors'][-1] < best_dsa_error:
                best_dsa_error = result['errors'][-1]
    
    # Calculate improvements
    improvements = []
    labels = []
    for key, result in sweep_results['results'].items():
        if result['algorithm'] == 'M-DSA' and len(result['errors']) > 0:
            improvement = (best_dsa_error - result['errors'][-1]) / best_dsa_error * 100
            improvements.append(improvement)
            labels.append(f"α={result['alpha']}, β={result['beta']}")
    
    if improvements:
        bars = ax4.bar(range(len(improvements)), improvements, alpha=0.7)
        ax4.set_xticks(range(len(improvements)))
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.set_ylabel('Improvement over Best DSA (%)')
        ax4.set_title('M-DSA Improvement over Best DSA')
        ax4.grid(True, alpha=0.3)
        
        # Color bars based on improvement
        for i, bar in enumerate(bars):
            if improvements[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = filename.replace('.npz', '_sweep_plot.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Sweep plot saved to: {plot_filename}")


def main():
    """Main experiment function."""
    # Parse arguments
    args = parse_arguments()
    
    # Check if sweep mode is enabled
    if args.sweep_mode or (args.alpha_list is not None) or (args.beta_list is not None):
        return run_hyperparameter_sweep(args)
    
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
