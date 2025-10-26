#!/usr/bin/env python3
"""
Standalone script to plot results from saved experiment files.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_results(results_file: str, save_plot: bool = True, show_plot: bool = True) -> str:
    """
    Generate and save plots comparing the algorithms.
    
    Args:
        results_file: Path to the saved results file
        save_plot: Whether to save the plot to file
        show_plot: Whether to display the plot
        
    Returns:
        plot_file: Path to the saved plot file (if saved)
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
    
    # Get experiment info
    args = data['args'].item()
    data_info = data['data_info'].item()
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    for i, (algo, errors) in enumerate(zip(algorithms, error_lists)):
        if len(errors) > 0:
            plt.semilogy(errors, label=algo, color=colors[i % len(colors)], 
                        linestyle=linestyles[i % len(linestyles)], linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cosine Angle Distance (log scale)', fontsize=12)
    
    # Create informative title
    title = f'Algorithm Comparison - {data_info["type"].upper()} Dataset'
    if data_info["type"] == "synthetic":
        title += f' (d={data_info["d"]}, N={data_info["N"]}, eigengap={data_info["eigengap"]})'
    else:
        title += f' (d={data_info["d"]}, N={data_info["N"]})'
    title += f' - K={args["K"]}, Nodes={args["num_nodes"]}'
    
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_file = None
    if save_plot:
        plot_file = results_file.replace('.npz', '_plot.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")
    
    # Show plot
    if show_plot:
        plt.show()
    
    return plot_file


def print_summary(results_file: str):
    """Print a summary of the experiment results."""
    data = np.load(results_file, allow_pickle=True)
    
    args = data['args'].item()
    data_info = data['data_info'].item()
    
    print("=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Dataset: {args['dataset']}")
    print(f"Components: {args['K']}")
    print(f"Nodes: {args['num_nodes']}")
    print(f"Iterations: {args['max_iters']}")
    print(f"Alpha: {args['alpha']}")
    print(f"Beta: {args['beta']}")
    
    if data_info['type'] == 'synthetic':
        print(f"Dimension: {data_info['d']}")
        print(f"Samples: {data_info['N']}")
        print(f"Eigenvalue gap: {data_info['eigengap']}")
    else:
        print(f"Dimension: {data_info['d']}")
        print(f"Samples: {data_info['N']}")
    
    print(f"\nFinal Errors:")
    for key in data.keys():
        if key.startswith('errors_'):
            algo_name = key[7:]
            errors = data[key]
            if len(errors) > 0:
                print(f"  {algo_name}: {errors[-1]:.6f}")
            else:
                print(f"  {algo_name}: FAILED")
    
    print("=" * 80)


def main():
    """Main function for plotting results."""
    parser = argparse.ArgumentParser(description='Plot results from momentum experiment')
    parser.add_argument('results_file', type=str, help='Path to the results .npz file')
    parser.add_argument('--no-save', action='store_true', help='Do not save the plot')
    parser.add_argument('--no-show', action='store_true', help='Do not display the plot')
    parser.add_argument('--summary', action='store_true', help='Print experiment summary')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file '{args.results_file}' not found.")
        return
    
    if args.summary:
        print_summary(args.results_file)
    
    plot_results(args.results_file, 
                save_plot=not args.no_save, 
                show_plot=not args.no_show)


if __name__ == "__main__":
    main()
