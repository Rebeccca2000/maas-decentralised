import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import json
import argparse
from datetime import datetime

# Import decentralized ABM model
from decentralized_abm_model import DecentralizedMaaSModel

def run_decentralized_model(params, steps, output_dir="results"):
    """
    Run the decentralized MaaS model with given parameters.
    
    Args:
        params: Dict of model parameters
        steps: Number of steps to run
        output_dir: Directory to save results
    
    Returns:
        The model instance and results dataframe
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Record start time
    start_time = time.time()
    
    # Initialize model
    print(f"Initializing decentralized model with {params['num_commuters']} commuters...")
    model = DecentralizedMaaSModel(**params)
    
    # Run model
    print(f"Running model for {steps} steps...")
    results_df = model.run_model(steps)
    

    print("Waiting for pending transactions to complete...")
    # Actively process transactions during wait instead of passive sleep
    for _ in range(6):  # Do 6 cycles of processing (equivalent to 30 seconds)
        model.blockchain_interface._check_pending_transactions()
        model.blockchain_interface.update_cache()
        time.sleep(5)  # Wait 5 seconds between each check
    # time.sleep(30)  # Wait 30 seconds
    # Record end time
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Model run completed in {run_time:.2f} seconds")
    
    # Save runtime stats
    runtime_stats = {
        'total_runtime': run_time,
        'steps': steps,
        'avg_step_time': run_time / steps,
        'parameters': params,
        'commuters': params['num_commuters'],
        'execution_times': model.get_average_execution_time(),
        'completed_trips': model.count_completed_trips(),
        'active_nfts': sum(1 for listing in model.marketplace.listings.values() 
                        if listing['status'] == 'active'),
        'average_price': model.calculate_average_nft_price(),
        # IMPORTANT FIX: Get transaction count from blockchain if marketplace is empty
        'transaction_count': (
            len(model.marketplace.transaction_history) if model.marketplace.transaction_history
            else model.blockchain_interface.stats.get('transactions_confirmed', 0)
        )
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save runtime stats to JSON
    stats_file = os.path.join(output_dir, f"runtime_stats_{timestamp}.json")
    def convert_keys_to_str(obj):
        """Convert dictionary keys from tuples to strings for JSON serialization"""
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, tuple) else k: convert_keys_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_keys_to_str(item) for item in obj]
        else:
            return obj

    # Use this before JSON dumping
    runtime_stats = convert_keys_to_str(runtime_stats)

    with open(stats_file, 'w') as f:
        json.dump(runtime_stats, f, indent=4)
    
    # Save results dataframe to CSV
    if results_df is not None:
        results_file = os.path.join(output_dir, f"model_data_{timestamp}.csv")
        results_df.to_csv(results_file, index=False)
    
    # Create visualizations
    visualize_results(model, timestamp, output_dir)
    
    print(f"Results saved to {output_dir}")
    return model, results_df

def visualize_results(model, timestamp, output_dir):
    """
    Create visualizations from the model run
    
    Args:
        model: The DecentralizedMaaSModel instance
        timestamp: Timestamp string for file naming
        output_dir: Directory to save visualizations
    """
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Active Commuters over time
    if hasattr(model.datacollector, 'get_model_vars_dataframe'):
        model_vars = model.datacollector.get_model_vars_dataframe()
        if 'Active Commuters' in model_vars:
            axes[0, 0].plot(model_vars.index, model_vars['Active Commuters'])
            axes[0, 0].set_title('Active Commuters Over Time')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Count')
    
    # 2. NFT Prices over time
    if hasattr(model, 'marketplace') and hasattr(model.marketplace, 'price_history'):
        # Find most active routes
        top_routes = []
        for route, history in model.marketplace.price_history.items():
            if len(history) > 5:  # Only consider routes with sufficient data
                top_routes.append((route, len(history)))
        
        # Sort by activity and take top 3
        top_routes.sort(key=lambda x: x[1], reverse=True)
        top_routes = top_routes[:3]
        
        for route, _ in top_routes:
            history = model.marketplace.price_history[route]
            times = [t for t, _ in history]
            prices = [p for _, p in history]
            axes[0, 1].plot(times, prices, label=f"Route {route[:10]}...")
            
        axes[0, 1].set_title('NFT Prices Over Time (Top Routes)')
        axes[0, 1].set_xlabel('Model Time')
        axes[0, 1].set_ylabel('Price')
        axes[0, 1].legend()
    
    # 3. Transaction Volume
    if hasattr(model, 'marketplace'):
        # Group transactions by time (steps)
        tx_times = [tx['time'] for tx in model.marketplace.transaction_history]
        # Use step intervals of 10
        interval = 10
        max_step = model.schedule.time
        steps = list(range(0, max_step + interval, interval))
        counts = []
        
        for i in range(len(steps) - 1):
            count = sum(1 for t in tx_times if steps[i] <= t < steps[i + 1])
            counts.append(count)
            
        axes[1, 0].bar(steps[:-1], counts)
        axes[1, 0].set_title('Transaction Volume Over Time')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Number of Transactions')
    
    # 4. Mode Share
    if hasattr(model, 'marketplace'):
        mode_counts = {}
        for tx in model.marketplace.transaction_history:
            # Check if tx has nft_id key
            if 'nft_id' not in tx:
                continue
                
            nft_id = tx['nft_id']
            if nft_id in model.marketplace.listings:
                listing = model.marketplace.listings[nft_id]
                mode = listing.get('details', {}).get('mode', 'unknown')
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
                
        if mode_counts:
            modes = list(mode_counts.keys())
            counts = [mode_counts[mode] for mode in modes]
            axes[1, 1].bar(modes, counts)
            axes[1, 1].set_title('Mode Share')
            axes[1, 1].set_xlabel('Transport Mode')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"overview_{timestamp}.png"), dpi=300)
    plt.close()
    
    # Create more detailed visualizations
    
    
    # Execution time breakdown
    if hasattr(model, 'execution_times') and model.execution_times:
        avg_times = {}
        for component, times in model.execution_times.items():
            if times:
                avg_times[component] = sum(times) / len(times)
        
        if avg_times:
            plt.figure(figsize=(12, 6))
            components = list(avg_times.keys())
            times = [avg_times[c] for c in components]
            plt.bar(components, times)
            plt.title('Average Execution Time by Component')
            plt.ylabel('Time (seconds)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"execution_times_{timestamp}.png"), dpi=300)
            plt.close()

def get_default_params():
    """Return default parameters for the decentralized model"""
    return {
        'num_commuters': 20,  # Start small for debugging
        'grid_width': 50,
        'grid_height': 50,
        'income_weights': [0.3, 0.5, 0.2],  # Low, middle, high
        'health_weights': [0.8, 0.2],  # Good, poor
        'payment_weights': [0.7, 0.3],  # PAYG, subscription
        'age_distribution': {(18, 30): 0.25, (31, 45): 0.3, (46, 65): 0.3, (66, 85): 0.15},
        'disability_weights': [0.1, 0.9],  # True, False
        'tech_access_weights': [0.85, 0.15],  # True, False
        'uber_like1_capacity': 10,
        'uber_like1_price': 1.2,
        'uber_like2_capacity': 8,
        'uber_like2_price': 1.0,
        'bike_share1_capacity': 20,
        'bike_share1_price': 0.5,
        'bike_share2_capacity': 15,
        'bike_share2_price': 0.4,
        'stations': None,  # No stations for simplicity
        'routes': None,    # No predefined routes
        'transfers': None, # No transfers
        'blockchain_config': "blockchain_config.json",
        'market_type': "hybrid",  # Use hybrid market model
        'time_decay_factor': 0.1,
        'min_price_ratio': 0.5
    }

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run Decentralized MaaS Model')
    parser.add_argument('--steps', type=int, default=250,
                        help='Number of steps to run the simulation')
    parser.add_argument('--commuters', type=int, default=50,
                        help='Number of commuter agents')
    parser.add_argument('--market', type=str, default='hybrid', choices=['order_book', 'hybrid'],
                        help='Market mechanism to use')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration JSON file')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with minimal agents')
    
    args = parser.parse_args()
    
    # Get default parameters
    params = get_default_params()
    
    # Update with command line arguments
    params['num_commuters'] = args.commuters
    params['market_type'] = args.market
    
    # Debug mode
    if args.debug:
        print("Running in debug mode with minimal configuration")
        params['num_commuters'] = 5
        params['uber_like1_capacity'] = 3
        params['uber_like2_capacity'] = 3
        params['bike_share1_capacity'] = 5
        params['bike_share2_capacity'] = 5
    
    # Load config from file if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_params = json.load(f)
                # Update params with config file values
                params.update(config_params)
            print(f"Loaded configuration from {args.config}")
        except Exception as e:
            print(f"Error loading config file: {e}")
    
    # Run the model
    model, results = run_decentralized_model(params, steps=args.steps, output_dir=args.output)
    
    print("Simulation complete!")
    print(f"- Completed trips: {model.count_completed_trips()}")
    print(f"- Active NFT listings: {sum(1 for listing in model.marketplace.listings.values() if listing['status'] == 'active')}")
    # IMPORTANT FIX: Use blockchain transaction count if marketplace is empty
    print(f"- Total transactions: {len(model.marketplace.transaction_history) or model.blockchain_interface.stats.get('transactions_confirmed', 0)}")
    print(f"- Average NFT price: {model.calculate_average_nft_price():.2f}")
    # Blockchain stats
    print("Blockchain Statistics:")
    stats = model.blockchain_interface.get_stats()
    print(f"- Transactions submitted: {stats['transactions_submitted']}")
    print(f"- Transactions confirmed: {stats['transactions_confirmed']}")
    print(f"- Cache hits: {stats['cache_hits']}")
    print(f"- Cache misses: {stats['cache_misses']}")
    
    # Show performance stats
    exec_times = model.get_average_execution_time()
    print("\nPerformance Breakdown:")
    for component, time_value in exec_times.items():
        print(f"- {component}: {time_value:.4f} seconds")