
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import json

# Import centralized ABM model
from run_visualisation_03 import MobilityModel

# Import decentralized ABM model
from decentralized_abm_model import DecentralizedMaaSModel

def run_comparison_experiment(experiment_params, output_dir="comparison_results"):
    """
    Run comparison experiment between centralized and decentralized models

    Parameters:
    - experiment_params: Dict of experiment parameters
    - output_dir: Directory to save results
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract common parameters
    common_params = {
        'num_commuters': experiment_params.get('num_commuters', 100),
        'grid_width': experiment_params.get('grid_width', 50),
        'grid_height': experiment_params.get('grid_height', 50),
        'income_weights': experiment_params.get('income_weights', [0.5, 0.3, 0.2]),
        'health_weights': experiment_params.get('health_weights', [0.9, 0.1]),
        'payment_weights': experiment_params.get('payment_weights', [0.8, 0.2]),
        'age_distribution': experiment_params.get('age_distribution',
                                               {(18, 25): 0.2, (26, 35): 0.3, (36, 45): 0.2,
                                                (46, 55): 0.15, (56, 65): 0.1, (66, 75): 0.05}),
        'disability_weights': experiment_params.get('disability_weights', [0.15, 0.85]),
        'tech_access_weights': experiment_params.get('tech_access_weights', [0.95, 0.05]),
        'uber_like1_capacity': experiment_params.get('uber_like1_capacity', 15),
        'uber_like1_price': experiment_params.get('uber_like1_price', 15),
        'uber_like2_capacity': experiment_params.get('uber_like2_capacity', 18),
        'uber_like2_price': experiment_params.get('uber_like2_price', 16),
        'bike_share1_capacity': experiment_params.get('bike_share1_capacity', 10),
        'bike_share1_price': experiment_params.get('bike_share1_price', 2),
        'bike_share2_capacity': experiment_params.get('bike_share2_capacity', 12),
        'bike_share2_price': experiment_params.get('bike_share2_price', 3),
        'stations': experiment_params.get('stations', {}),
        'routes': experiment_params.get('routes', {}),
        'transfers': experiment_params.get('transfers', {})
    }

    # Extract model-specific parameters
    centralized_params = experiment_params.get('centralized_params', {})
    decentralized_params = experiment_params.get('decentralized_params', {})

    # Number of steps to run each model
    num_steps = experiment_params.get('num_steps', 100)

    # Run centralized model
    print(f"Running centralized model with {common_params['num_commuters']} commuters for {num_steps} steps...")
    centralized_start_time = time.time()

    # Combine parameters
    centralized_full_params = {**common_params, **centralized_params}

    centralized_model = MobilityModel(**centralized_full_params)
    centralized_model.run_model(num_steps)

    centralized_end_time = time.time()
    centralized_runtime = centralized_end_time - centralized_start_time
    print(f"Centralized model completed in {centralized_runtime:.2f} seconds")

    # Run decentralized model
    print(f"Running decentralized model with {common_params['num_commuters']} commuters for {num_steps} steps...")
    decentralized_start_time = time.time()

    # Combine parameters
    decentralized_full_params = {**common_params, **decentralized_params}

    # Add market type parameter
    market_type = experiment_params.get('market_type', 'hybrid')
    decentralized_full_params['market_type'] = market_type

    decentralized_model = DecentralizedMaaSModel(**decentralized_full_params)
    decentralized_model.run_model(num_steps)

    decentralized_end_time = time.time()
    decentralized_runtime = decentralized_end_time - decentralized_start_time
    print(f"Decentralized model completed in {decentralized_runtime:.2f} seconds")

    # Collect results
    results = {
        'parameters': experiment_params,
        'centralized': {
            'runtime': centralized_runtime,
            'completed_trips': centralized_model.service_provider_agent.completed_trips_count,
            'revenue': centralized_model.service_provider_agent.total_revenue,
            'average_price': centralized_model.service_provider_agent.average_price,
            'mode_share': calculate_mode_share(centralized_model)
        },
        'decentralized': {
            'runtime': decentralized_runtime,
            'completed_trips': decentralized_model.count_completed_trips(),
            'active_nfts': sum(1 for listing in decentralized_model.marketplace.listings.values()
                             if listing['status'] == 'active'),
            'average_price': decentralized_model.calculate_average_nft_price(),
            'bundle_transactions': decentralized_model.count_bundle_transactions(),
            'mode_share': calculate_mode_share_decentralized(decentralized_model)
        }
    }

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"comparison_results_{timestamp}.json")

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    # Create visualizations
    create_visualizations(results, output_dir, timestamp)

    return results

def calculate_mode_share(centralized_model):
    """Calculate mode share for centralized model"""
    # Extract mode usage from logs
    mode_counts = {}

    for commuter in centralized_model.commuter_agents:
        for request in commuter.requests.values():
            if 'selected_route' in request and request['status'] == 'finished':
                mode = request['selected_route'].get('mode', 'unknown')
                mode_counts[mode] = mode_counts.get(mode, 0) + 1

    # Calculate percentages
    total_trips = sum(mode_counts.values())
    if total_trips == 0:
        return {}

    return {mode: count / total_trips for mode, count in mode_counts.items()}

def calculate_mode_share_decentralized(decentralized_model):
    """Calculate mode share for decentralized model"""
    # Extract mode usage from transaction history
    mode_counts = {}

    for transaction in decentralized_model.marketplace.transaction_history:
        nft_id = transaction['nft_id']
        if nft_id in decentralized_model.marketplace.listings:
            listing = decentralized_model.marketplace.listings[nft_id]
            details = listing['details']
            mode = details.get('service_type', 'unknown')
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

    # Add bundle transactions
    bundle_count = decentralized_model.count_bundle_transactions()
    if bundle_count > 0:
        mode_counts['bundle'] = bundle_count

    # Calculate percentages
    total_trips = sum(mode_counts.values())
    if total_trips == 0:
        return {}

    return {mode: count / total_trips for mode, count in mode_counts.items()}

def create_visualizations(results, output_dir, timestamp):
    """Create visualization of comparison results"""
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Runtime comparison
    axes[0, 0].bar(['Centralized', 'Decentralized'],
                 [results['centralized']['runtime'], results['decentralized']['runtime']])
    axes[0, 0].set_title('Runtime Comparison (seconds)')
    axes[0, 0].set_ylabel('Seconds')

    # 2. Completed trips comparison
    axes[0, 1].bar(['Centralized', 'Decentralized'],
                 [results['centralized']['completed_trips'], results['decentralized']['completed_trips']])
    axes[0, 1].set_title('Completed Trips Comparison')
    axes[0, 1].set_ylabel('Count')

    # 3. Price comparison
    axes[1, 0].bar(['Centralized', 'Decentralized'],
                 [results['centralized']['average_price'], results['decentralized']['average_price']])
    axes[1, 0].set_title('Average Price Comparison')
    axes[1, 0].set_ylabel('Price')

    # 4. Mode share comparison
    centralized_modes = results['centralized']['mode_share']
    decentralized_modes = results['decentralized']['mode_share']

    # Combine all modes
    all_modes = set(centralized_modes.keys()) | set(decentralized_modes.keys())

    # Prepare data for grouped bar chart
    mode_names = list(all_modes)
    centralized_shares = [centralized_modes.get(mode, 0) * 100 for mode in mode_names]
    decentralized_shares = [decentralized_modes.get(mode, 0) * 100 for mode in mode_names]

    x = np.arange(len(mode_names))
    width = 0.35

    axes[1, 1].bar(x - width/2, centralized_shares, width, label='Centralized')
    axes[1, 1].bar(x + width/2, decentralized_shares, width, label='Decentralized')

    axes[1, 1].set_title('Mode Share Comparison')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(mode_names, rotation=45, ha='right')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comparison_results_{timestamp}.png"), dpi=300)
    plt.close()

    # Create more detailed visualizations

    # NFT Marketplace activity (decentralized only)
    if 'active_nfts' in results['decentralized']:
        plt.figure(figsize=(10, 6))
        plt.bar(['Active NFTs', 'Bundle Transactions'],
               [results['decentralized']['active_nfts'],
                results['decentralized']['bundle_transactions']])
        plt.title('Decentralized Marketplace Activity')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, f"marketplace_activity_{timestamp}.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    # Define experiment parameters
    experiment_params = {
        'num_commuters': 100,
        'grid_width': 50,
        'grid_height': 50,
        'num_steps': 100,
        'market_type': 'hybrid',  # Use hybrid market model
        'centralized_params': {
            # Add any specific parameters for centralized model
            'db_connection_string': "postgresql://username:password@localhost:5432/mobility"
        },
        'decentralized_params': {
            # Add any specific parameters for decentralized model
            'blockchain_config': "blockchain_config.json"
        }
    }

    # Run comparison experiment
    results = run_comparison_experiment(experiment_params)

    print("Experiment completed successfully!")
