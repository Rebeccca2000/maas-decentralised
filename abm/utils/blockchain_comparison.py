# abm/utils/blockchain_comparison.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abm.simulation import MobilityModel
import json
import os

class MaaSComparison:
    """Class to compare centralized and decentralized MaaS models"""
    
    def __init__(self, config_file="comparison_config.json"):
        """Initialize with configuration"""
        self.config = self._load_config(config_file)
        self.results = {
            "centralized": {},
            "decentralized": {},
            "comparison": {}
        }
        
    def _load_config(self, config_file):
        """Load comparison configuration"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "model_params": {
                    "db_connection_string": "postgresql://user:password@localhost:5432/maas_db",
                    "num_commuters": 100,
                    "grid_width": 50,
                    "grid_height": 50,
                    "simulation_steps": 100
                },
                "centralized_params": {
                    "use_blockchain": False
                },
                "decentralized_params": {
                    "use_blockchain": True,
                    "blockchain_config": "blockchain_config.json"
                },
                "comparison_metrics": [
                    "mode_share",
                    "travel_time",
                    "travel_cost",
                    "provider_revenue",
                    "equity_score"
                ]
            }
    
    def run_centralized_model(self):
        """Run the centralized MaaS model"""
        print("Running centralized MaaS model...")
        
        # Create model with centralized parameters
        model_params = self.config["model_params"].copy()
        model_params.update(self.config["centralized_params"])
        
        # Create a schema name for this run
        schema_name = f"centralized_{int(time.time())}"
        model_params["schema"] = schema_name
        
        # Create and run model
        model = MobilityModel(**model_params)
        model.run_model(self.config["model_params"]["simulation_steps"])
        
        # Collect results
        self.results["centralized"] = self._collect_model_metrics(model)
        self.results["centralized"]["schema"] = schema_name
        
        print("Centralized model run complete")
        
    def run_decentralized_model(self):
        """Run the decentralized MaaS model"""
        print("Running decentralized MaaS model...")
        
        # Create model with decentralized parameters
        model_params = self.config["model_params"].copy()
        model_params.update(self.config["decentralized_params"])
        
        # Create a schema name for this run
        schema_name = f"decentralized_{int(time.time())}"
        model_params["schema"] = schema_name
        
        # Create and run model
        model = MobilityModel(**model_params)
        model.run_model(self.config["model_params"]["simulation_steps"])
        
        # Collect results
        self.results["decentralized"] = self._collect_model_metrics(model)
        self.results["decentralized"]["schema"] = schema_name
        
        print("Decentralized model run complete")
    
    def _collect_model_metrics(self, model):
        """Collect metrics from a model run"""
        metrics = {}
        
        # Mode share
        modes = ['walk', 'bike', 'car', 'public', 'MaaS_Bundle']
        mode_counts = {mode: 0 for mode in modes}
        total_trips = 0
        
        for commuter in model.commuter_agents:
            for request in commuter.requests.values():
                if request['status'] in ['Service Selected', 'finished']:
                    total_trips += 1
                    mode = request.get('selected_route', {}).get('mode', '').split('_')[0]
                    # Default to 'unknown' if mode not recognized
                    mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        # Calculate percentages
        if total_trips > 0:
            mode_share = {mode: count / total_trips * 100 for mode, count in mode_counts.items()}
        else:
            mode_share = {mode: 0 for mode in mode_counts}
            
        metrics["mode_share"] = mode_share
        metrics["total_trips"] = total_trips
        
        # Average travel time and cost
        total_time = 0
        total_cost = 0
        completed_trips = 0
        
        for commuter in model.commuter_agents:
            for request in commuter.requests.values():
                if request['status'] == 'finished':
                    completed_trips += 1
                    selected_route = request.get('selected_route', {})
                    total_time += selected_route.get('time', 0)
                    total_cost += selected_route.get('price', 0)
        
        if completed_trips > 0:
            metrics["avg_travel_time"] = total_time / completed_trips
            metrics["avg_travel_cost"] = total_cost / completed_trips
        else:
            metrics["avg_travel_time"] = 0
            metrics["avg_travel_cost"] = 0
            
        # Provider revenue
        provider_revenue = {}
        
        # Query database for revenue by provider
        with model.Session() as session:
            # Set schema if needed
            if model.schema:
                session.execute(text(f"SET search_path TO {model.schema}"))
                
            # Query revenue from service_booking_log
            query = """
                SELECT record_company_name, SUM(total_price) as revenue
                FROM service_booking_log
                GROUP BY record_company_name
            """
            
            results = session.execute(text(query)).fetchall()
            for row in results:
                provider_revenue[row[0]] = float(row[1])
                
        metrics["provider_revenue"] = provider_revenue
        
        # Equity metrics by income level
        equity_by_income = {}
        income_levels = ['low', 'middle', 'high']
        
        for income in income_levels:
            # Filter commuters by income level
            income_commuters = [c for c in model.commuter_agents if c.income_level == income]
            income_trips = 0
            income_cost = 0
            income_time = 0
            
            for commuter in income_commuters:
                for request in commuter.requests.values():
                    if request['status'] == 'finished':
                        income_trips += 1
                        selected_route = request.get('selected_route', {})
                        income_time += selected_route.get('time', 0)
                        income_cost += selected_route.get('price', 0)
            
            if income_trips > 0:
                equity_by_income[income] = {
                    "trips": income_trips,
                    "avg_time": income_time / income_trips,
                    "avg_cost": income_cost / income_trips
                }
            else:
                equity_by_income[income] = {
                    "trips": 0,
                    "avg_time": 0,
                    "avg_cost": 0
                }
                
        metrics["equity_by_income"] = equity_by_income
        
        # Calculate equity score
        # Lower score means more equitable
        equity_scores = {}
        
        if all(equity_by_income[income]["trips"] > 0 for income in income_levels):
            # Travel time equity
            avg_times = [equity_by_income[income]["avg_time"] for income in income_levels]
            time_std = np.std(avg_times)
            time_mean = np.mean(avg_times)
            time_cv = time_std / time_mean if time_mean > 0 else 0
            equity_scores["time_equity"] = time_cv
            
            # Travel cost equity
            avg_costs = [equity_by_income[income]["avg_cost"] for income in income_levels]
            cost_std = np.std(avg_costs)
            cost_mean = np.mean(avg_costs)
            cost_cv = cost_std / cost_mean if cost_mean > 0 else 0
            equity_scores["cost_equity"] = cost_cv
            
            # Trip rate equity (trips per person)
            commuters_by_income = {income: len([c for c in model.commuter_agents if c.income_level == income]) 
                                   for income in income_levels}
            trip_rates = [equity_by_income[income]["trips"] / commuters_by_income[income] 
                          if commuters_by_income[income] > 0 else 0 
                          for income in income_levels]
            rate_std = np.std(trip_rates)
            rate_mean = np.mean(trip_rates)
            rate_cv = rate_std / rate_mean if rate_mean > 0 else 0
            equity_scores["trip_rate_equity"] = rate_cv
            
            # Overall equity score (weighted average)
            equity_scores["overall"] = (
                equity_scores["time_equity"] * 0.3 + 
                equity_scores["cost_equity"] * 0.4 + 
                equity_scores["trip_rate_equity"] * 0.3
            )
        else:
            equity_scores = {
                "time_equity": 0,
                "cost_equity": 0,
                "trip_rate_equity": 0,
                "overall": 0
            }
            
        metrics["equity_scores"] = equity_scores
        
        # Blockchain-specific metrics if applicable
        if hasattr(model, 'use_blockchain') and model.use_blockchain:
            blockchain_metrics = {
                "onchain_requests": 0,
                "onchain_services": 0,
                "nft_trades": 0,
                "avg_auction_time": 0
            }
            
            # Count on-chain requests
            onchain_requests = sum(1 for c in model.commuter_agents
                                  for r in c.requests.values()
                                  if r.get('blockchain_id') is not None)
            blockchain_metrics["onchain_requests"] = onchain_requests
            
            # Count NFT trades
            # This would require additional tracking in the model
            
            metrics["blockchain"] = blockchain_metrics
            
        return metrics
    
    def compare_results(self):
        """Compare results between centralized and decentralized models"""
        if not self.results["centralized"] or not self.results["decentralized"]:
            print("Both models must be run before comparison")
            return
            
        comparison = {}
        
        # Mode share comparison
        central_mode_share = self.results["centralized"]["mode_share"]
        decentral_mode_share = self.results["decentralized"]["mode_share"]
        
        mode_share_diff = {}
        for mode in set(central_mode_share.keys()).union(decentral_mode_share.keys()):
            central_val = central_mode_share.get(mode, 0)
            decentral_val = decentral_mode_share.get(mode, 0)
            mode_share_diff[mode] = decentral_val - central_val
            
        comparison["mode_share_diff"] = mode_share_diff
        
        # Travel metrics comparison
        comparison["travel_time_diff"] = (
            self.results["decentralized"]["avg_travel_time"] - 
            self.results["centralized"]["avg_travel_time"]
        )
        
        comparison["travel_cost_diff"] = (
            self.results["decentralized"]["avg_travel_cost"] - 
            self.results["centralized"]["avg_travel_cost"]
        )
        
        # Equity comparison
        comparison["equity_diff"] = (
            self.results["decentralized"]["equity_scores"]["overall"] - 
            self.results["centralized"]["equity_scores"]["overall"]
        )
        
        # Provider revenue comparison
        central_revenue = self.results["centralized"]["provider_revenue"]
        decentral_revenue = self.results["decentralized"]["provider_revenue"]
        
        revenue_diff = {}
        for provider in set(central_revenue.keys()).union(decentral_revenue.keys()):
            central_val = central_revenue.get(provider, 0)
            decentral_val = decentral_revenue.get(provider, 0)
            revenue_diff[provider] = decentral_val - central_val
            
        comparison["revenue_diff"] = revenue_diff
        
        # Overall statistics
        comparison["total_trips_diff"] = (
            self.results["decentralized"]["total_trips"] - 
            self.results["centralized"]["total_trips"]
        )
        
        # Blockchain efficiency metrics
        if "blockchain" in self.results["decentralized"]:
            comparison["blockchain_usage"] = {
                "pct_onchain_requests": (
                    self.results["decentralized"]["blockchain"]["onchain_requests"] / 
                    self.results["decentralized"]["total_trips"] * 100
                    if self.results["decentralized"]["total_trips"] > 0 else 0
                )
            }
            
        self.results["comparison"] = comparison
        return comparison
        
    def visualize_results(self, output_dir="comparison_results"):
        """Visualize comparison results with charts and tables"""
        if not self.results["comparison"]:
            self.compare_results()
            
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Mode share comparison
        plt.figure(figsize=(12, 6))
        modes = list(self.results["comparison"]["mode_share_diff"].keys())
        differences = list(self.results["comparison"]["mode_share_diff"].values())
        
        bars = plt.bar(modes, differences)
        
        # Color bars based on value
        for i, diff in enumerate(differences):
            if diff > 0:
                bars[i].set_color('green')
            else:
                bars[i].set_color('red')
                
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title('Difference in Mode Share: Decentralized - Centralized')
        plt.ylabel('Percentage Points Difference')
        plt.grid(axis='y', alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, 'mode_share_diff.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Travel metrics comparison
        metrics = ['Travel Time', 'Travel Cost', 'Equity Score']
        values = [
            self.results["comparison"]["travel_time_diff"],
            self.results["comparison"]["travel_cost_diff"],
            self.results["comparison"]["equity_diff"]
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values)
        
        # Color bars based on value
        for i, diff in enumerate(values):
            if diff < 0:  # Lower is better for these metrics
                bars[i].set_color('green')
            else:
                bars[i].set_color('red')
                
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title('Difference in Travel Metrics: Decentralized - Centralized')
        plt.ylabel('Difference')
        plt.grid(axis='y', alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, 'travel_metrics_diff.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Provider revenue comparison
        plt.figure(figsize=(12, 6))
        providers = list(self.results["comparison"]["revenue_diff"].keys())
        rev_diffs = list(self.results["comparison"]["revenue_diff"].values())
        
        bars = plt.bar(providers, rev_diffs)
        
        # Color bars based on value
        for i, diff in enumerate(rev_diffs):
            if diff > 0:
                bars[i].set_color('green')
            else:
                bars[i].set_color('red')
                
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title('Difference in Provider Revenue: Decentralized - Centralized')
        plt.ylabel('Revenue Difference')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.savefig(os.path.join(output_dir, 'revenue_diff.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Equity by income level comparison
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        income_levels = ['low', 'middle', 'high']
        central_times = [self.results["centralized"]["equity_by_income"][inc]["avg_time"] for inc in income_levels]
        decentral_times = [self.results["decentralized"]["equity_by_income"][inc]["avg_time"] for inc in income_levels]
        
        x = np.arange(len(income_levels))
        width = 0.35
        
        plt.bar(x - width/2, central_times, width, label='Centralized')
        plt.bar(x + width/2, decentral_times, width, label='Decentralized')
        
        plt.xlabel('Income Level')
        plt.ylabel('Average Travel Time')
        plt.title('Travel Time by Income')
        plt.xticks(x, income_levels)
        plt.legend()
        
        plt.subplot(132)
        central_costs = [self.results["centralized"]["equity_by_income"][inc]["avg_cost"] for inc in income_levels]
        decentral_costs = [self.results["decentralized"]["equity_by_income"][inc]["avg_cost"] for inc in income_levels]
        
        plt.bar(x - width/2, central_costs, width, label='Centralized')
        plt.bar(x + width/2, decentral_costs, width, label='Decentralized')
        
        plt.xlabel('Income Level')
        plt.ylabel('Average Travel Cost')
        plt.title('Travel Cost by Income')
        plt.xticks(x, income_levels)
        plt.legend()
        
        plt.subplot(133)
        central_trips = [self.results["centralized"]["equity_by_income"][inc]["trips"] for inc in income_levels]
        decentral_trips = [self.results["decentralized"]["equity_by_income"][inc]["trips"] for inc in income_levels]
        
        plt.bar(x - width/2, central_trips, width, label='Centralized')
        plt.bar(x + width/2, decentral_trips, width, label='Decentralized')
        
        plt.xlabel('Income Level')
        plt.ylabel('Number of Trips')
        plt.title('Trip Count by Income')
        plt.xticks(x, income_levels)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'equity_by_income.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Save summary table
        summary = {
            'Metric': [
                'Total Trips', 
                'Mode Share - Walk', 
                'Mode Share - Bike', 
                'Mode Share - Car', 
                'Mode Share - Public', 
                'Mode Share - MaaS', 
                'Avg Travel Time', 
                'Avg Travel Cost', 
                'Equity Score'
            ],
            'Centralized': [
                self.results["centralized"]["total_trips"],
                self.results["centralized"]["mode_share"].get('walk', 0),
                self.results["centralized"]["mode_share"].get('bike', 0),
                self.results["centralized"]["mode_share"].get('car', 0),
                self.results["centralized"]["mode_share"].get('public', 0),
                self.results["centralized"]["mode_share"].get('MaaS_Bundle', 0),
                self.results["centralized"]["avg_travel_time"],
                self.results["centralized"]["avg_travel_cost"],
                self.results["centralized"]["equity_scores"]["overall"]
            ],
            'Decentralized': [
                self.results["decentralized"]["total_trips"],
                self.results["decentralized"]["mode_share"].get('walk', 0),
                self.results["decentralized"]["mode_share"].get('bike', 0),
                self.results["decentralized"]["mode_share"].get('car', 0),
                self.results["decentralized"]["mode_share"].get('public', 0),
                self.results["decentralized"]["mode_share"].get('MaaS_Bundle', 0),
                self.results["decentralized"]["avg_travel_time"],
                self.results["decentralized"]["avg_travel_cost"],
                self.results["decentralized"]["equity_scores"]["overall"]
            ]
        }
        
        summary_df = pd.DataFrame(summary)
        summary_df['Difference'] = summary_df['Decentralized'] - summary_df['Centralized']
        summary_df['Pct Change'] = (summary_df['Difference'] / summary_df['Centralized'] * 100).round(2)
        
        summary_df.to_csv(os.path.join(output_dir, 'comparison_summary.csv'), index=False)
        
        # 6. If blockchain metrics available, visualize them
        if "blockchain" in self.results["decentralized"]:
            plt.figure(figsize=(8, 6))
            
            if "blockchain_usage" in self.results["comparison"]:
                pct_onchain = self.results["comparison"]["blockchain_usage"]["pct_onchain_requests"]
                plt.pie([pct_onchain, 100-pct_onchain], 
                       labels=['On-chain', 'Off-chain'],
                       autopct='%1.1f%%',
                       colors=['lightblue', 'lightgray'])
                
                plt.title('Percentage of Requests Processed On-chain')
                plt.savefig(os.path.join(output_dir, 'blockchain_usage.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
        # Save all results as JSON
        with open(os.path.join(output_dir, 'full_results.json'), 'w') as f:
            # Convert numpy values to native Python types
            def convert_to_json_serializable(obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(i) for i in obj]
                else:
                    return obj
            
            json_serializable_results = convert_to_json_serializable(self.results)
            json.dump(json_serializable_results, f, indent=2)
            
        print(f"Visualizations and results saved to {output_dir}")