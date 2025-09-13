# File: abm/agents/run_decentralized_model.py
# SIMPLIFIED VERSION - Run the marketplace-based MaaS simulation

import sys
import os
import time
import random
import logging
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.blockchain_interface import BlockchainInterface
from decentralized_commuter import DecentralizedCommuter
from decentralized_provider import DecentralizedProvider

class SimplifiedMaaSModel(Model):
    """
    Simplified MaaS model using marketplace architecture
    """
    
    def __init__(self, 
                 num_commuters=20,
                 num_providers=10,
                 width=25,
                 height=25):
        
        super().__init__()
        
        # Grid and schedule
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        
        # Initialize marketplace API (formerly blockchain_interface)
        self.marketplace = BlockchainInterface(
            config_path="../../blockchain_config.json",
            async_mode=True
        )
        
        # Metrics
        self.total_requests = 0
        self.total_matches = 0
        self.total_completed = 0
        
        # Create commuters
        for i in range(num_commuters):
            # Random attributes
            age = random.randint(18, 70)
            income = random.choice(['low', 'middle', 'high'])
            payment = random.choice(['PAYG', 'subscription'])
            health = 'poor' if random.random() < 0.1 else 'good'
            
            # Random position
            x = random.randrange(width)
            y = random.randrange(height)
            
            # Create agent
            commuter = DecentralizedCommuter(
                unique_id=i,
                model=self,
                pos=(x, y),
                age=age,
                income_level=income,
                payment_scheme=payment,
                health_status=health,
                blockchain_interface=self.marketplace
            )
            
            # Add to grid and schedule
            self.grid.place_agent(commuter, (x, y))
            self.schedule.add(commuter)
        
        # Create providers
        provider_configs = [
            ("UberLike", "car", 4, 15),
            ("BikeShare", "bike", 1, 5),
            ("BusCompany", "bus", 30, 3),
            ("TaxiCo", "car", 4, 12),
            ("ScooterShare", "bike", 1, 4)
        ]
        
        for i in range(num_providers):
            config = provider_configs[i % len(provider_configs)]
            
            # Random position for service center
            x = random.randrange(width)
            y = random.randrange(height)
            
            # Create provider
            provider = DecentralizedProvider(
                unique_id=100 + i,
                model=self,
                pos=(x, y),
                company_name=f"{config[0]}-{i}",
                mode_type=config[1],
                capacity=config[2],
                base_price=config[3],
                blockchain_interface=self.marketplace
            )
            
            # Add to grid and schedule
            self.grid.place_agent(provider, (x, y))
            self.schedule.add(provider)
        
        print(f"Model initialized with {num_commuters} commuters and {num_providers} providers")
        print(f"Marketplace API connected: {self.marketplace.w3.is_connected()}")
    
    def step(self):
        """Execute one step of the model"""
        # Run all agents
        self.schedule.step()
        
        # Process any pending marketplace matching every 5 steps
        if self.schedule.time % 5 == 0:
            self.process_marketplace_matching()
        
        # Update metrics
        self.update_metrics()
    
    def process_marketplace_matching(self):
        """Process all pending requests in marketplace"""
        # Get active requests from marketplace
        active_requests = self.marketplace.get_marketplace_requests(status='active')
        
        for request in active_requests:
            request_id = request['request_id']
            
            # Check if request has offers
            offers = self.marketplace.get_request_offers(request_id)
            
            if len(offers) >= 2:  # Wait for at least 2 offers
                # Run matching
                success, match = self.marketplace.run_marketplace_matching(request_id)
                
                if success:
                    self.total_matches += 1
                    print(f"Step {self.schedule.time}: Matched request {request_id}")
    
    def update_metrics(self):
        """Update model metrics"""
        # Count total requests
        total_requests = len(self.marketplace.marketplace_db['requests'])
        if total_requests > self.total_requests:
            self.total_requests = total_requests
        
        # Count completed trips
        for agent in self.schedule.agents:
            if isinstance(agent, DecentralizedCommuter):
                self.total_completed += agent.completed_trips
                agent.completed_trips = 0  # Reset counter

def run_simulation(steps=100, num_commuters=20, num_providers=10):
    """
    Run the simplified MaaS simulation
    """
    print("=" * 60)
    print("SIMPLIFIED MaaS MARKETPLACE SIMULATION")
    print("=" * 60)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create model
    model = SimplifiedMaaSModel(
        num_commuters=num_commuters,
        num_providers=num_providers
    )
    
    # Run simulation
    start_time = time.time()
    
    for step in range(steps):
        model.step()
        
        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"Step {step}/{steps} - Requests: {model.total_requests}, "
                  f"Matches: {model.total_matches}, Completed: {model.total_completed}")
    
    end_time = time.time()
    
    # Print results
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"Total simulation time: {end_time - start_time:.2f} seconds")
    print(f"Total requests created: {model.total_requests}")
    print(f"Total matches made: {model.total_matches}")
    print(f"Total trips completed: {model.total_completed}")
    
    # Print marketplace statistics
    marketplace = model.marketplace
    print(f"\nMarketplace Statistics:")
    print(f"- Registered commuters: {len(marketplace.marketplace_db['commuters'])}")
    print(f"- Registered providers: {len(marketplace.marketplace_db['providers'])}")
    print(f"- Total offers submitted: {len(marketplace.marketplace_db['offers'])}")
    print(f"- Transactions queued: {marketplace.tx_count}")
    
    return model

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Run simplified MaaS simulation')
    parser.add_argument('--steps', type=int, default=100, help='Number of simulation steps')
    parser.add_argument('--commuters', type=int, default=20, help='Number of commuters')
    parser.add_argument('--providers', type=int, default=10, help='Number of providers')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with fewer agents')
    
    args = parser.parse_args()
    
    # Debug mode uses fewer agents
    if args.debug:
        print("Running in DEBUG mode...")
        run_simulation(steps=20, num_commuters=5, num_providers=3)
    else:
        run_simulation(
            steps=args.steps,
            num_commuters=args.commuters,
            num_providers=args.providers
        )