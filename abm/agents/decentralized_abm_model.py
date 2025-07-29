from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from decentralized_commuter import DecentralizedCommuter
from decentralized_provider import DecentralizedProvider
from nft_marketplace import NFTMarketplace
from amm_time_sensitive_pricing import MobilityAMM, TimeSensitivePricing
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.blockchain_interface import BlockchainInterface

import numpy as np
import random
import logging
import time
import math

class DecentralizedMaaSModel(Model):
    """
    Agent-based model for decentralized Mobility-as-a-Service with blockchain integration.
    Includes NFT marketplace and AMM for time-sensitive pricing of mobility services.
    """
    def __init__(self, 
                 grid_width=25, 
                 grid_height=25, 
                 num_commuters=20,
                 income_weights=[0.3, 0.5, 0.2],  # Low, middle, high
                 health_weights=[0.8, 0.2],  # Good, poor
                 payment_weights=[0.7, 0.3],  # PAYG, subscription
                 age_distribution={(18, 30): 0.25, (31, 45): 0.3, (46, 65): 0.3, (66, 85): 0.15},
                 disability_weights=[0.1, 0.9],  # True, False
                 tech_access_weights=[0.85, 0.15],  # True, False
                 uber_like1_capacity=50,
                 uber_like1_price=1.2,
                 uber_like2_capacity=30,
                 uber_like2_price=1.0,
                 bike_share1_capacity=10,
                 bike_share1_price=0.5,
                 bike_share2_capacity=8,
                 bike_share2_price=0.4,
                 stations=None,
                 routes=None,
                 transfers=None,
                 blockchain_config="../../blockchain_config.json",
                 market_type="hybrid",
                 enable_amm=True,
                 amm_volume_threshold=2,
                 time_decay_factor=0.1,
                 min_price_ratio=0.5):
        """
        Initialize the decentralized MaaS model with blockchain integration.
        
        Args:
            grid_width, grid_height: Dimensions of the simulation grid
            num_commuters: Number of commuter agents
            income_weights: Distribution of income levels (low, middle, high)
            health_weights: Distribution of health status (good, poor)
            payment_weights: Distribution of payment schemes (PAYG, subscription)
            age_distribution: Distribution of age ranges
            disability_weights: Distribution of disability status
            tech_access_weights: Distribution of technology access
            uber_like1/2_capacity/price: Capacity and pricing for car services
            bike_share1/2_capacity/price: Capacity and pricing for bike services
            stations: Public transport stations (optional)
            routes: Public transport routes (optional)
            transfers: Public transport transfers (optional)
            blockchain_config: Configuration file for blockchain interface
            market_type: Market mechanism type (order_book, amm, hybrid)
            enable_amm: Whether to enable the AMM component
            amm_volume_threshold: Minimum transactions for AMM eligibility
            time_decay_factor: Rate of price decay as service time approaches
            min_price_ratio: Minimum price as ratio of initial price
        """
        super().__init__()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DecentralizedMaaSModel")
        
        # Basic model setup
        self.grid = MultiGrid(grid_width, grid_height, torus=False)
        self.schedule = RandomActivation(self)
        self.running = True
        self.current_step = 0
        
        # Store model parameters
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_commuters = num_commuters
        self.income_weights = income_weights
        self.health_weights = health_weights
        self.payment_weights = payment_weights
        self.age_distribution = age_distribution
        self.disability_weights = disability_weights
        self.tech_access_weights = tech_access_weights
        self.stations = stations if stations else []
        self.routes = routes if routes else []
        self.transfers = transfers if transfers else []
        
        # Add these to the model's __init__ method after other initialization
        self.transaction_count = 0
        self.active_listings_count = 0
        self.completed_trips_count = 0
        # Phase 2 specific parameters
        self.enable_amm = enable_amm
        self.amm_volume_threshold = amm_volume_threshold
        self.time_decay_factor = time_decay_factor
        self.min_price_ratio = min_price_ratio
        
        # Performance tracking
        self.execution_times = {
            'commuter_decisions': [],
            'provider_offers': [],
            'marketplace_update': [],
            'blockchain_operations': [],
            'amm_operations': []
        }
        
        # Initialize blockchain interface
        self.blockchain_interface = BlockchainInterface(blockchain_config, async_mode=True)
        self.blockchain_interface.model = self 
        # Initialize time-sensitive pricing component
        self.pricing_engine = TimeSensitivePricing(base_pricing_model="exponential_decay")
        self.pricing_engine.pricing_params["exponential_decay"]["decay_rate"] = time_decay_factor
        self.pricing_engine.pricing_params["exponential_decay"]["min_price_ratio"] = min_price_ratio
        
        # Initialize NFT marketplace with specified type (order_book, amm, or hybrid)
        self.marketplace = NFTMarketplace(self, self.blockchain_interface, market_type)
        
        # Initialize AMM component (Phase 2 addition)
        if self.enable_amm:
            self.amm = MobilityAMM(self.blockchain_interface, self)
            # Set AMM parameters
            self.amm.amm_params = {
                'time_decay_factor': time_decay_factor,
                'demand_sensitivity': 0.2,
                'min_price_ratio': min_price_ratio,
                'volume_threshold': amm_volume_threshold,
                'price_impact_factor': 0.05
            }
            self.logger.info("AMM component initialized with volume threshold: %d", amm_volume_threshold)
        else:
            self.amm = None
            self.logger.info("AMM component disabled")
        
        # Initialize provider agents
        self.providers = {}
        self._create_provider_agents(
            uber_like1_capacity, uber_like1_price, 
            uber_like2_capacity, uber_like2_price,
            bike_share1_capacity, bike_share1_price, 
            bike_share2_capacity, bike_share2_price
        )
        
        # Initialize commuter agents
        self.commuters = {}
        self._create_commuter_agents()
        
        # Set up data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Active Commuters": lambda m: self.count_active_commuters(),
                "Active Providers": lambda m: self.count_active_providers(),
                "Total Transactions": lambda m: len(self.marketplace.transaction_history),
                "Active NFT Listings": lambda m: sum(1 for listing in self.marketplace.listings.values() if listing['status'] == 'active'),
                "Average NFT Price": lambda m: self.calculate_average_nft_price(),
                "Completed Trips": lambda m: self.count_completed_trips(),
                "AMM Pools": lambda m: len(self.amm.liquidity_pools) if self.amm else 0,
                "AMM Trading Volume": lambda m: self.calculate_amm_volume(),
                "Execution Time": lambda m: self.get_average_execution_time()
            },
            agent_reporters={
                "Location": lambda a: a.location if hasattr(a, 'location') else None,
                "Current Mode": lambda a: a.current_mode if hasattr(a, 'current_mode') else None
            }
        )
        
        # Batch processing configuration
        self.batch_size = 10  # Process up to 10 requests/transactions in batch
        self.process_blockchain_interval = 5  # Process blockchain every 5 ticks
        
        # Pending operations tracking
        self.pending_requests = []
        self.pending_offers = []
        self.pending_transactions = []
        
        # Popular routes tracking (for AMM)
        self.route_transactions = {}  # Route key -> transaction count
        self.popular_routes = set()   # Set of popular route keys
        
        self.logger.info("DecentralizedMaaSModel initialized with %d commuters and %d providers",
                       len(self.commuters), len(self.providers))

    def _create_provider_agents(self, uber_like1_capacity, uber_like1_price, 
                           uber_like2_capacity, uber_like2_price,
                           bike_share1_capacity, bike_share1_price, 
                           bike_share2_capacity, bike_share2_price):
        """
        Create provider agents with parameters, including both individual providers
        and fractionally-owned fleet providers (like community public transport).
        
        This method creates multiple types of mobility providers, each with their own
        characteristics, service areas, and ownership structures.
        """
        # Car service providers (individual drivers or small companies)
        self._add_provider_agent("UberLike1", "car", uber_like1_price, uber_like1_capacity, 
                            service_area=25, provider_type="individual")
        self._add_provider_agent("UberLike2", "car", uber_like2_price, uber_like2_capacity, 
                            service_area=20, provider_type="individual")
        
        # Bike service providers (typically individual owners or small bike fleets)
        self._add_provider_agent("BikeShare1", "bike", bike_share1_price, bike_share1_capacity, 
                            service_area=15, provider_type="individual")
        self._add_provider_agent("BikeShare2", "bike", bike_share2_price, bike_share2_capacity, 
                            service_area=15, provider_type="individual")
        
        # Add public transport providers if parameters exist
        # Community-owned bus cooperative
        if hasattr(self, 'bus_capacity') and hasattr(self, 'bus_price'):
            # Create token holder structure representing community ownership
            bus_token_holders = {
                # Represents community members with stakes in the bus service
                # Each key is a stakeholder ID and value is their ownership stake
                1001: 20,  # Community member with 20% stake
                1002: 15,  # Another member with 15% stake
                1003: 10,  # Local business with 10% stake
                1004: 8,   # Neighborhood association with 8% stake
                1005: 7,   # Another stakeholder
                # More distributed stakeholders with smaller shares
                **{1006+i: 2 for i in range(20)}  # 20 small stakeholders with 2% each
            }
            
            self._add_provider_agent("CommunityBus", "bus", self.bus_price, self.bus_capacity,
                                service_area=40, provider_type="fractional_fleet",
                                fleet_token_holders=bus_token_holders)
                                
        # City train consortium operated as a DAO
        if hasattr(self, 'train_capacity') and hasattr(self, 'train_price'):
            # Create token holder structure for train DAO
            train_token_holders = {
                # Represents stakeholders in the train service DAO
                2001: 25,  # Major community fund with 25% stake
                2002: 20,  # City development group with 20% stake
                2003: 15,  # Transit advocacy organization with 15%
                # More distributed stakeholders
                **{2004+i: 2 for i in range(20)}  # 20 token holders with 2% each
            }
            
            self._add_provider_agent("TrainDAO", "train", self.train_price, self.train_capacity,
                                service_area=60, provider_type="fractional_fleet",
                                fleet_token_holders=train_token_holders)
        
        # Check if additional provider types are defined in parameters
        if hasattr(self, 'additional_providers'):
            for provider_info in self.additional_providers:
                self._add_provider_agent(
                    provider_info['name'],
                    provider_info['mode_type'],
                    provider_info['base_price'],
                    provider_info['capacity'],
                    service_area=provider_info.get('service_area', 40),
                    provider_type=provider_info.get('provider_type', 'individual'),
                    fleet_token_holders=provider_info.get('token_holders', None)
                )

    def _add_provider_agent(self, company_name, mode_type, base_price, capacity, 
                       service_area=40, provider_type="individual", 
                       fleet_token_holders=None):
        """
        Add a provider agent to the model with full support for decentralized ownership.
        
        Args:
            company_name: Name of the provider/company
            mode_type: Transport mode ('car', 'bike', 'bus', 'train', etc.)
            base_price: Base price per unit distance
            capacity: Maximum service capacity
            service_area: Service coverage radius
            provider_type: Either "individual" or "fractional_fleet"
            fleet_token_holders: For fractional fleets, a dict of token holder IDs and stakes
        
        Returns:
            The created provider agent
        """
        provider_id = len(self.providers) + 1
        
        # Generate a service center location appropriate to the provider type
        if provider_type == "fractional_fleet" and mode_type in ["bus", "train"]:
            # Public transport typically has more central service centers
            center_x = self.grid_width // 2
            center_y = self.grid_height // 2
            # Place near center with some randomness
            service_center = [
                int(center_x + random.uniform(-0.2, 0.2) * self.grid_width),
                int(center_y + random.uniform(-0.2, 0.2) * self.grid_height)
            ]
        else:
            # Individual providers are more dispersed
            service_center = [
                random.randint(service_area, self.grid_width - service_area),
                random.randint(service_area, self.grid_height - service_area)
            ]
        
        # Create provider agent with appropriate parameters
        provider = DecentralizedProvider(
            provider_id, 
            self,
            company_name,
            mode_type,
            base_price,
            capacity,
            service_area,
            self.blockchain_interface,
            quality_score=random.randint(65, 95),
            reliability=random.randint(70, 95),
            response_time=random.randint(5, 15),
            provider_type=provider_type,  # Pass provider type
            fleet_token_holders=fleet_token_holders  # Pass token holders for fractional ownership
        )
        
        # Set service center
        provider.service_center = service_center
        
        # For fractional fleets like public transport, create predefined routes
        if provider_type == "fractional_fleet" and mode_type in ["bus", "train"]:
            # Create route corridor for this public transport service
            if not hasattr(provider, 'location_preferences'):
                provider.location_preferences = {}
                
            # Generate endpoints for the route corridor
            if mode_type == "train":
                # Trains typically connect distant points
                route_length = min(self.grid_width, self.grid_height) * 0.7
                angle = random.uniform(0, 2 * math.pi)
                
                # Generate endpoints from center outward
                center = [self.grid_width // 2, self.grid_height // 2]
                start = [
                    int(center[0] + route_length * 0.5 * math.cos(angle)),
                    int(center[1] + route_length * 0.5 * math.sin(angle))
                ]
                end = [
                    int(center[0] - route_length * 0.5 * math.cos(angle)),
                    int(center[1] - route_length * 0.5 * math.sin(angle))
                ]
                
                # Keep within grid bounds
                start[0] = max(0, min(start[0], self.grid_width - 1))
                start[1] = max(0, min(start[1], self.grid_height - 1))
                end[0] = max(0, min(end[0], self.grid_width - 1))
                end[1] = max(0, min(end[1], self.grid_height - 1))
                
                # Set route corridor with narrow width (trains have fixed routes)
                provider.location_preferences['route_corridor'] = {
                    'start': start,
                    'end': end,
                    'width': service_area * 0.3  # Narrow corridor for trains
                }
            elif mode_type == "bus":
                # Buses typically serve more areas with wider corridors
                # Create multiple corridors for bus routes
                corridors = []
                
                # Create 2-3 corridors for this bus service
                for _ in range(random.randint(2, 3)):
                    angle = random.uniform(0, 2 * math.pi)
                    route_length = min(self.grid_width, self.grid_height) * 0.5
                    
                    center = [self.grid_width // 2, self.grid_height // 2]
                    start = [
                        int(center[0] + route_length * 0.5 * math.cos(angle)),
                        int(center[1] + route_length * 0.5 * math.sin(angle))
                    ]
                    end = [
                        int(center[0] - route_length * 0.5 * math.cos(angle)),
                        int(center[1] - route_length * 0.5 * math.sin(angle))
                    ]
                    
                    # Keep within grid bounds
                    start[0] = max(0, min(start[0], self.grid_width - 1))
                    start[1] = max(0, min(start[1], self.grid_height - 1))
                    end[0] = max(0, min(end[0], self.grid_width - 1))
                    end[1] = max(0, min(end[1], self.grid_height - 1))
                    
                    corridors.append({
                        'start': start,
                        'end': end,
                        'width': service_area * 0.6  # Wider corridor for buses
                    })
                
                # Set primary corridor
                provider.location_preferences['route_corridor'] = corridors[0]
                # Store additional corridors
                provider.location_preferences['additional_corridors'] = corridors[1:]
        
        # Add provider to scheduler
        self.schedule.add(provider)
        
        # Store provider in our records
        self.providers[provider_id] = provider
        
        # Generate a random location in the grid for visualization
        x = random.randrange(self.grid_width)
        y = random.randrange(self.grid_height)
        self.grid.place_agent(provider, (x, y))
        # CRITICAL: Register provider on blockchain
        success, address = self.blockchain_interface.register_provider(provider)
        if not success:
            self.logger.error(f"Failed to register provider {provider_id}")
        else:
            self.logger.info(f"Successfully registered provider {provider_id} at {address}")
        
        # Log different messages based on provider type
        if provider_type == "individual":
            self.logger.info(f"Added individual provider {company_name} with mode {mode_type}, "
                        f"base price {base_price}, capacity {capacity}")
        else:
            stakeholder_count = len(fleet_token_holders) if fleet_token_holders else 0
            self.logger.info(f"Added fractional fleet {company_name} with mode {mode_type}, "
                        f"base price {base_price}, capacity {capacity}, "
                        f"stakeholders: {stakeholder_count}")
        
        return provider

    def _create_commuter_agents(self):
        """Create commuter agents based on population parameters"""
        for i in range(self.num_commuters):
            commuter_id = i + 1
            
            # Generate random location
            x = random.randrange(self.grid_width)
            y = random.randrange(self.grid_height)
            
            # Determine income level based on weights
            income_levels = ['low', 'middle', 'high']
            income_level = random.choices(income_levels, self.income_weights)[0]
            
            # Determine health status based on weights
            health_statuses = ['good', 'poor']
            health_status = random.choices(health_statuses, self.health_weights)[0]
            
            # Determine payment scheme based on weights
            payment_schemes = ['PAYG', 'subscription']
            payment_scheme = random.choices(payment_schemes, self.payment_weights)[0]
            
            # Generate age based on distribution
            age = self._generate_age_from_distribution()
            
            # Determine disability status based on weights
            has_disability = random.choices([True, False], self.disability_weights)[0]
            
            # Determine tech access based on weights
            tech_access = random.choices([True, False], self.tech_access_weights)[0]
            
            # Create commuter agent
            commuter = DecentralizedCommuter(
                commuter_id,
                self,
                (x, y),
                age,
                income_level,
                has_disability,
                tech_access,
                health_status,
                payment_scheme,
                self.blockchain_interface
            )
            
            # Add commuter to scheduler
            self.schedule.add(commuter)
            
            # Store commuter in our records
            self.commuters[commuter_id] = commuter
            
            # Place commuter on grid
            self.grid.place_agent(commuter, (x, y))

            # CRITICAL: Register commuter on blockchain
            success, address = self.blockchain_interface.register_commuter(commuter)
            if not success:
                self.logger.error(f"Failed to register commuter {commuter_id}")
            else:
                self.logger.info(f"Successfully registered commuter {commuter_id} at {address}")

    def _generate_age_from_distribution(self):
        """Generate age based on age distribution"""
        age_ranges = list(self.age_distribution.keys())
        weights = list(self.age_distribution.values())
        
        selected_range = random.choices(age_ranges, weights=weights)[0]
        
        return random.randint(selected_range[0], selected_range[1])

    def check_is_peak(self, time_value=None):
        """
        Check if current time is peak hour
        
        Args:
            time_value: Optional time value to check (defaults to current model time)
            
        Returns:
            Boolean indicating if time is peak hour
        """
        if time_value is None:
            time_value = self.current_step
            
        time_of_day = time_value % 144  # 144 ticks per day
        
        # Morning peak (6:30am-10am) or evening peak (3pm-7pm)
        if (36 <= time_of_day < 60) or (90 <= time_of_day < 114):
            return True
        return False

    def get_demand_factor(self, time_value, mode_type):
        """
        Get demand multiplier for a given time and mode
        
        Args:
            time_value: Time value to check
            mode_type: Transport mode type
            
        Returns:
            Demand factor (multiplier)
        """
        time_of_day = time_value % 144  # 144 ticks per day
        
        # Base demand factors by time of day
        if 36 <= time_of_day < 60:  # Morning peak
            base_factor = 1.4
        elif 90 <= time_of_day < 114:  # Evening peak
            base_factor = 1.3
        elif 60 <= time_of_day < 90:  # Daytime
            base_factor = 1.0
        elif 114 <= time_of_day < 126:  # Evening
            base_factor = 0.9
        else:  # Night
            base_factor = 0.7
        
        # Mode-specific adjustments
        if mode_type == "car":
            if self.check_is_peak(time_value):
                return base_factor * 1.2  # Cars more in demand during peaks
            return base_factor
        elif mode_type == "bike":
            if 60 <= time_of_day < 114:  # Daytime and evening peak (good weather assumed)
                return base_factor * 1.1
            return base_factor * 0.8  # Less demand at night/early morning
        else:
            return base_factor

    def step(self):
        """Main model step function with batched operations"""
        if self.current_step == 0:
            # On first step, ensure registrations are processed
            self._ensure_commuter_registrations()

        # Increment current step
        self.current_step += 1
        
        # Process blockchain operations every step for more responsive state updates
        start_time = time.time()
        self.blockchain_interface.update_cache()
        
        # IMPORTANT FIX: Process transaction queue more aggressively
        if self.blockchain_interface.tx_queue and len(self.blockchain_interface.tx_queue) > 0:
            self.blockchain_interface._process_transaction_batch()  # Process transactions every step
        
        self._check_blockchain_results()
        self.execution_times['blockchain_operations'].append(time.time() - start_time)
        
            
        # Process commuter decisions in batches
        start_time = time.time()
        self._process_commuter_decisions()
        self.execution_times['commuter_decisions'].append(time.time() - start_time)
        
        # Process provider offers in batches
        start_time = time.time()
        self._process_provider_offers()
        self.execution_times['provider_offers'].append(time.time() - start_time)
        
        # Update NFT marketplace
        start_time = time.time()
        self.marketplace.update_listings()
        self.execution_times['marketplace_update'].append(time.time() - start_time)
        
        # Update AMM component (Phase 2 addition)
        if self.enable_amm:
            start_time = time.time()
            self._update_amm_component()
            self.execution_times['amm_operations'].append(time.time() - start_time)
        
        # Update popular routes
        if self.current_step % 20 == 0:  # Every 20 steps
            self._update_popular_routes()
        
        # Update all agents
        self.schedule.step()
        
        # Collect data
        self.transaction_count = max(self.transaction_count, len(self.marketplace.transaction_history))
        self.completed_trips_count = len([tx for tx in self.marketplace.transaction_history 
                                        if tx.get('status') == 'completed'])
        self.datacollector.collect(self)
        
        # Update model statistics from blockchain interface
        if hasattr(self.blockchain_interface, 'stats'):
            tx_confirmed = self.blockchain_interface.stats.get('transactions_confirmed', 0)
            if tx_confirmed > 0 and self.transaction_count == 0:
                self.transaction_count = tx_confirmed
                self.logger.info(f"Updated transaction count from blockchain: {tx_confirmed}")
        # In step() method:
        if len(self.blockchain_interface.tx_queue) > 0 and self.current_step % 10 == 0:
            # Process transaction queue every 10 steps
            self.blockchain_interface._process_transaction_batch()

        # Log progress periodically
        if self.current_step % 50 == 0:
            self.logger.info(f"Step {self.current_step}: {len(self.commuters)} commuters, "
                            f"{self.transaction_count} transactions, "
                            f"{len(self.popular_routes)} popular routes")

        # Process blockchain operations more aggressively
        for _ in range(3):  # Try multiple times per step
            self.blockchain_interface._check_pending_transactions()
            self.blockchain_interface.update_cache()

    def _check_blockchain_results(self):
        """Check and process blockchain transaction results"""
        # Check pending commuter registrations
        if hasattr(self.blockchain_interface, 'pending_registrations'):
            for commuter_id, reg_data in list(self.blockchain_interface.pending_registrations.items()):
                if reg_data['status'] == 'confirmed' and commuter_id in self.commuters:
                    commuter = self.commuters[commuter_id]
                    # Update commuter state to reflect successful registration
                    if hasattr(commuter, 'blockchain_status'):
                        commuter.blockchain_status = 'registered'
        
        # Check pending transactions
        for tx_hash, tx_data in list(self.blockchain_interface.pending_transactions.items()):
            if hasattr(tx_data, 'status') and tx_data.status == 'confirmed':
                # Process confirmed transaction based on type
                if tx_data.tx_type == 'request':
                    request_id = tx_data.params.get('requestId')
                    if request_id and hasattr(self.blockchain_interface, 'state_cache') and 'requests' in self.blockchain_interface.state_cache:
                        if request_id in self.blockchain_interface.state_cache['requests']:
                            # Update request status
                            self.blockchain_interface.state_cache['requests'][request_id]['status'] = 'active'
                            # Add to transaction count
                            self.transaction_count += 1
                elif tx_data.tx_type == 'marketplace' and tx_data.function_name == 'purchaseNFT':
                    # Update transaction count
                    self.transaction_count += 1

    def _process_commuter_decisions(self):
        """Process commuter decisions in batches"""
        # Create new travel requests
        self._generate_commuter_requests()
        
        # Process pending requests in batches
        if self.pending_requests:
            # Take a batch of requests
            batch = self.pending_requests[:self.batch_size]
            self.pending_requests = self.pending_requests[self.batch_size:]
            
            # Create requests as a batch to optimize blockchain transactions
            request_batch = []
            # Process each request in the batch
            for request in batch:
                commuter_id = request['commuter_id']
                commuter = self.commuters.get(commuter_id)
                
                if not commuter:
                    continue
                # Store for batch processing
                request_batch.append(request)
                # Store in commuter's local state
                if not hasattr(commuter, 'requests'):
                    commuter.requests = {}
                commuter.requests[request['request_id']] = request
                # Check marketplace for existing NFTs matching needs
                market_options = self.marketplace.search_nfts({
                    'origin_area': [request['origin'], 5],
                    'destination_area': [request['destination'], 5],
                    'time_window': [request['start_time'] - 1800, request['start_time'] + 1800]
                })
                
                # For popular routes, check if AMM is available
                route_key = self._get_route_key(request['origin'], request['destination'], request['start_time'])
                use_amm = self.enable_amm and route_key in self.popular_routes
                
                # Decide whether to use AMM, purchase from market, or request new service
                if use_amm and random.random() < 0.4:  # 40% chance to use AMM for popular routes
                    # Create AMM pool if it doesn't exist
                    if route_key not in self.amm.liquidity_pools:
                        self._create_amm_pool_for_route(route_key, request)
                    
                    # Get quote from AMM
                    amm_quote = self.amm.get_quote(
                        request['origin'], 
                        request['destination'], 
                        commuter._calculate_max_price() if hasattr(commuter, '_calculate_max_price') else 20, 
                        is_buy=True
                    )
                    
                    if amm_quote:
                        # Execute swap
                        success, swap_details = self.amm.execute_swap(amm_quote, commuter_id)
                        if success:
                            self.logger.info(f"Commuter {commuter_id} purchased service via AMM for route {route_key}")
                            # Update transaction count for successful AMM swap
                            self.transaction_count += 1
                elif market_options and hasattr(commuter, 'evaluate_marketplace_options'):
                    # Choose NFT from market based on commuter's strategy weights
                    strategy_weights = getattr(commuter, 'strategy_weights', {'market_purchase': 0.3})
                    if strategy_weights.get('market_purchase', 0) > random.random():
                        # Get request ID for evaluation
                        request_id = request.get('request_id', 0)
                        
                        # Create request object if needed for evaluation
                        if request_id not in commuter.requests:
                            commuter.requests[request_id] = request
                        
                        # Evaluate options using utility function
                        ranked_options = commuter.evaluate_marketplace_options(request_id)
                        
                        if ranked_options:
                            # Select highest utility option
                            selected_nft = ranked_options[0][1]
                            success = self.blockchain_interface.purchase_nft(selected_nft['nft_id'], commuter_id)
                            if success:
                                self.logger.info(f"Commuter {commuter_id} purchased NFT {selected_nft['nft_id']} from marketplace")
                                # Update transaction count
                                self.transaction_count += 1
                else:
                    # Standard service request
                    success = self.blockchain_interface.create_travel_request(commuter, request)
                    self.logger.debug(f"Commuter {commuter_id} created travel request, success: {success}")
            # Process batch of requests
            if request_batch:
                self.blockchain_interface.process_requests_batch(request_batch)
    
    
    def _generate_commuter_requests(self):
        """Generate new travel requests based on time-of-day patterns"""
        # Potential trip generation for each commuter
        for commuter_id, commuter in self.commuters.items():
            # Only generate trips for commuters without active requests
            if not hasattr(commuter, 'has_active_request') or not commuter.has_active_request():
                # Check if commuter should create a trip based on time patterns
                if self._should_create_trip(commuter):
                    # Generate trip details
                    origin = commuter.location
                    destination = self._generate_destination(commuter)
                    
                    # Add some randomness to start time
                    # Current time plus 1-12 steps (representing 5 min to 1 hour)
                    start_time = self.current_step + random.randint(1, 12)
                    
                    # Determine travel purpose
                    travel_purpose = self._determine_travel_purpose(self.current_step)
                    
                    # Create request
                    request_id = self._generate_request_id()
                    request = {
                        'request_id': request_id,
                        'commuter_id': commuter_id,
                        'origin': origin,
                        'destination': destination,
                        'start_time': start_time,
                        'travel_purpose': travel_purpose,
                        'flexible_time': commuter.determine_schedule_flexibility(travel_purpose),
                        'status': 'active',
                        'created_at': self.current_step
                    }
                    
                    # Add to pending requests
                    self.pending_requests.append(request)

    def _generate_request_id(self):
        """Generate a unique request ID"""
        # Use timestamp and random number for uniqueness
        return int(time.time() * 1000 + random.randint(0, 999))

    def _determine_travel_purpose(self, current_time):
        """
        Determine likely travel purpose based on time of day
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Travel purpose code (0: work, 1: school, 2: shopping, etc.)
        """
        time_of_day = current_time % 144  # 144 ticks per day
        
        # Morning peak (likely work/school)
        if 36 <= time_of_day < 60:
            return random.choices([0, 1], weights=[0.8, 0.2])[0]  # 80% work, 20% school
        # Mid-day (likely shopping/errands/medical)
        elif 60 <= time_of_day < 90:
            return random.choices([2, 3, 6], weights=[0.5, 0.3, 0.2])[0]  # Shopping, medical, leisure
        # Evening peak (likely return from work/leisure)
        elif 90 <= time_of_day < 114:
            return random.choices([0, 6], weights=[0.6, 0.4])[0]  # Work, leisure
        # Evening (likely leisure)
        elif 114 <= time_of_day < 126:
            return 6  # Leisure
        # Night/early morning (likely other)
        else:
            return 7  # Other

    def _should_create_trip(self, commuter):
        """Determine if commuter should create a trip"""
        # Time-based probabilities
        time_of_day = self.current_step % 144  # 144 ticks per day
        
        # Morning peak (6:30am-10am)
        if 36 <= time_of_day < 60:
            base_probability = 0.05
        # Evening peak (3pm-7pm)
        elif 90 <= time_of_day < 114:
            base_probability = 0.04
        # Daytime off-peak
        elif 60 <= time_of_day < 90:
            base_probability = 0.02
        # Evening off-peak
        elif 114 <= time_of_day < 126:
            base_probability = 0.015
        # Night time
        else:
            base_probability = 0.005
        
        # Adjust based on commuter attributes
        if commuter.income_level == 'high':
            base_probability *= 1.2
        elif commuter.income_level == 'low':
            base_probability *= 0.8
            
        if commuter.has_disability or commuter.age >= 65:
            base_probability *= 0.7
            
        if commuter.payment_scheme == 'subscription':
            base_probability *= 1.3
        
        # Apply random check
        return random.random() < base_probability

    def _generate_destination(self, commuter):
        """Generate a destination for a commuter trip"""
        # Ensure minimum trip distance
        min_distance = 5
        max_attempts = 5
        
        for _ in range(max_attempts):
            # Generate random point
            x = random.randint(0, self.grid_width - 1)
            y = random.randint(0, self.grid_height - 1)
            destination = (x, y)
            
            # Calculate distance
            distance = ((destination[0] - commuter.location[0])**2 + 
                        (destination[1] - commuter.location[1])**2)**0.5
            
            if distance >= min_distance:
                return destination
        
        # If all attempts fail, return a point at minimum distance
        angle = random.uniform(0, 2 * np.pi)
        x = int(commuter.location[0] + min_distance * np.cos(angle))
        y = int(commuter.location[1] + min_distance * np.sin(angle))
        
        # Ensure within grid bounds
        x = max(0, min(x, self.grid_width - 1))
        y = max(0, min(y, self.grid_height - 1))
        
        return (x, y)

    def _process_provider_offers(self):
        """Process provider offers in batches"""
        # Use a combination of pending and cached active requests
        active_requests = []
        
        # Add pending requests that are marked active
        active_requests.extend([
            request for request in self.pending_requests 
            if request.get('status', 'active') == 'active'
        ])
        
        # Also check blockchain interface cache for active requests
        if hasattr(self.blockchain_interface, 'state_cache') and 'requests' in self.blockchain_interface.state_cache:
            self.logger.info(f"Total requests in state cache: {len(self.blockchain_interface.state_cache.get('requests', {}))}")
            self.logger.info(f"Requests with 'active' status: {sum(1 for r in self.blockchain_interface.state_cache.get('requests', {}).values() if r.get('status') == 'active')}")

            for req_id, req_data in self.blockchain_interface.state_cache['requests'].items():
                if req_data.get('status') == 'active':
                    # Convert cached request to proper format
                    
                    try:
                        request_data = req_data.get('data', {})
                        if isinstance(request_data, dict) and 'origin' in request_data and 'destination' in request_data:
                            request = {
                                'request_id': req_id,
                                'origin': request_data.get('origin', [0, 0]),
                                'destination': request_data.get('destination', [0, 0]),
                                'start_time': request_data.get('startTime', 0),
                                'status': 'active'
                            }
                            active_requests.append(request)
                    except Exception as e:
                        self.logger.error(f"Error processing cached request: {e}")
        
        self.logger.info(f"Processing provider offers for {len(active_requests)} active requests")
        # For each provider, generate offers
        for provider_id, provider in self.providers.items():
            # Each provider considers a subset of requests based on capacity
            provider_capacity = provider.available_capacity
            requests_to_consider = random.sample(
                active_requests, 
                min(provider_capacity, len(active_requests))
            ) if active_requests else []
            
            # Filter requests by service area
            requests_to_consider = [
                req for req in requests_to_consider
                if provider._can_service_route(req['origin'], req['destination'])
            ]
            
            # Generate offers
            for request in requests_to_consider:
                offer = provider.generate_offer(request)
                if offer:
                    self.pending_offers.append(offer)
                    self.logger.info(f"Provider {provider_id} generated offer for request {request['request_id']}")
        
        # Process offers in batches
        if self.pending_offers:
            # Take a batch of offers
            batch = self.pending_offers[:self.batch_size]
            self.pending_offers = self.pending_offers[self.batch_size:]
            
            # Process each offer in the batch
            for offer in batch:
                if 'provider_id' not in offer:
                    continue
                    
                provider_id = offer['provider_id']
                provider = self.providers.get(provider_id)
                
                if not provider:
                    continue
                
                # Submit offer via blockchain
                success = provider.blockchain_interface.submit_offer(
                    provider, 
                    offer['request_id'], 
                    offer['price'], 
                    {
                        'route': offer['route'],
                        'time': offer['estimated_time'],
                        'start_time': offer['start_time'],
                        'mode': provider.mode_type
                    }
                )
                
                if success:
                    self.logger.info(f"Provider {provider_id} submitted offer for request {offer['request_id']} at price {offer['price']}")
                else:
                    self.logger.warning(f"Failed to submit offer for request {offer['request_id']}")

    def _update_amm_component(self):
        """Update AMM pools and pricing"""
        if not self.enable_amm:
            return
        
        # Update all AMM pools
        self.amm.update_pools()
        
        # Check for providers to add liquidity
        self._check_provider_liquidity_opportunities()
        
        # Check for price arbitrage opportunities
        self._check_arbitrage_opportunities()

    def _check_provider_liquidity_opportunities(self):
        """Check for providers who could benefit from adding liquidity to AMM pools"""
        # This simulates providers deciding to participate in AMM liquidity provision
        # Only check occasionally to reduce computational overhead
        if self.current_step % 15 != 0:
            return
            
        # For each provider, check if they should add liquidity to any pool
        for provider_id, provider in self.providers.items():
            # Only consider providers with sufficient capacity
            if provider.available_capacity < provider.capacity * 0.3:
                continue
                
            # Check which routes the provider can service
            for route_key in self.popular_routes:
                # Extract route information
                origin, destination, _ = self._parse_route_key(route_key)
                
                # Check if provider can service this route
                if not provider._can_service_route(origin, destination):
                    continue
                    
                # Decide whether to add liquidity (30% chance if eligible)
                if random.random() < 0.3:
                    # Calculate amount based on capacity and price
                    liquidity_amount = provider.base_price * 10
                    
                    # Add liquidity to pool
                    provider.add_liquidity_to_amm(origin, destination, liquidity_amount)
                    self.logger.info(f"Provider {provider_id} added {liquidity_amount} liquidity to AMM pool for route {route_key}")

    def _check_arbitrage_opportunities(self):
        """
        Check for price differences between market listings and AMM pools
        that could be exploited for arbitrage
        """
        # Only check occasionally
        if self.current_step % 10 != 0:
            return
            
        # For each popular route with an AMM pool
        for route_key in self.popular_routes:
            if route_key not in self.amm.liquidity_pools:
                continue
                
            pool = self.amm.liquidity_pools[route_key]
            amm_price = pool['last_price']
            
            # Extract route information
            origin, destination, time_window = self._parse_route_key(route_key)
            
            # Find market listings for similar route
            market_listings = self.marketplace.search_nfts({
                'origin_area': [origin, 5],
                'destination_area': [destination, 5],
                'time_window': [time_window - 3600, time_window + 3600]
            })
            
            if not market_listings:
                continue
                
            # Calculate average market price
            market_prices = [listing['price'] for listing in market_listings]
            avg_market_price = sum(market_prices) / len(market_prices)
            
            # Check for significant price difference (>10%)
            if abs(avg_market_price - amm_price) / avg_market_price > 0.1:
                self.logger.info(f"Arbitrage opportunity on route {route_key}: "
                                f"AMM price: {amm_price}, Market price: {avg_market_price}")
                
                # In a more complex model, agents could exploit this opportunity

    def _update_popular_routes(self):
        """Update the set of popular routes based on transaction history"""
        # Count transactions by route
        self.route_transactions = {}
        
        # Analyze transaction history
        for tx in self.marketplace.transaction_history:
            nft_id = tx.get('nft_id')
            if nft_id in self.marketplace.listings:
                listing = self.marketplace.listings[nft_id]
                if 'details' in listing:
                    # Get route details
                    details = listing['details']
                    route_key = self._get_route_key(
                        details.get('origin', [0, 0]),
                        details.get('destination', [0, 0]),
                        details.get('service_time', 0)
                    )
                    
                    # Count transaction
                    self.route_transactions[route_key] = self.route_transactions.get(route_key, 0) + 1
        
        # Identify popular routes (those with transaction count >= threshold)
        self.popular_routes = {
            route for route, count in self.route_transactions.items()
            if count >= self.amm_volume_threshold
        }
        
        # Create AMM pools for popular routes that don't have one yet
        if self.enable_amm:
            for route_key in self.popular_routes:
                if route_key not in self.amm.liquidity_pools:
                    # Extract route information
                    origin, destination, time_window = self._parse_route_key(route_key)
                    
                    # Estimate initial price based on route characteristics
                    initial_price = self._estimate_route_price(origin, destination)
                    
                    # Create AMM pool
                    self._create_amm_pool_for_route(route_key, {
                        'origin': origin,
                        'destination': destination,
                        'start_time': time_window
                    }, initial_price)
        
        self.logger.info(f"Updated popular routes: {len(self.popular_routes)} routes exceed threshold {self.amm_volume_threshold}")

    def _create_amm_pool_for_route(self, route_key, request, initial_price=None):
        """
        Create an AMM liquidity pool for a route
        
        Args:
            route_key: Route key
            request: Request details containing origin, destination
            initial_price: Optional initial price (calculated if not provided)
        """
        if not self.enable_amm or route_key in self.amm.liquidity_pools:
            return
            
        # Extract or parse route information
        if isinstance(route_key, str):
            origin, destination, _ = self._parse_route_key(route_key)
        else:
            origin = request['origin']
            destination = request['destination']
            route_key = self._get_route_key(origin, destination, request['start_time'])
        
        # Calculate initial price if not provided
        if initial_price is None:
            initial_price = self._estimate_route_price(origin, destination)
        
        # Create pool
        self.amm.create_liquidity_pool({
            'route_key': route_key,
            'origin': origin,
            'destination': destination,
            'initial_price': initial_price,
            'initial_liquidity': initial_price * 20  # Seed liquidity
        })
        
        self.logger.info(f"Created AMM pool for route {route_key} with initial price {initial_price}")

    def _get_route_key(self, origin, destination, time_window=None):
        """
        Create a standardized key for a route
        
        Args:
            origin: Origin coordinates [x, y]
            destination: Destination coordinates [x, y]
            time_window: Optional time window
            
        Returns:
            Route key string
        """
        # Convert to tuples
        origin = tuple(origin)
        destination = tuple(destination)
        
        if time_window is not None:
            # Round time to nearest hour (simplifies clustering similar times)
            rounded_time = round(time_window / 12) * 12  # 12 ticks per hour
            return f"{origin}_{destination}_{rounded_time}"
        else:
            return f"{origin}_{destination}"

    def _parse_route_key(self, route_key):
        """
        Parse a route key into origin, destination, and time
        
        Args:
            route_key: Route key string
            
        Returns:
            Tuple of (origin, destination, time_window)
        """
        parts = route_key.split('_')
        if len(parts) >= 5:  # Full format with coordinates and time
            # Format is like "(x,y)_(x,y)_time"
            origin_str = parts[0] + '_' + parts[1]
            dest_str = parts[2] + '_' + parts[3]
            time_window = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0
            
            # Parse coordinates
            try:
                origin = eval(origin_str)
                destination = eval(dest_str)
                return origin, destination, time_window
            except:
                self.logger.warning(f"Error parsing route key: {route_key}")
                return (0, 0), (0, 0), 0
        else:
            # Simple format or parsing error
            self.logger.warning(f"Unsupported route key format: {route_key}")
            return (0, 0), (0, 0), 0

    def _estimate_route_price(self, origin, destination):
        """
        Estimate price for a route based on distance and market conditions
        
        Args:
            origin: Origin coordinates
            destination: Destination coordinates
            
        Returns:
            Estimated price
        """
        # Calculate distance
        distance = ((destination[0] - origin[0])**2 + (destination[1] - origin[1])**2)**0.5
        
        # Base price calculation
        base_price = max(5, distance * 0.8)
        
        # Check if this is a popular route with existing transactions
        route_key = self._get_route_key(origin, destination)
        
        # If we have market data, use it to influence the price
        market_price = self.marketplace.get_market_price({
            'origin': origin,
            'destination': destination,
            'service_time': self.current_step + 24  # 2 hours in future
        })
        
        if market_price:
            # Blend market price with base price (60% market, 40% base)
            return 0.6 * market_price + 0.4 * base_price
        
        # Apply time-of-day factor
        time_factor = 1.0
        if self.check_is_peak():
            time_factor = 1.2  # 20% premium during peak hours
        
        return base_price * time_factor

    def calculate_average_nft_price(self):
        """Calculate average price of active NFT listings"""
        self.logger.info(f"Current marketplace transactions: {len(self.marketplace.transaction_history)}")
        active_listings = [listing for listing in self.marketplace.listings.values() 
                        if listing.get('status') == 'active']
        
        # IMPORTANT FIX: Sync transaction count with blockchain before calculating stats
        if self.transaction_count == 0 and hasattr(self.blockchain_interface, 'stats'):
            confirmed_tx = self.blockchain_interface.stats.get('transactions_confirmed', 0)
            if confirmed_tx > 0:
                self.transaction_count = confirmed_tx
                self.logger.info(f"Synced transaction count from blockchain: {confirmed_tx}")
                
        # IMPORTANT FIX: Update completed trips count if it's zero but we have transactions
        if self.completed_trips_count == 0 and self.transaction_count > 0:
            self.completed_trips_count = self.transaction_count
            self.logger.info(f"Synced completed trips count from transactions: {self.transaction_count}")
        
        if not active_listings:
            return 0
        
        # Update active listings count for statistics
        self.active_listings_count = len(active_listings)
            
        return sum(listing.get('current_price', 0) for listing in active_listings) / len(active_listings)
    
    def count_active_commuters(self):
        """Count active commuters with ongoing requests"""
        return sum(1 for commuter in self.commuters.values() 
                 if hasattr(commuter, 'has_active_request') and commuter.has_active_request())

    def count_active_providers(self):
        """Count providers with available capacity"""
        return sum(1 for provider in self.providers.values() 
                 if provider.available_capacity > 0)

    def count_completed_trips(self):
        """Count total completed trips"""
        # Instead of calculating this every time, use the tracked counter
        return self.completed_trips_count

    def calculate_amm_volume(self):
        """Calculate total AMM trading volume"""
        if not self.enable_amm:
            return 0
            
        total_volume = 0
        for pool in self.amm.liquidity_pools.values():
            for trade in pool.get('trades', []):
                total_volume += trade.get('input_amount', 0)
                
        return total_volume

    def get_average_execution_time(self):
        """Get average execution time across all components"""
        execution_times = {}
        
        for component, times in self.execution_times.items():
            if times:
                execution_times[component] = sum(times) / len(times)
                
        return execution_times

    def on_service_completed(self, provider_id, commuter_id, was_on_time, nft_id):
        """
        Callback for completed service to update statistics and agent behavior
        
        Args:
            provider_id: ID of service provider
            commuter_id: ID of commuter
            was_on_time: Whether service was provided on time
            nft_id: ID of the service NFT
        """
        # Update provider statistics
        if provider_id in self.providers:
            provider = self.providers[provider_id]
            
            # Update provider reliability based on performance
            if was_on_time:
                provider.reliability = min(100, provider.reliability + 0.5)
            else:
                provider.reliability = max(50, provider.reliability - 1.0)
            
            # Restore capacity
            provider.available_capacity = min(provider.capacity, provider.available_capacity + 1)
        
        # Update commuter satisfaction
        if commuter_id in self.commuters:
            commuter = self.commuters[commuter_id]
            
            if hasattr(commuter, 'update_provider_satisfaction'):
                satisfaction = 1.0 if was_on_time else 0.5
                commuter.update_provider_satisfaction(provider_id, satisfaction)
        
        # IMPORTANT FIX: Update trip count directly 
        self.completed_trips_count += 1
        self.logger.info(f"Service completion callback: Incremented trips to {self.completed_trips_count}")
        
        # Update route transaction counts
        if nft_id in self.marketplace.listings:
            listing = self.marketplace.listings[nft_id]
            if 'details' in listing:
                # Get route details
                details = listing['details']
                route_key = self._get_route_key(
                    details.get('origin', [0, 0]),
                    details.get('destination', [0, 0]),
                    details.get('service_time', 0)
                )
                
                # Count transaction
                self.route_transactions[route_key] = self.route_transactions.get(route_key, 0) + 1
        
        self.logger.info(f"Service completed: Provider {provider_id}, Commuter {commuter_id}, NFT {nft_id}, On time: {was_on_time}")
        
    def run_model(self, steps):
        """Run model for specified number of steps"""
        # Ensure registrations are confirmed before starting simulation
        self._ensure_commuter_registrations()
        
        for i in range(steps):
            self.step()
            
            # Log progress
            if (i+1) % 20 == 0:
                self.logger.info(f"Completed step {i+1}/{steps}")
                
        self.logger.info(f"Model run completed: {steps} steps executed")
        return self.datacollector.get_model_vars_dataframe()

    def _ensure_commuter_registrations(self):
        """Wait for commuter registrations to be confirmed"""
        self.logger.info("Ensuring all commuter registrations are confirmed...")
        
        # Process pending transactions
        self.blockchain_interface._check_pending_transactions()
        
        # Wait for a reasonable time
        for _ in range(5):  # Try up to 5 times
            all_confirmed = True
            for commuter_id in self.commuters:
                if not self.blockchain_interface.is_commuter_registered(commuter_id):
                    all_confirmed = False
                    self.logger.info(f"Waiting for commuter {commuter_id} registration...")
                    break
            
            if all_confirmed:
                self.logger.info("All commuter registrations confirmed!")
                return
                
            # Process more transactions and wait
            self.blockchain_interface._check_pending_transactions()
            time.sleep(2)
        
        self.logger.warning("Not all registrations confirmed. Proceeding with caution.")