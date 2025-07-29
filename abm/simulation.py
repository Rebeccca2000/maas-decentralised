from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from abm.agents.commuter import Commuter
from abm.agents.service_provider import ServiceProvider
from abm.agents.maas_agent import MaaS
from abm.utils.blockchain_interface import BlockchainInterface
import uuid
import random
import json

class MobilityModel(Model):
    def __init__(self, db_connection_string, num_commuters, grid_width, grid_height, data_income_weights,
                data_health_weights, data_payment_weights, data_age_distribution,
                data_disability_weights, data_tech_access_weights,
                ASC_VALUES, UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS,
                UTILITY_FUNCTION_BASE_COEFFICIENTS, PENALTY_COEFFICIENTS,
                AFFORDABILITY_THRESHOLDS, FLEXIBILITY_ADJUSTMENTS, VALUE_OF_TIME,
                public_price_table, ALPHA_VALUES,
                DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS,
                BACKGROUND_TRAFFIC_AMOUNT, CONGESTION_ALPHA,
                CONGESTION_BETA, CONGESTION_CAPACITY, CONGESTION_T_IJ_FREE_FLOW,
                uber_like1_capacity, uber_like1_price, 
                uber_like2_capacity, uber_like2_price, 
                bike_share1_capacity, bike_share1_price, 
                bike_share2_capacity, bike_share2_price, 
                subsidy_dataset, subsidy_config, 
                use_blockchain=False, blockchain_config="blockchain_config.json", schema=None):
        # Original initialization
        self.db_engine = create_engine(db_connection_string)
        self.schema = schema
        
        if self.schema:
            self.engine = create_engine(db_connection_string)
            with self.engine.connect() as connection:
                connection.execute(text(f"SET search_path TO {self.schema}"))
                
        self.Session = scoped_session(sessionmaker(bind=self.db_engine))
        self.session = self.Session()
        
        # Reset database with dynamic parameters
        reset_database(self.db_engine, self.session,
                       uber_like1_capacity, uber_like1_price, 
                       uber_like2_capacity, uber_like2_price, 
                       bike_share1_capacity, bike_share1_price, 
                       bike_share2_capacity, bike_share2_price, self.schema)
        
        super().__init__()
        self.db_connection_string = db_connection_string

        self.grid = MultiGrid(grid_width, grid_height, torus=False)
        self.schedule = RandomActivation(self)

        self.num_commuters = num_commuters
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.data_income_weights = data_income_weights
        self.data_health_weights = data_health_weights
        self.data_payment_weights = data_payment_weights
        self.data_age_distribution = data_age_distribution
        self.data_disability_weights = data_disability_weights
        self.data_tech_access_weights = data_tech_access_weights
        self.subsidy_config = subsidy_config
        
        # Parameters for commuter agent
        self.asc_values = ASC_VALUES
        self.utility_function_high_income_car_coefficients = UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS
        self.utility_function_base_coefficients = UTILITY_FUNCTION_BASE_COEFFICIENTS
        self.penalty_coefficients = PENALTY_COEFFICIENTS
        self.affordability_thresholds = AFFORDABILITY_THRESHOLDS
        self.flexibility_adjustments = FLEXIBILITY_ADJUSTMENTS
        self.value_of_time = VALUE_OF_TIME
        
        # Parameters for ServiceProvider
        self.alpha_values = ALPHA_VALUES
        self.public_price_table = public_price_table
        
        # Initialize ServiceProvider agent
        self.service_provider_agent = ServiceProvider(
            unique_id='service_provider_1', 
            model=self,
            db_connection_string=db_connection_string,
            ALPHA_VALUES=self.alpha_values,
            public_price_table=self.public_price_table,
            schema=self.schema
        )
        self.schedule.add(self.service_provider_agent)
        
        # Initialize stations
        self.current_step = 0
        for mode, mode_stations in database_01.stations.items():
            for station_id, (x, y) in mode_stations.items():
                station_agent = StationAgent(
                    unique_id=f"{mode}_{station_id}", 
                    model=self, 
                    location=(x, y), 
                    mode=mode
                )
                self.grid.place_agent(station_agent, (x, y))
                self.schedule.add(station_agent)

        # Initialize commuter agents
        self.commuter_agents = []
        income_levels = ['low', 'middle', 'high']
        health_statuses = ['good', 'poor']
        payment_schemes = ['PAYG', 'subscription']
        
        # Setup age distribution
        age_distribution = self.data_age_distribution
        cumulative_age_weights = []
        current_weight = 0
        for age_range, weight in age_distribution.items():
            current_weight += weight
            cumulative_age_weights.append((age_range, current_weight))

        def get_random_age():
            rnd = random.random()
            for age_range, cumulative_weight in cumulative_age_weights:
                if rnd <= cumulative_weight:
                    return random.randint(age_range[0], age_range[1])
            return random.randint(0, 70)
            
        # Create commuters
        for i in range(num_commuters):
            income_level = random.choices(income_levels, self.data_income_weights)[0]
            health_status = random.choices(health_statuses, self.data_health_weights)[0]
            payment_scheme = random.choices(payment_schemes, self.data_payment_weights)[0]
            age = get_random_age()
            has_disability = random.choices([True, False], self.data_disability_weights)[0]
            tech_access = random.choices([True, False], self.data_tech_access_weights)[0]

            commuter = Commuter(
                unique_id=i + 2,
                model=self,
                commuter_location=(random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)),
                age=age,
                income_level=income_level,
                has_disability=has_disability,
                tech_access=tech_access,
                health_status=health_status,
                payment_scheme=payment_scheme,
                ASC_VALUES=self.asc_values,
                UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS=self.utility_function_high_income_car_coefficients,
                UTILITY_FUNCTION_BASE_COEFFICIENTS=self.utility_function_base_coefficients,
                PENALTY_COEFFICIENTS=self.penalty_coefficients,
                AFFORDABILITY_THRESHOLDS=self.affordability_thresholds,
                FLEXIBILITY_ADJUSTMENTS=self.flexibility_adjustments,
                VALUE_OF_TIME=self.value_of_time,
                subsidy_dataset=subsidy_dataset
            )
            self.commuter_agents.append(commuter)
            self.schedule.add(commuter)
            self.grid.place_agent(commuter, commuter.location)
            self.record_commuter_info(commuter)
        
        # Parameters for MaaS agent
        self.congestion_alpha = CONGESTION_ALPHA
        self.congestion_beta = CONGESTION_BETA
        self.congestion_capacity = CONGESTION_CAPACITY
        self.conjestion_t_ij_free_flow = CONGESTION_T_IJ_FREE_FLOW
        self.dynamic_maas_surcharge_base_coefficient = DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS
        self.background_traffic_amount = BACKGROUND_TRAFFIC_AMOUNT
        
        # Initialize MaaS agent
        self.maas_agent = MaaS(
            unique_id="maas_1", 
            model=self,
            service_provider_agent=self.service_provider_agent,
            commuter_agents=self.commuter_agents,
            DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS=self.dynamic_maas_surcharge_base_coefficient,
            BACKGROUND_TRAFFIC_AMOUNT=self.background_traffic_amount,
            stations=database_01.stations, 
            routes=database_01.routes,
            transfers=database_01.transfers, 
            num_commuters=self.num_commuters,
            grid_width=self.grid_width, 
            grid_height=self.grid_height,
            CONGESTION_ALPHA=self.congestion_alpha,
            CONGESTION_BETA=self.congestion_beta,
            CONGESTION_CAPACITY=self.congestion_capacity,
            CONGESTION_T_IJ_FREE_FLOW=self.conjestion_t_ij_free_flow,
            subsidy_config=self.subsidy_config,
            schema=self.schema
        )
        self.schedule.add(self.maas_agent)
        
        # Blockchain integration
        self.use_blockchain = use_blockchain
        if self.use_blockchain:
            self.blockchain_interface = BlockchainInterface(config_file=blockchain_config)
            self.pending_auctions = {}  # Track auctions that are in progress
            print("Blockchain integration enabled. Connected to network.")
        
    def get_current_step(self):
        return self.current_step
    
    def step(self):
        """Run a step of the model"""
        self.current_step += 1
        print(f"Step {self.current_step}")
        
        # Update service provider time steps
        self.service_provider_agent.update_time_steps()

        # Initialize availability
        availability_dict = self.service_provider_agent.initialize_availability(self.current_step - 1)

        # Process commuter requests
        for commuter in self.commuter_agents:
            # Create new requests
            self.create_time_based_trip(self.current_step, commuter)
            
            # Clean up stale requests
            for request_id, request in list(commuter.requests.items()):
                if request['status'] == 'active' and request['start_time'] < self.current_step:
                    request['status'] = 'expired'
                    print(f"Marking stale request {request_id} as expired")
            
            # Process valid active requests
            self.process_commuter_requests(commuter, availability_dict)
            
            # Update commuter info in database
            self.update_commuter_info_log(commuter)
            
            # Update commuter location
            commuter.update_location()
            commuter.check_travel_status()
            
        # If using blockchain, process pending auctions
        if self.use_blockchain:
            self.process_blockchain_auctions()
            
        # Update service provider availability and pricing
        self.service_provider_agent.update_availability()
        self.service_provider_agent.dynamic_pricing_share()
        
        # Insert background traffic
        with self.Session() as session:
            num_routes = self.maas_agent.insert_time_varying_traffic(session)
            
        # Advance simulation
        self.schedule.step()
    
    def process_commuter_requests(self, commuter, availability_dict):
        """Process active requests for a commuter"""
        for request_id, request in list(commuter.requests.items()):
            try:
                if request['status'] == 'active' and request['start_time'] >= self.current_step:
                    # If using blockchain, submit to blockchain auction
                    if self.use_blockchain and commuter.blockchain_preference():
                        # If not already on blockchain, create request
                        if not request.get('blockchain_id'):
                            success, blockchain_id = self.blockchain_interface.create_travel_request(
                                commuter, request
                            )
                            if success and blockchain_id:
                                request['blockchain_id'] = blockchain_id
                                # Add to pending auctions
                                self.pending_auctions[blockchain_id] = {
                                    'request_id': request_id,
                                    'commuter_id': commuter.unique_id,
                                    'status': 'active',
                                    'created_at': self.current_step
                                }
                        continue
                    
                    # Traditional (off-chain) processing
                    travel_options_without_MaaS = self.maas_agent.options_without_maas(
                        request_id, request['start_time'], request['origin'], request['destination']
                    )
                    
                    travel_options_with_MaaS = self.maas_agent.maas_options(
                        commuter.payment_scheme, request_id, request['start_time'], 
                        request['origin'], request['destination']
                    )
                    
                    # Rank options and book service
                    ranked_options = commuter.rank_service_options(
                        travel_options_without_MaaS, travel_options_with_MaaS, request_id
                    )
                    
                    if ranked_options:
                        booking_success, availability_dict = self.maas_agent.book_service(
                            request_id, ranked_options, self.current_step, availability_dict
                        )
                        if not booking_success:
                            print(f"Booking for request {request_id} was not successful.")
                    else:
                        print(f"No viable options for request {request_id}.")
            except Exception as e:
                print(f"Error processing request {request_id}: {str(e)}")
                
    def process_blockchain_auctions(self):
        """Process pending blockchain auctions"""
        current_time = self.current_step
        auctions_to_finalize = []
        
        # Check for auctions that need to be finalized
        for blockchain_id, auction_info in self.pending_auctions.items():
            if auction_info['status'] == 'active':
                # Auctions last for 10 steps (configurable)
                if current_time - auction_info['created_at'] >= 10:
                    auctions_to_finalize.append(blockchain_id)
        
        # Finalize auctions
        for blockchain_id in auctions_to_finalize:
            auction_info = self.pending_auctions[blockchain_id]
            success, winning_offers = self.blockchain_interface.finalize_auction(blockchain_id)
            
            if success and winning_offers:
                # Process winning offers
                request_id = auction_info['request_id']
                commuter_id = auction_info['commuter_id']
                
                # Get commuter agent
                commuter = next((c for c in self.commuter_agents if c.unique_id == commuter_id), None)
                if commuter:
                    # Get request
                    request = commuter.requests.get(request_id)
                    if request:
                        # Extract winning offer details
                        winning_offer = winning_offers[0]  # Take first if multiple
                        provider_id = winning_offer['providerId']
                        price = self.blockchain_interface.w3.fromWei(winning_offer['price'], 'ether')
                        route_details = json.loads(winning_offer['routeDetails'])
                        
                        # Accept service and complete agreement
                        commuter.accept_service_blockchain(
                            blockchain_id, provider_id, price, route_details
                        )
                
                # Update auction status
                self.pending_auctions[blockchain_id]['status'] = 'finalized'
                
    def blockchain_preference(self, commuter):
        """Determine if commuter prefers using blockchain"""
        # This can be a sophisticated decision based on commuter attributes
        # For now, use a simple probability
        base_probability = 0.5  # 50% chance by default
        
        # Adjust based on income level
        if commuter.income_level == 'high':
            base_probability += 0.2  # High income more likely to use blockchain
        elif commuter.income_level == 'low':
            base_probability -= 0.1  # Low income less likely to use blockchain
            
        # Adjust based on tech access
        if not commuter.tech_access:
            base_probability -= 0.3  # Without tech access, much less likely
            
        # Adjust based on payment scheme
        if commuter.payment_scheme == 'subscription':
            base_probability += 0.1  # Subscription users more likely
            
        # Ensure probability is between 0 and 1
        blockchain_prob = max(0, min(base_probability, 1.0))
        
        # Make random decision
        return random.random() < blockchain_prob
    
    def create_time_based_trip(self, current_step, commuter):
        """Create trip requests with realistic timing patterns based on time of day and commuter attributes"""
        # Original implementation for creating trips
        # Only create new trips if commuter isn't already traveling
        if not commuter.requests or self.all_requests_finished(commuter):
            # Skip if daily trip limit reached
            if not self.should_create_trip(commuter, current_step):
                return False
                
            # Time of day context
            ticks_in_day = 144
            current_day_tick = current_step % ticks_in_day
            current_day = current_step // ticks_in_day
            day_of_week = current_day % 7
            is_weekend = day_of_week >= 5
            
            # Calculate baseline probabilities based on demographics
            base_probability = 0.05
            
            # Income level adjustments
            if commuter.income_level == 'high':
                base_probability *= 1.5
            elif commuter.income_level == 'low':
                base_probability *= 0.8
                
            # Age/disability adjustments
            if commuter.age >= 65 or commuter.has_disability:
                base_probability *= 0.7
                
            # Payment scheme adjustments
            if commuter.payment_scheme == 'subscription':
                base_probability *= 1.3
                
            # Time of day probability
            time_multiplier = self.calculate_time_multiplier(current_day_tick, is_weekend)
            
            # Final probability
            trip_probability = base_probability * time_multiplier
            
            # Decide whether to create a trip
            if random.random() < trip_probability:
                # Determine trip purpose based on time of day
                purpose = self.determine_trip_purpose(current_day_tick, is_weekend)
                
                # Generate destination based on purpose
                origin = commuter.location
                destination = self.get_purpose_based_destination(purpose, origin, commuter)
                
                # Set trip timing with realistic start time distribution
                min_delay = 1
                max_delay = 5
                
                if purpose in ['work', 'school'] and current_day_tick < 30:
                    max_delay = 4
                    
                start_time = current_step + min_delay + random.randint(0, max_delay)
                start_time = min(start_time, current_step + 5)  # Ensure not too far ahead
                
                # Create the request
                request_id = uuid.uuid4()
                commuter.create_request(request_id, origin, destination, start_time, purpose)
                return True
            
            return False
            
    def calculate_time_multiplier(self, current_day_tick, is_weekend):
        """Calculate time-of-day probability multiplier"""
        import math
        
        if not is_weekend:
            # Weekday pattern with morning and evening peaks
            morning_peak_center = 48  # 8am
            morning_intensity = math.exp(-0.5 * ((current_day_tick - morning_peak_center) / 8) ** 2)
            
            evening_peak_center = 105  # 5:30pm
            evening_intensity = math.exp(-0.5 * ((current_day_tick - evening_peak_center) / 10) ** 2)
            
            return max(morning_intensity * 3.0, evening_intensity * 2.5, 0.2)
        else:
            # Weekend pattern with midday peak
            midday_peak_center = 72  # Noon
            midday_intensity = math.exp(-0.5 * ((current_day_tick - midday_peak_center) / 16) ** 2)
            return midday_intensity * 1.5
            
    def determine_trip_purpose(self, current_day_tick, is_weekend):
        """Determine trip purpose based on time of day"""
        if not is_weekend and current_day_tick < 60:  # Morning on weekday
            purpose_weights = {'work': 0.7, 'school': 0.2, 'shopping': 0.05, 'medical': 0.03, 'trip': 0.02}
        elif not is_weekend and current_day_tick >= 90 and current_day_tick < 114:  # Evening on weekday
            purpose_weights = {'work': 0.1, 'school': 0.05, 'shopping': 0.3, 'trip': 0.5, 'medical': 0.05}
        elif is_weekend:  # Weekend
            purpose_weights = {'shopping': 0.4, 'trip': 0.4, 'medical': 0.05, 'work': 0.1, 'school': 0.05}
        else:  # Middle of weekday
            purpose_weights = {'work': 0.2, 'school': 0.1, 'shopping': 0.3, 'medical': 0.2, 'trip': 0.2}
        
        # Select purpose based on weights
        purposes = list(purpose_weights.keys())
        weights = list(purpose_weights.values())
        return random.choices(purposes, weights=weights)[0]
        
    # Rest of the original methods (should_create_trip, all_requests_finished, etc.)
    # ...
    
    def run_model(self, num_steps):
        """Run the model for the specified number of steps"""
        for _ in range(num_steps):
            self.step()