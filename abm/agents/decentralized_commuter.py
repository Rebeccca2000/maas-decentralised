from mesa import Agent
import numpy as np
import math
import uuid
import logging
import random
from datetime import datetime, timedelta

class DecentralizedCommuter(Agent):
    """
    A commuter agent with blockchain identity capable of requesting, purchasing, 
    and trading mobility services through the blockchain.
    
    This agent makes decisions based on utility calculations and learns from past 
    experiences to adapt its strategy over time.
    """
    def __init__(self, unique_id, model, location, age, income_level,
                 has_disability, tech_access, health_status, payment_scheme,
                 blockchain_interface):
        """
        Initialize a new decentralized commuter agent.
        
        Args:
            unique_id: Unique identifier for the agent
            model: The model containing the agent
            location: Current location coordinates (x, y)
            age: Age of the commuter
            income_level: Income level ('low', 'middle', 'high')
            has_disability: Whether the commuter has a disability
            tech_access: Whether the commuter has access to technology
            health_status: Health status ('good', 'poor')
            payment_scheme: Payment scheme ('PAYG', 'subscription')
            blockchain_interface: Interface to the blockchain
        """
        super().__init__(unique_id, model)
        # Basic attributes
        self.location = location
        self.age = age
        self.income_level = income_level
        self.has_disability = has_disability
        self.tech_access = tech_access
        self.health_status = health_status
        self.payment_scheme = payment_scheme
        
        # Travel patterns
        self.home_location = location
        self.work_location = None
        self.regular_destinations = {}  
        self.current_destination = None
        self.current_path = []
        self.current_mode = None
        
        # Blockchain-specific attributes
        self.blockchain_interface = blockchain_interface
        self.blockchain_address = None
        self.owned_nfts = {}  # NFTs owned by this commuter
        self.requests = {}  # All requests created by this commuter
        self.active_trips = {}  # Currently active trips
        self.trip_history = []  # History of completed trips
        self.transaction_history = []  # History of blockchain transactions
        
        # Decision-making parameters
        self.utility_coefficients = self._initialize_utility_coefficients()
        self.risk_aversion = self._initialize_risk_aversion()
        self.time_flexibility = self._initialize_time_flexibility()
        self.price_sensitivity = self._initialize_price_sensitivity()
        self.comfort_preference = self._initialize_comfort_preference()
        self.reliability_preference = self._initialize_reliability_preference()
        
        # Learning parameters
        self.market_experience = {}  # Provider ID -> satisfaction level
        self.mode_preference = self._initialize_mode_preference()
        self.strategy_weights = {
            'direct_booking': 0.5,  # Book directly with provider
            'market_purchase': 0.3,  # Purchase from NFT marketplace
            'bundled_service': 0.2   # Use bundled services
        }
        
        # Track requests that need attention
        self.pending_requests = []
        
        # Set up logging
        self.logger = logging.getLogger(f"Commuter-{unique_id}")
        self.logger.setLevel(logging.INFO)
        
        # Register with blockchain
        self._register_with_blockchain()

    def _initialize_utility_coefficients(self):
        """
        Initialize utility coefficients based on the commuter's attributes.
        
        Returns:
            Dictionary of utility coefficients for different factors
        """
        # Base coefficients
        coefficients = {
            'price': -1.0,           # Negative because higher price means lower utility
            'time': -0.8,            # Negative because longer time means lower utility
            'comfort': 0.5,          # Positive because higher comfort means higher utility
            'reliability': 0.7,      # Positive because higher reliability means higher utility
            'convenience': 0.4,      # Positive because higher convenience means higher utility
            'familiarity': 0.3,      # Positive because familiar services have higher utility
            'safety': 0.6            # Positive because safer services have higher utility
        }
        
        # Adjust based on income level
        if self.income_level == 'low':
            coefficients['price'] *= 1.5       # More price-sensitive
            coefficients['comfort'] *= 0.8     # Less comfort-sensitive
        elif self.income_level == 'high':
            coefficients['price'] *= 0.7       # Less price-sensitive
            coefficients['comfort'] *= 1.2     # More comfort-sensitive
            coefficients['convenience'] *= 1.3 # Values convenience more
        
        # Adjust based on age
        if self.age > 60:
            coefficients['comfort'] *= 1.3     # Older people value comfort more
            coefficients['safety'] *= 1.2      # Older people value safety more
            coefficients['time'] *= 0.9        # May be less time-sensitive
        elif self.age < 30:
            coefficients['time'] *= 1.2        # Younger people are more time-sensitive
            coefficients['familiarity'] *= 0.8 # Less concerned about familiarity
        
        # Adjust for disability
        if self.has_disability:
            coefficients['comfort'] *= 1.5
            coefficients['convenience'] *= 1.4
            coefficients['time'] *= 0.8  # May be less time-sensitive due to accessibility needs
        
        # Adjust for health status
        if self.health_status == 'poor':
            coefficients['comfort'] *= 1.2
            coefficients['safety'] *= 1.3
        
        # Add some randomness for heterogeneity
        for key in coefficients:
            coefficients[key] *= random.uniform(0.9, 1.1)
        
        return coefficients

    def _initialize_risk_aversion(self):
        """
        Initialize risk aversion parameter based on the commuter's attributes.
        
        Returns:
            Float between 0 and 1, higher means more risk-averse
        """
        # Base risk aversion
        risk_aversion = 0.5
        
        # Adjust based on age (older people tend to be more risk-averse)
        if self.age > 60:
            risk_aversion += 0.2
        elif self.age < 30:
            risk_aversion -= 0.1
        
        # Adjust based on income (higher income can afford more risk)
        if self.income_level == 'high':
            risk_aversion -= 0.1
        elif self.income_level == 'low':
            risk_aversion += 0.1
        
        # Adjust based on health status
        if self.health_status == 'poor':
            risk_aversion += 0.15
        
        # Adjust for tech access (less tech access means more risk-averse with digital platforms)
        if not self.tech_access:
            risk_aversion += 0.15
        
        # Add randomness
        risk_aversion += random.uniform(-0.1, 0.1)
        
        # Ensure value is between 0 and 1
        return max(0, min(1, risk_aversion))

    def _initialize_time_flexibility(self):
        """
        Initialize time flexibility parameter based on the commuter's attributes.
        
        Returns:
            Float between 0 and 1, higher means more flexible with time
        """
        # Base time flexibility
        flexibility = 0.5
        
        # Adjust based on age
        if self.age > 60:
            flexibility += 0.2  # Retired, more flexible
        elif 30 <= self.age <= 50:
            flexibility -= 0.2  # Working age, less flexible
        
        # Adjust based on income
        if self.income_level == 'high':
            flexibility -= 0.1  # Higher value of time
        
        # Adjust based on payment scheme
        if self.payment_scheme == 'subscription':
            flexibility += 0.1  # Subscription users might be more flexible
        
        # Add randomness
        flexibility += random.uniform(-0.1, 0.1)
        
        # Ensure value is between 0 and 1
        return max(0, min(1, flexibility))

    def _initialize_price_sensitivity(self):
        """
        Initialize price sensitivity parameter based on the commuter's attributes.
        
        Returns:
            Float between 0 and 1, higher means more sensitive to price
        """
        # Base price sensitivity
        sensitivity = 0.5
        
        # Adjust based on income level
        if self.income_level == 'low':
            sensitivity += 0.3
        elif self.income_level == 'high':
            sensitivity -= 0.3
        
        # Adjust based on payment scheme
        if self.payment_scheme == 'subscription':
            sensitivity -= 0.1  # Subscription users might be less price-sensitive
        
        # Add randomness
        sensitivity += random.uniform(-0.1, 0.1)
        
        # Ensure value is between 0 and 1
        return max(0, min(1, sensitivity))

    def _initialize_comfort_preference(self):
        """
        Initialize comfort preference parameter based on the commuter's attributes.
        
        Returns:
            Float between 0 and 1, higher means more preference for comfort
        """
        # Base comfort preference
        preference = 0.5
        
        # Adjust based on age
        if self.age > 60:
            preference += 0.2
        
        # Adjust based on income level
        if self.income_level == 'high':
            preference += 0.2
        elif self.income_level == 'low':
            preference -= 0.1
        
        # Adjust based on disability and health
        if self.has_disability:
            preference += 0.3
        if self.health_status == 'poor':
            preference += 0.2
        
        # Add randomness
        preference += random.uniform(-0.1, 0.1)
        
        # Ensure value is between 0 and 1
        return max(0, min(1, preference))

    def _initialize_reliability_preference(self):
        """
        Initialize reliability preference parameter based on the commuter's attributes.
        
        Returns:
            Float between 0 and 1, higher means more preference for reliability
        """
        # Base reliability preference
        preference = 0.5
        
        # Adjust based on purpose (would depend on typical travel purposes)
        # Will be adjusted dynamically based on purpose
        
        # Adjust based on tech access
        if not self.tech_access:
            preference += 0.1  # Less tech-savvy users may value reliability more
        
        # Add randomness
        preference += random.uniform(-0.1, 0.1)
        
        # Ensure value is between 0 and 1
        return max(0, min(1, preference))

    def _initialize_mode_preference(self):
        """
        Initialize mode preference based on the commuter's attributes.
        
        Returns:
            Dictionary of mode preferences
        """
        # Base preferences
        preferences = {
            'car': 0.2,
            'bike': 0.2, 
            'bus': 0.2,
            'train': 0.2,
            'walk': 0.2
        }
        
        # Adjust based on age
        if self.age > 60:
            preferences['car'] += 0.1
            preferences['walk'] -= 0.1
        elif self.age < 30:
            preferences['bike'] += 0.1
            preferences['car'] -= 0.05
        
        # Adjust based on disability
        if self.has_disability:
            preferences['car'] += 0.2
            preferences['bike'] -= 0.1
            preferences['walk'] -= 0.1
        
        # Adjust based on health status
        if self.health_status == 'poor':
            preferences['car'] += 0.1
            preferences['walk'] -= 0.1
            preferences['bike'] -= 0.1
        
        # Add randomness
        for mode in preferences:
            preferences[mode] += random.uniform(-0.05, 0.05)
        
        # Normalize preferences
        total = sum(preferences.values())
        for mode in preferences:
            preferences[mode] /= total
        
        return preferences

    def _register_with_blockchain(self):
        """Register the commuter on the blockchain."""
        self.logger.info(f"Registering commuter {self.unique_id} with blockchain")
        success, self.blockchain_address = self.blockchain_interface.register_commuter(self)
        
        if success:
            self.logger.info(f"Commuter {self.unique_id} registered at address {self.blockchain_address}")
            # Don't create requests until registration is confirmed
            self.registration_confirmed = False
        else:
            self.logger.error(f"Failed to register commuter {self.unique_id} with blockchain")
            self.registration_confirmed = False
            
    def has_active_request(self):
        """Check if the commuter has any active requests"""
        for request_id, request in self.requests.items():
            if request['status'] in ['active', 'seeking_offers', 'service_selected']:
                return True
        return False
    
    def get_personal_requirements(self):
        """
        Get the personal requirements of the commuter.
        
        Returns:
            Dictionary of personal requirements
        """
        requirements = {
            'wheelchair': self.has_disability,
            'assistance': self.has_disability or (self.age > 70),
            'child_seat': False,  # Default, can be updated based on trip purpose
            'pet_friendly': False  # Default, can be updated based on trip purpose
        }
        
        # Could add more complex logic based on commuter attributes
        
        return requirements

    def determine_schedule_flexibility(self, travel_purpose):
        """
        Determine the schedule flexibility based on travel purpose and commuter attributes.
        
        Args:
            travel_purpose: Purpose of travel
            
        Returns:
            String indicating flexibility level
        """
        # Base flexibility based on commuter's time flexibility parameter
        if self.time_flexibility < 0.3:
            base_flexibility = "low"
        elif self.time_flexibility < 0.7:
            base_flexibility = "medium"
        else:
            base_flexibility = "high"
        
        # Adjust based on travel purpose
        if travel_purpose == 'work':
            # Work trips usually have less flexibility
            if base_flexibility == "high":
                return "medium"
            elif base_flexibility == "medium":
                return "low"
            else:
                return "very_low"
        elif travel_purpose == 'medical':
            # Medical appointments usually have less flexibility
            if base_flexibility == "high":
                return "medium"
            elif base_flexibility == "medium":
                return "low"
            else:
                return "very_low"
        elif travel_purpose == 'leisure':
            # Leisure trips usually have more flexibility
            if base_flexibility == "low":
                return "medium"
            elif base_flexibility == "medium":
                return "high"
            else:
                return "very_high"
        
        # Default to base flexibility for other purposes
        return base_flexibility

    def create_request(self, origin, destination, start_time, travel_purpose='work', requirements=None):
        """
        Create a travel request and submit it to the blockchain.
        
        Args:
            origin: Origin coordinates [x, y]
            destination: Destination coordinates [x, y]
            start_time: Start time for the trip
            travel_purpose: Purpose of travel
            requirements: Optional custom requirements
            
        Returns:
            Request ID
        """
        # Check if registration is confirmed
        if not self.blockchain_interface.is_commuter_registered(self.unique_id):
            self.logger.info(f"Commuter {self.unique_id} not yet registered. Queueing request.")
            # Store the request to process later
            if not hasattr(self, 'pending_outgoing_requests'):
                self.pending_outgoing_requests = []
            
            self.pending_outgoing_requests.append({
                'origin': origin,
                'destination': destination,
                'start_time': start_time,
                'travel_purpose': travel_purpose,
                'requirements': requirements,
                'queued_at': self.model.schedule.time
            })
            
            return None  # Return None to indicate request is queued
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        
        # Get personal requirements
        personal_reqs = self.get_personal_requirements()
        if requirements:
            # Override with custom requirements if provided
            for key, value in requirements.items():
                personal_reqs[key] = value
        
        # Determine schedule flexibility
        flexibility = self.determine_schedule_flexibility(travel_purpose)
        
        # Map purpose to enum value (matching blockchain expectations)
        purpose_map = {
            'work': 0,
            'school': 1,
            'shopping': 2,
            'medical': 3,
            'trip': 4,
            'leisure': 6,
            'other': 7
        }
        
        # Use default if purpose not in map
        purpose_value = purpose_map.get(travel_purpose, 7)
        
        # Convert requirements to format expected by blockchain
        requirement_keys = list(personal_reqs.keys())
        requirement_values = [personal_reqs[key] for key in requirement_keys]
        
        # Create request structure
        request = {
            'request_id': request_id,
            'commuter_id': self.unique_id,
            'origin': origin,
            'destination': destination,
            'start_time': start_time,
            'travel_purpose': purpose_value,
            'flexible_time': flexibility,
            'requirement_keys': requirement_keys,
            'requirement_values': requirement_values,
            'status': 'active',
            'blockchain_status': 'pending',  # Track blockchain state
            'created_at': self.model.schedule.time,
            'selected_strategy': None,  # Will be updated when strategy is selected
            'selected_option': None     # Will be updated when option is selected
        }
        
        self.logger.info(f"Creating request {request_id} from {origin} to {destination} at time {start_time}")
        
        # Store request locally
        self.requests[request_id] = request
        self.pending_requests.append(request_id)
        
        # Create request on blockchain asynchronously
        result = self.blockchain_interface.create_travel_request(self, request)
        
        self.logger.info(f"Request {request_id} created with blockchain result: {result}")
        
        return request_id

    def evaluate_marketplace_options(self, request_id):
        """
        Evaluate options from direct booking and NFT marketplace.
        
        Args:
            request_id: The request ID to evaluate options for
            
        Returns:
            List of (utility, option) tuples, sorted by utility (highest first)
        """
        if request_id not in self.requests:
            self.logger.error(f"Request {request_id} not found")
            return []
        
        request = self.requests[request_id]
        self.logger.info(f"Evaluating marketplace options for request {request_id}")
        
        # Get direct booking options from service providers in the model
        direct_options = []
        for provider in self.model.schedule.agents:
            if hasattr(provider, 'get_service_offers'):
                offers = provider.get_service_offers(
                    request['origin'],
                    request['destination'],
                    request['start_time']
                )
                
                for offer in offers:
                    # Standardize offer format
                    direct_options.append({
                        'type': 'direct_booking',
                        'provider_id': provider.unique_id,
                        'provider_name': getattr(provider, 'company_name', f"Provider-{provider.unique_id}"),
                        'price': offer['price'],
                        'time': offer['duration'],
                        'mode': offer['mode'],
                        'comfort': offer.get('comfort', 0.5),
                        'reliability': getattr(provider, 'reliability', 0.5),
                        'departure_time': offer['departure_time'],
                        'route': offer.get('route', [])
                    })
        
        # Get NFT marketplace options
        market_params = {
            'min_price': 0,
            'max_price': self._calculate_max_price(request),
            'min_departure': request['start_time'] - int(self.time_flexibility * 3600),  # Convert to seconds
            'max_departure': request['start_time'] + int(self.time_flexibility * 3600),
            'origin_area': [request['origin'][0], request['origin'][1], 5],  # 5 unit radius
            'dest_area': [request['destination'][0], request['destination'][1], 5]
        }
        
        market_results = self.blockchain_interface.search_nft_market(market_params)
        
        market_options = []
        for result in market_results:
            # Standardize market option format
            market_options.append({
                'type': 'nft_market',
                'nft_id': result['token_id'],
                'provider_id': result.get('provider_id', 0),
                'seller_id': result.get('seller', ''),
                'price': float(result['price']),
                'service_time': result.get('start_time', request['start_time']),
                'time': result.get('duration', 0),
                'mode': result.get('mode', 'unknown'),
                'comfort': result.get('comfort', 0.5),
                'reliability': result.get('reliability', 0.5),
                'route': result.get('route_details', {}).get('route', [])
            })
        
        self.logger.info(f"Found {len(direct_options)} direct options and {len(market_options)} market options")
        
        # Evaluate all options using utility function
        ranked_options = []
        all_options = direct_options + market_options
        
        for option in all_options:
            utility = self.calculate_option_utility(option, request_id)
            ranked_options.append((utility, option))
        
        # Sort by utility (highest first)
        ranked_options.sort(reverse=True, key=lambda x: x[0])
        
        return ranked_options

    def calculate_option_utility(self, option, request_id):
        """
        Calculate the utility of a mobility option based on commuter preferences.
        
        Args:
            option: Mobility option (direct booking or NFT)
            request_id: The request ID
            
        Returns:
            Utility value (higher is better)
        """
        request = self.requests[request_id]
        
        # Base utility calculation for common factors
        price_utility = self.utility_coefficients['price'] * option['price'] * self.price_sensitivity
        time_utility = self.utility_coefficients['time'] * option['time']
        comfort_utility = self.utility_coefficients['comfort'] * option['comfort'] * self.comfort_preference
        reliability_utility = self.utility_coefficients['reliability'] * option['reliability'] * self.reliability_preference
        
        # Calculate time mismatch penalty
        time_mismatch = abs(option.get('service_time', request['start_time']) - request['start_time'])
        time_mismatch_penalty = 0
        
        # Convert time mismatch to hours
        time_mismatch_hours = time_mismatch / 3600
        
        # Adjust based on flexibility
        if request['flexible_time'] == 'low':
            time_mismatch_penalty = -0.3 * time_mismatch_hours
        elif request['flexible_time'] == 'medium':
            time_mismatch_penalty = -0.15 * time_mismatch_hours
        elif request['flexible_time'] == 'high':
            time_mismatch_penalty = -0.05 * time_mismatch_hours
        else:
            time_mismatch_penalty = -0.1 * time_mismatch_hours
        
        # Add mode preference
        mode = option.get('mode', 'car')
        mode_utility = 0
        if mode in self.mode_preference:
            mode_utility = self.mode_preference[mode] * 0.5  # Scale factor
        
        # Add risk component based on option type
        risk_utility = 0
        if option['type'] == 'nft_market':
            # NFT purchases have different risk profile
            # Higher risk aversion means more penalty for NFT purchases
            risk_factor = self.risk_aversion * 0.2
            
            # Time proximity risk (earlier is better for NFTs to allow resale)
            time_to_service = option.get('service_time', request['start_time']) - self.model.schedule.time
            time_proximity_risk = 0
            
            if time_to_service > 24 * 3600:  # More than 24 hours in advance
                time_proximity_risk = 0.1  # Slight bonus
            elif time_to_service < 1 * 3600:  # Less than 1 hour
                time_proximity_risk = -0.3  # Significant penalty
            
            # Seller reputation risk
            seller_reputation = 0  # Default neutral
            if option.get('seller_id') in self.market_experience:
                seller_reputation = self.market_experience[option['seller_id']]
            
            risk_utility = -risk_factor + time_proximity_risk + (seller_reputation * 0.2)
        
        # Calculate total utility
        total_utility = (
            price_utility + 
            time_utility + 
            comfort_utility + 
            reliability_utility + 
            time_mismatch_penalty + 
            mode_utility +
            risk_utility
        )
        
        # Apply learning from past experiences with this provider
        if option.get('provider_id') in self.market_experience:
            exp_factor = self.market_experience[option['provider_id']]
            total_utility *= (1 + 0.2 * exp_factor)  # Adjust utility based on experience
        
        # Add a small random component (exploration)
        total_utility += random.uniform(-0.05, 0.05)
        
        return total_utility

    def _calculate_max_price(self, request=None):
        """
        Calculate the maximum price the commuter is willing to pay.
        
        Args:
            request: Optional request to calculate max price for
            
        Returns:
            Maximum price
        """
        # Base price depends on income level
        if self.income_level == 'low':
            base_max = 50
        elif self.income_level == 'middle':
            base_max = 100
        else:  # high
            base_max = 200
        
        # Adjust based on trip distance if request provided
        if request:
            origin = request['origin']
            destination = request['destination']
            distance = math.sqrt((destination[0] - origin[0])**2 + (destination[1] - origin[1])**2)
            
            # Simple linear adjustment based on distance
            distance_factor = 1 + (distance / 100)  # Adjust scale as needed
            base_max *= distance_factor
        
        # Adjust based on purpose if provided
        if request and 'travel_purpose' in request:
            purpose = request['travel_purpose']
            
            # Business/work trips might have higher willingness to pay
            if purpose == 0:  # work
                base_max *= 1.2
            # Medical trips might have higher urgency
            elif purpose == 3:  # medical
                base_max *= 1.3
        
        # Adjust based on time flexibility
        if request and 'flexible_time' in request:
            if request['flexible_time'] == 'low':
                base_max *= 1.2  # Less flexible = willing to pay more
            elif request['flexible_time'] == 'high':
                base_max *= 0.8  # More flexible = willing to pay less
        
        # Add some randomness
        base_max *= random.uniform(0.9, 1.1)
        
        return base_max
    

    def evaluate_bundle_options(self, bundle_id):
        """
        Evaluate different bundle options.
        
        Args:
            bundle_id: Bundle request ID
            
        Returns:
            List of (utility, option) tuples, sorted by utility
        """
        if not hasattr(self, 'bundle_requests') or bundle_id not in self.bundle_requests:
            self.logger.error(f"Bundle request {bundle_id} not found")
            return []
        
        bundle_request = self.bundle_requests[bundle_id]
        
        # Generate all possible combinations of offers
        bundle_options = []
        
        # This is a simplified approach - in a full implementation, 
        # you would need to handle combinatorial complexity with heuristics
        
        # For each segment, select the best offer based on utility
        selected_offers = {}
        
        for segment_id, offers in bundle_request['components'].items():
            best_offer = None
            best_utility = float('-inf')
            
            for offer in offers:
                # Calculate utility for this offer
                utility = self._calculate_bundle_offer_utility(offer, segment_id)
                
                if utility > best_utility:
                    best_utility = utility
                    best_offer = offer
            
            if best_offer:
                selected_offers[segment_id] = best_offer
        
        # Check if we have offers for all segments
        if len(selected_offers) == len(bundle_request['segments']):
            # Calculate total price and utility
            total_price = sum(offer['price'] for offer in selected_offers.values())
            
            # Apply bundle discount
            discounted_price = total_price * 0.95  # 5% discount for the whole bundle
            
            total_utility = sum(self._calculate_bundle_offer_utility(offer, segment_id) 
                            for segment_id, offer in selected_offers.items())
            
            # Create bundle option
            bundle_option = {
                'bundle_id': bundle_id,
                'selected_offers': selected_offers,
                'total_price': discounted_price,
                'total_utility': total_utility,
                'components': {
                    segment_id: {
                        'provider_id': offer['provider_id'],
                        'price': offer['price'],
                        'start_time': offer['start_time'],
                        'duration': offer['estimated_time'],
                        'route': offer['route'],
                        'mode': offer['mode'],
                        'offer_signature': offer.get('signature')
                    }
                    for segment_id, offer in selected_offers.items()
                }
            }
            
            bundle_options.append((total_utility, bundle_option))
        
        # Sort by utility (highest first)
        bundle_options.sort(reverse=True, key=lambda x: x[0])
        
        return bundle_options

    def _calculate_bundle_offer_utility(self, offer, segment_id):
        """
        Calculate utility of a bundle component offer.
        
        Args:
            offer: Offer for a bundle component
            segment_id: Segment ID
            
        Returns:
            Utility value
        """
        # Extract offer details
        price = offer['price']
        duration = offer['estimated_time']
        mode = offer['mode']
        reliability = offer.get('reliability', 0.7)
        comfort = offer.get('quality_score', 70) / 100
        
        # Basic utility calculation
        price_utility = self.utility_coefficients['price'] * price * self.price_sensitivity
        time_utility = self.utility_coefficients['time'] * duration
        comfort_utility = self.utility_coefficients['comfort'] * comfort * self.comfort_preference
        reliability_utility = self.utility_coefficients['reliability'] * reliability * self.reliability_preference
        
        # Mode preference
        mode_utility = 0
        if mode in self.mode_preference:
            mode_utility = self.mode_preference[mode] * 0.5
        
        # Calculate total utility
        total_utility = (
            price_utility + 
            time_utility + 
            comfort_utility + 
            reliability_utility + 
            mode_utility
        )
        
        # Small random component for variety
        total_utility += random.uniform(-0.05, 0.05)
        
        return total_utility

    def purchase_bundle(self, bundle_id):
        """
        Purchase a selected bundle.
        
        Args:
            bundle_id: Bundle ID to purchase
            
        Returns:
            Success status
        """
        if not hasattr(self, 'bundle_requests') or bundle_id not in self.bundle_requests:
            self.logger.error(f"Bundle request {bundle_id} not found")
            return False
        
        bundle_request = self.bundle_requests[bundle_id]
        
        # Get bundle options
        bundle_options = self.evaluate_bundle_options(bundle_id)
        
        if not bundle_options:
            self.logger.error(f"No viable options for bundle {bundle_id}")
            return False
        
        # Select best option
        best_utility, best_option = bundle_options[0]
        
        # Execute bundle purchase via blockchain
        success, bundle_id = self.blockchain_interface.execute_bundle_purchase(
            best_option, self.unique_id)
        
        if success:
            self.logger.info(f"Successfully purchased bundle {bundle_id}")
            
            # Update bundle status
            bundle_request['status'] = 'purchased'
            
            # Add bundle to active trips
            if not hasattr(self, 'active_bundles'):
                self.active_bundles = {}
            
            self.active_bundles[bundle_id] = {
                'bundle_id': bundle_id,
                'components': best_option['components'],
                'total_price': best_option['total_price'],
                'purchase_time': self.model.schedule.time,
                'status': 'active'
            }
            
            return True
        else:
            self.logger.error(f"Failed to purchase bundle {bundle_id}")
            return False
    def select_and_purchase_option(self, request_id, strategy=None):
        """
        Select and purchase the best mobility option.
        
        Args:
            request_id: The request ID
            strategy: Optional strategy to use ('direct_booking', 'market_purchase', 'bundled_service')
            
        Returns:
            True if successful, False otherwise
        """
        if request_id not in self.requests:
            self.logger.error(f"Request {request_id} not found")
            return False
        
        # Select strategy if not provided based on weights
        if not strategy:
            strategies = list(self.strategy_weights.keys())
            weights = [self.strategy_weights[s] for s in strategies]
            strategy = random.choices(strategies, weights=weights, k=1)[0]
        
        self.logger.info(f"Using strategy: {strategy} for request {request_id}")
        self.requests[request_id]['selected_strategy'] = strategy
        
        # Get ranked options
        ranked_options = self.evaluate_marketplace_options(request_id)
        
        if not ranked_options:
            self.logger.warning(f"No options found for request {request_id}")
            return False
        
        # Filter options based on strategy
        filtered_options = []
        if strategy == 'direct_booking':
            filtered_options = [(u, opt) for u, opt in ranked_options if opt['type'] == 'direct_booking']
        elif strategy == 'market_purchase':
            filtered_options = [(u, opt) for u, opt in ranked_options if opt['type'] == 'nft_market']
        elif strategy == 'bundled_service':
            # For bundled service, we would need to get bundle options
            # This is a placeholder - implementation would depend on how bundles are represented
            # For now, just use all options
            filtered_options = ranked_options
        
        # If no options match the strategy, fall back to all options
        if not filtered_options:
            self.logger.warning(f"No options match strategy {strategy}, using all options")
            filtered_options = ranked_options
        
        # Select best option (highest utility)
        best_utility, best_option = filtered_options[0]
        
        self.logger.info(f"Selected option: {best_option['type']} with utility {best_utility}")
        self.requests[request_id]['selected_option'] = best_option
        
        # Execute purchase based on option type
        success = False
        if best_option['type'] == 'direct_booking':
            # Call the appropriate provider method to book directly
            for provider in self.model.schedule.agents:
                if provider.unique_id == best_option['provider_id'] and hasattr(provider, 'accept_booking'):
                    success = provider.accept_booking(
                        self.unique_id,
                        request_id,
                        best_option['price'],
                        best_option.get('departure_time', self.requests[request_id]['start_time']),
                        best_option.get('route', [])
                    )
                    break
        elif best_option['type'] == 'nft_market':
            # Purchase from NFT marketplace
            success = self.blockchain_interface.purchase_nft(best_option['nft_id'], self.unique_id)
            
            if success:
                # Record the NFT as owned
                self.owned_nfts[best_option['nft_id']] = {
                    'request_id': request_id,
                    'price': best_option['price'],
                    'service_time': best_option.get('service_time', self.requests[request_id]['start_time']),
                    'duration': best_option.get('time', 0),
                    'provider_id': best_option.get('provider_id', 0),
                    'mode': best_option.get('mode', 'unknown'),
                    'route': best_option.get('route', []),
                    'status': 'active',
                    'purchase_time': self.model.schedule.time
                }
        
        # Update request status
        if success:
            self.requests[request_id]['status'] = 'service_selected'
            self.requests[request_id]['blockchain_status'] = 'confirmed'
            
            # Add to active trips
            self.active_trips[request_id] = {
                'request': self.requests[request_id],
                'option': best_option,
                'start_time': best_option.get('departure_time', self.requests[request_id]['start_time']),
                'status': 'booked'
            }
            
            # Remove from pending requests
            if request_id in self.pending_requests:
                self.pending_requests.remove(request_id)
            
            # Update experience with this provider
            provider_id = best_option.get('provider_id', None)
            if provider_id:
                if provider_id not in self.market_experience:
                    self.market_experience[provider_id] = 0
                # Small positive update for successful booking
                self.market_experience[provider_id] += 0.05
            
            self.logger.info(f"Successfully purchased option for request {request_id}")
        else:
            self.logger.warning(f"Failed to purchase option for request {request_id}")
        
        return success

    # def evaluate_owned_nfts_for_resale(self):
    #     """
    #     Evaluate each owned NFT to decide whether to keep or sell.
    #     """
    #     current_time = self.model.schedule.time
        
    #     # Evaluate each owned NFT
    #     for nft_id, nft_details in list(self.owned_nfts.items()):
    #         # Skip if service already used or sold
    #         if nft_details['status'] != 'active':
    #             continue
            
    #         # Skip if service time has passed
    #         if nft_details['service_time'] < current_time:
    #             # Mark as expired
    #             nft_details['status'] = 'expired'
    #             self.logger.info(f"NFT {nft_id} has expired")
    #             continue
            
    #         # Calculate continued utility value (CUV)
    #         cuv = self._calculate_continued_utility(nft_id)
            
    #         # Estimate current market value
    #         market_value = self._estimate_market_value(nft_id)
            
    #         # Decision threshold (including transaction costs)
    #         threshold = 0.2  # 20% gain threshold to account for fees
            
    #         self.logger.debug(f"NFT {nft_id} - CUV: {cuv}, Market value: {market_value}")
            
    #         # If market value exceeds utility by threshold, list for sale
    #         if market_value > cuv * (1 + threshold):
    #             # Calculate optimal listing price
    #             listing_price = market_value * 0.95  # Slight discount for quicker sale
                
    #             # Decide on dynamic pricing parameters
    #             time_to_service = nft_details['service_time'] - current_time
                
    #             # Longer time to service = more aggressive price decay
    #             if time_to_service > 24 * 3600:  # More than 24 hours
    #                 decay_rate = 0.1  # Faster decay
    #                 min_price = cuv * 0.8  # Lower minimum price
    #             else:
    #                 decay_rate = 0.05  # Slower decay
    #                 min_price = cuv * 0.9  # Higher minimum price
                
    #             # List for sale with dynamic pricing
    #             time_parameters = {
    #                 'initial_price': listing_price,
    #                 'final_price': min_price,
    #                 'decay_duration': int(time_to_service * 0.7)  # Use 70% of remaining time
    #             }
                
    #             success = self.blockchain_interface.list_nft_for_sale(nft_id, listing_price, time_parameters)
                
    #             if success:
    #                 # Update NFT status
    #                 nft_details['status'] = 'listed'
    #                 self.logger.info(f"Listed NFT {nft_id} for sale at {listing_price}")
    #             else:
    #                 self.logger.warning(f"Failed to list NFT {nft_id} for sale")

    # def _calculate_continued_utility(self, nft_id):
    #     """
    #     Calculate the utility of keeping and using the NFT.
        
    #     Args:
    #         nft_id: The NFT ID
            
    #     Returns:
    #         Utility value
    #     """
    #     nft = self.owned_nfts[nft_id]
    #     current_time = self.model.schedule.time
        
    #     # Base utility calculation
    #     base_utility = -1 * (
    #         self.utility_coefficients['price'] * nft['price'] +
    #         self.utility_coefficients['time'] * nft['duration']
    #     )
        
    #     # Adjust for time proximity
    #     time_to_service = nft['service_time'] - current_time
        
    #     # If service time is very close, utility increases (harder to replace)
    #     if time_to_service < 3600:  # Within 1 hour
    #         urgency_factor = 2.0 - (time_to_service / 3600)  # From 1.0 to 2.0
    #         base_utility *= urgency_factor
    #     # If somewhat close, still increase utility
    #     elif time_to_service < 24 * 3600:  # Within 24 hours
    #         urgency_factor = 1.0 + (24 * 3600 - time_to_service) / (24 * 3600)
    #         base_utility *= urgency_factor
    #     # If very far in future, utility might decrease (easier to replace)
    #     elif time_to_service > 7 * 24 * 3600:  # More than 7 days away
    #         flexibility_factor = 0.8
    #         base_utility *= flexibility_factor
        
    #     # Adjust for upcoming needs
    #     # Check if we have upcoming requests that might need this service
    #     for req_id, req in self.requests.items():
    #         if req['status'] == 'active' and req_id != nft.get('request_id'):
    #             req_origin = req['origin']
    #             req_dest = req['destination']
    #             req_time = req['start_time']
                
    #             # Check if NFT route is similar
    #             route_match = False
    #             nft_route = nft.get('route', [])
                
    #             if nft_route:
    #                 # Simple check: does route start near request origin and end near request destination?
    #                 if (len(nft_route) >= 2 and
    #                     self._calculate_distance(nft_route[0], req_origin) < 10 and
    #                     self._calculate_distance(nft_route[-1], req_dest) < 10):
    #                     route_match = True
                
    #             # Check if time is close
    #             time_match = abs(nft['service_time'] - req_time) < 3600  # Within 1 hour
                
    #             if route_match and time_match:
    #                 # This NFT could be useful for an upcoming request
    #                 base_utility *= 1.5
    #                 break
        
    #     return base_utility

    # def _estimate_market_value(self, nft_id):
    #     """
    #     Estimate the current market value of an NFT.
        
    #     Args:
    #         nft_id: The NFT ID
            
    #     Returns:
    #         Estimated market value
    #     """
    #     nft = self.owned_nfts[nft_id]
    #     current_time = self.model.schedule.time
        
    #     # Original price as baseline
    #     base_price = nft['price']
        
    #     # Time-based adjustment
    #     time_to_service = nft['service_time'] - current_time
        
    #     if time_to_service < 3600:  # Within 1 hour
    #         # Price drops rapidly near service time (less than 20% of original value)
    #         time_factor = max(0.2, time_to_service / 3600)
    #     elif time_to_service < 24 * 3600:  # Within 24 hours
    #         # Linear decrease from 80% to 60% value
    #         time_factor = 0.6 + (0.2 * time_to_service / (24 * 3600))
    #     elif time_to_service < 7 * 24 * 3600:  # Within 7 days
    #         # Stable pricing in medium range (80% of value)
    #         time_factor = 0.8
    #     else:  # Far future
    #         # Premium for advance booking (up to 120% of value)
    #         time_factor = min(1.2, 0.8 + (time_to_service - 7 * 24 * 3600) / (30 * 24 * 3600))
        
    #     # Market demand adjustment based on similar recent transactions
    #     # For simplicity, use a random factor, but in a real implementation,
    #     # would check actual market demand
    #     demand_factor = random.uniform(0.9, 1.1)
        
    #     estimated_value = base_price * time_factor * demand_factor
        
    #     return estimated_value

    def update(self):
        """
        Main update method called on each simulation step.
        """
        # Process pending blockchain operations
        self._update_request_status()
        
        # Process active trips
        self._update_active_trips()
        
        # Evaluate owned NFTs for potential resale
        # self.evaluate_owned_nfts_for_resale()
        
        # Check pending requests and take action if needed
        self._process_pending_requests()        
        
        # Generate new trips if needed (optional, based on model design)
        self._generate_new_trips()

    def _update_request_status(self):
        """
        Update local status based on blockchain status.
        """
        for request_id, request in self.requests.items():
            if request['blockchain_status'] == 'pending':
                # Check blockchain status
                updated_status = self.blockchain_interface.check_request_status(request_id)
                if updated_status:
                    request['blockchain_status'] = updated_status
                    self.logger.info(f"Request {request_id} blockchain status updated to {updated_status}")

    def _update_active_trips(self):
        """
        Update the status of active trips.
        """
        current_time = self.model.schedule.time
        
        # Check each active trip
        for trip_id, trip in list(self.active_trips.items()):
            # Skip if trip is already completed
            if trip['status'] == 'completed':
                continue
            
            # Check if trip start time has passed
            if current_time >= trip['start_time']:
                # If trip was only booked, mark as in progress
                if trip['status'] == 'booked':
                    trip['status'] = 'in_progress'
                    self.logger.info(f"Trip {trip_id} is now in progress")
                
                # Check if trip should be completed
                if trip['status'] == 'in_progress':
                    # Calculate expected end time
                    option = trip['option']
                    duration = option.get('time', 1800)  # Default 30 minutes
                    
                    if current_time >= trip['start_time'] + duration:
                        # Trip is complete
                        trip['status'] = 'completed'
                        self.logger.info(f"Trip {trip_id} completed")
                        
                        # Add to trip history
                        self.trip_history.append({
                            'trip_id': trip_id,
                            'request': trip['request'],
                            'option': option,
                            'start_time': trip['start_time'],
                            'end_time': current_time,
                            'duration': current_time - trip['start_time'],
                            'satisfaction': self._calculate_trip_satisfaction(trip)
                        })
                        
                        # Update experience with provider
                        provider_id = option.get('provider_id')
                        if provider_id:
                            satisfaction = self._calculate_trip_satisfaction(trip)
                            self._update_provider_experience(provider_id, satisfaction)
                        
                        # If this trip was associated with an NFT, mark it as used
                        if option['type'] == 'nft_market' and 'nft_id' in option:
                            nft_id = option['nft_id']
                            if nft_id in self.owned_nfts:
                                self.owned_nfts[nft_id]['status'] = 'used'
                                self.logger.info(f"NFT {nft_id} marked as used")

    def _calculate_trip_satisfaction(self, trip):
        """
        Calculate satisfaction with a completed trip.
        
        Args:
            trip: Trip data
            
        Returns:
            Satisfaction score between -1 and 1
        """
        # Base satisfaction is neutral
        satisfaction = 0.0
        
        option = trip['option']
        request = trip['request']
        
        # Factors that affect satisfaction
        
        # 1. Timeliness - was the service on time?
        expected_start = request['start_time']
        actual_start = trip['start_time']
        time_diff = abs(actual_start - expected_start)
        
        if time_diff < 300:  # Within 5 minutes
            timeliness = 0.2
        elif time_diff < 900:  # Within 15 minutes
            timeliness = 0.1
        elif time_diff < 1800:  # Within 30 minutes
            timeliness = 0
        else:  # More than 30 minutes
            timeliness = -0.2
        
        # 2. Price - was it good value?
        price = option['price']
        max_price = self._calculate_max_price(request)
        
        if price < 0.5 * max_price:
            price_satisfaction = 0.2
        elif price < 0.8 * max_price:
            price_satisfaction = 0.1
        elif price < max_price:
            price_satisfaction = 0
        else:
            price_satisfaction = -0.1
        
        # 3. Comfort and quality (from option data)
        comfort = option.get('comfort', 0.5)
        comfort_satisfaction = (comfort - 0.5) * 0.4  # Scale from -0.2 to 0.2
        
        # 4. Mode preference
        mode = option.get('mode', 'car')
        mode_preference = self.mode_preference.get(mode, 0.2)
        mode_satisfaction = (mode_preference - 0.2) * 0.5  # Scale from -0.1 to 0.3
        
        # Combine factors
        satisfaction = timeliness + price_satisfaction + comfort_satisfaction + mode_satisfaction
        
        # Add random component for variation
        satisfaction += random.uniform(-0.1, 0.1)
        
        # Clamp to [-1, 1] range
        satisfaction = max(-1, min(1, satisfaction))
        
        return satisfaction

    def _update_provider_experience(self, provider_id, satisfaction):
        """
        Update experience with a provider based on trip satisfaction.
        
        Args:
            provider_id: Provider ID
            satisfaction: Satisfaction score between -1 and 1
        """
        if provider_id not in self.market_experience:
            self.market_experience[provider_id] = 0
        
        # Map satisfaction [-1, 1] to experience update [-0.2, 0.2]
        experience_update = satisfaction * 0.2
        
        # Apply update with some damping (80% new, 20% old)
        self.market_experience[provider_id] = (
            0.8 * self.market_experience[provider_id] + 
            0.2 * experience_update
        )
        
        # Ensure value is between -1 and 1
        self.market_experience[provider_id] = max(-1, min(1, self.market_experience[provider_id]))
        
        self.logger.debug(f"Updated experience with provider {provider_id} to {self.market_experience[provider_id]}")

    def _process_pending_requests(self):
        """
        Process pending requests and take action if needed.
        """
        # Check if registration is confirmed
        if self.blockchain_interface.is_commuter_registered(self.unique_id):
            # We're registered! Process pending requests
            requests_to_process = self.pending_outgoing_requests.copy()
            self.pending_outgoing_requests = []
            
            for req in requests_to_process:
                self.logger.info(f"Processing queued request for commuter {self.unique_id}")
                self.create_request(
                    req['origin'],
                    req['destination'],
                    req['start_time'],
                    req['travel_purpose'],
                    req['requirements']
                )
        # Make a copy to avoid modifying during iteration
        for request_id in list(self.pending_requests):
            if request_id not in self.requests:
                # Request has been removed, remove from pending
                self.pending_requests.remove(request_id)
                continue
            
            request = self.requests[request_id]
            
            # If blockchain status is confirmed, can proceed with selection
            if request['blockchain_status'] == 'confirmed' and request['status'] == 'active':
                self.logger.info(f"Processing pending request {request_id}")
                
                # Attempt to select and purchase an option
                success = self.select_and_purchase_option(request_id)
                
                if success:
                    self.logger.info(f"Successfully processed request {request_id}")
                else:
                    self.logger.warning(f"Failed to process request {request_id}")
                    
                    # If failed, keep in pending list for retry later
                    # Add a small random delay before retry
                    if random.random() < 0.3:  # 30% chance to retry immediately
                        pass  # Leave in pending list for next update
                    else:
                        # Remove from pending for now, will check again later
                        self.pending_requests.remove(request_id)

    

    def _generate_new_trips(self):
        """
        Generate new trips based on commuter's travel patterns.
        This is model-specific and depends on how the simulation is structured.
        """
        # This is a placeholder - implementation depends on the model design
        # In many models, trip generation would be controlled at the model level
        # rather than by individual agents
        pass

    def _calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point [x, y]
            point2: Second point [x, y]
            
        Returns:
            Euclidean distance
        """
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    def get_position(self):
        """
        Get current position for visualization.
        
        Returns:
            Current position (x, y)
        """
        # If on an active trip, position might be along the route
        for trip_id, trip in self.active_trips.items():
            if trip['status'] == 'in_progress':
                # Calculate position along route
                route = trip['option'].get('route', [])
                if len(route) >= 2:
                    start_time = trip['start_time']
                    duration = trip['option'].get('time', 1800)  # Default 30 minutes
                    current_time = self.model.schedule.time
                    
                    # Calculate progress along route (0 to 1)
                    progress = min(1.0, (current_time - start_time) / duration)
                    
                    # Interpolate position along route
                    if progress <= 0:
                        return route[0]
                    elif progress >= 1:
                        return route[-1]
                    else:
                        # Find appropriate segment
                        segment_count = len(route) - 1
                        segment_idx = min(int(progress * segment_count), segment_count - 1)
                        segment_progress = (progress * segment_count) - segment_idx
                        
                        # Interpolate within segment
                        start = route[segment_idx]
                        end = route[segment_idx + 1]
                        
                        return [
                            start[0] + segment_progress * (end[0] - start[0]),
                            start[1] + segment_progress * (end[1] - start[1])
                        ]
        
        # If not on a trip, return home location
        return self.location
    