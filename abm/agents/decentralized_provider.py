"""
Decentralized Provider Agent for Blockchain-based MaaS System

This module implements truly decentralized service providers in a Mobility-as-a-Service system,
featuring blockchain integration, NFT-based tokenization, and realistic agent behavior.
It supports both individual service providers and fractional fleet ownership models.
"""

from mesa import Agent
import numpy as np
import math
import uuid
import random
import logging
from datetime import datetime, timedelta


class DecentralizedProvider(Agent):
    """
    A service provider agent with blockchain identity representing either:
    1. An individual mobility service provider (driver, bike owner)
    2. A fractionally-owned fleet (community-owned public transport)
    
    This agent implements realistic decision-making based on personal preferences,
    availability, and dynamic market conditions.
    """
    def __init__(self, unique_id, model, name, mode_type, base_price,
                 capacity, service_area, blockchain_interface, quality_score=70,
                 reliability=70, response_time=10, provider_type="individual",
                 fleet_token_holders=None):
        """
        Initialize a new decentralized provider agent.
        
        Args:
            unique_id: Unique identifier for the agent
            model: The model containing the agent
            name: Name of the provider (individual name or fleet name)
            mode_type: Transport mode ('car', 'bike', 'bus', 'train', etc.)
            base_price: Base price per unit distance
            capacity: Maximum service capacity (typically 1 for individuals)
            service_area: Service coverage radius
            blockchain_interface: Interface to the blockchain
            quality_score: Initial quality score (0-100)
            reliability: Initial reliability score (0-100)
            response_time: Average response time in minutes
            provider_type: "individual" or "fractional_fleet"
            fleet_token_holders: For fractional fleets, list of token holder IDs and their stakes
        """
        super().__init__(unique_id, model)
        # Basic attributes
        self.name = name
        self.mode_type = mode_type
        self.base_price = base_price
        self.capacity = capacity
        self.available_capacity = capacity
        self.service_area = service_area
        self.quality_score = quality_score
        self.reliability = reliability
        self.response_time = response_time
        self.is_verified = False
        self.is_active = True
        self.provider_type = provider_type
        
        # Physical attributes
        self.home_location = self._generate_home_location()
        self.current_location = self.home_location.copy()
        self.service_center = self.home_location.copy()
        self.target_location = None
        self.is_moving = False
        self.current_route = []
        self.speed = self._get_mode_speed()
        
        # Personal preferences and constraints
        self.working_hours = self._generate_working_schedule()
        self.location_preferences = self._generate_location_preferences()
        self.min_trip_value = self._calculate_min_trip_value()
        self.preferred_customers = set()  # IDs of preferred customers
        self.blacklisted_customers = set()  # IDs of problematic customers
        
        # Fractional fleet attributes
        self.fleet_token_holders = fleet_token_holders if fleet_token_holders else {}
        self.governance_votes = {}  # Tracks governance proposals and votes
        self.route_flexibility = 0.2 if self.provider_type == "fractional_fleet" else 0.7
        
        # Blockchain-specific attributes
        self.blockchain_interface = blockchain_interface
        self.blockchain_address = None
        self.issued_nfts = {}  # NFTs issued by this provider
        self.active_bids = {}  # Currently active offers/bids
        self.service_history = []  # History of completed services
        self.transaction_history = []  # History of blockchain transactions
        
        # Pricing parameters - now more personalized
        self.price_strategy = self._choose_pricing_strategy()
        self.price_adjustment_factor = random.uniform(0.15, 0.25)  # Variation between providers
        self.time_decay_factor = random.uniform(0.1, 0.2)  # Variation in time decay
        self.dynamic_pricing_enabled = True
        self.price_floor = random.uniform(0.5, 0.7)  # Individualized price floor
        self.price_ceiling = random.uniform(1.8, 2.2)  # Individualized price ceiling
        self.price_history = []
        self.demand_history = []
        self.surge_sensitivity = random.uniform(0.8, 1.2)  # How much provider responds to surge
        
        # Learning parameters
        self.competitor_price_history = {}
        self.win_rate_history = []
        self.customer_satisfaction = {}  # Customer ID -> satisfaction scores
        self.reputation_score = 70  # Starting reputation score (0-100)
        self.reputation_history = [(model.schedule.time, 70)]  # Track reputation over time
        
        # Set up logging
        self.logger = logging.getLogger(f"Provider-{unique_id}")
        self.logger.setLevel(logging.INFO)
        
        # Register with blockchain
        self._register_with_blockchain()

    def _generate_home_location(self):
        """Generate home/base location for the provider"""
        # For individual providers, this would be their home
        # For fleets, this would be a depot or hub
        grid_width = self.model.grid_width
        grid_height = self.model.grid_height
        
        if self.provider_type == "fractional_fleet" and self.mode_type in ["bus", "train"]:
            # Public transport typically based in central locations
            center_x = grid_width // 2
            center_y = grid_height // 2
            # Distribute around center
            home_x = int(center_x + random.uniform(-0.2, 0.2) * grid_width)
            home_y = int(center_y + random.uniform(-0.2, 0.2) * grid_height)
        else:
            # Individual providers more randomly distributed
            if random.random() < 0.7:  # 70% in residential areas (outer regions)
                # Generate in the outer regions of the grid
                edge_distance = min(grid_width, grid_height) * 0.3
                quadrant = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
                
                if quadrant[0] == 0:
                    home_x = random.randint(0, int(edge_distance))
                else:
                    home_x = random.randint(int(grid_width - edge_distance), grid_width - 1)
                    
                if quadrant[1] == 0:
                    home_y = random.randint(0, int(edge_distance))
                else:
                    home_y = random.randint(int(grid_height - edge_distance), grid_height - 1)
            else:  # 30% more centrally located
                home_x = random.randint(int(grid_width * 0.3), int(grid_width * 0.7))
                home_y = random.randint(int(grid_height * 0.3), int(grid_height * 0.7))
                
        return [home_x, home_y]

    def _generate_working_schedule(self):
        """Generate personal working hours schedule"""
        # This represents when the provider is willing to work
        working_hours = {}
        
        # Days of week (0=Monday, 6=Sunday)
        for day in range(7):
            if self.provider_type == "fractional_fleet":
                # Public transport usually operates all day
                if day < 5:  # Weekday
                    working_hours[day] = [(6, 23)]  # 6am to 11pm
                else:  # Weekend
                    working_hours[day] = [(7, 22)]  # 7am to 10pm
            else:
                # Individual providers have more varied schedules
                if random.random() < 0.8:  # 80% work on weekdays
                    if day < 5:
                        # Choose either morning shift, evening shift, or full day
                        shift_type = random.choice(["morning", "evening", "full", "afternoon"])
                        if shift_type == "morning":
                            working_hours[day] = [(6, 12)]  # 6am to 12pm
                        elif shift_type == "evening":
                            working_hours[day] = [(16, 23)]  # 4pm to 11pm
                        elif shift_type == "afternoon":
                            working_hours[day] = [(12, 18)]  # 12pm to 6pm
                        else:  # full
                            working_hours[day] = [(8, 18)]  # 8am to 6pm
                
                # Weekend availability
                if day >= 5:
                    if random.random() < 0.5:  # 50% work on weekends
                        # Weekend shifts tend to be flexible
                        start_hour = random.randint(7, 12)
                        end_hour = random.randint(16, 22)
                        working_hours[day] = [(start_hour, end_hour)]
        
        return working_hours

    def _generate_location_preferences(self):
        """Generate provider preferences for service locations"""
        preferences = {
            "preferred_regions": [],
            "avoid_regions": [],
            "max_distance_from_home": None
        }
        
        grid_width = self.model.grid_width
        grid_height = self.model.grid_height
        
        # Divide grid into regions
        regions = []
        region_size = min(grid_width, grid_height) // 3
        for x in range(0, grid_width, region_size):
            for y in range(0, grid_height, region_size):
                region = {
                    "center": [x + region_size//2, y + region_size//2],
                    "radius": region_size//2
                }
                regions.append(region)
        
        # Select preferred and avoided regions
        if self.provider_type == "individual":
            # Individual providers have stronger preferences
            num_preferred = random.randint(1, min(3, len(regions)))
            preferences["preferred_regions"] = random.sample(regions, num_preferred)
            
            # Avoid 1-2 regions
            remaining_regions = [r for r in regions if r not in preferences["preferred_regions"]]
            if remaining_regions:
                num_avoid = random.randint(0, min(2, len(remaining_regions)))
                preferences["avoid_regions"] = random.sample(remaining_regions, num_avoid)
                
            # Set max distance from home based on mode
            if self.mode_type == "bike":
                preferences["max_distance_from_home"] = grid_width * 0.3
            elif self.mode_type == "car":
                preferences["max_distance_from_home"] = grid_width * 0.7
        else:
            # Fractional fleets have routes rather than preferences
            if self.mode_type in ["bus", "train"]:
                # Create a route corridor
                start_region = random.choice(regions)
                far_regions = [r for r in regions if 
                              self._calculate_distance(start_region["center"], r["center"]) > region_size * 2]
                
                if far_regions:
                    end_region = random.choice(far_regions)
                    preferences["route_corridor"] = {
                        "start": start_region["center"],
                        "end": end_region["center"],
                        "width": region_size * self.route_flexibility
                    }
                    
                    # Preferred regions along the route
                    for region in regions:
                        # Check if region is near the route corridor
                        if self._point_to_line_distance(
                            region["center"], 
                            start_region["center"], 
                            end_region["center"]) < region_size:
                            preferences["preferred_regions"].append(region)
        
        return preferences

    def _point_to_line_distance(self, point, line_start, line_end):
        """Calculate shortest distance from point to line segment"""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Line vector
        dx = x2 - x1
        dy = y2 - y1
        
        # Line length squared
        line_length_sq = dx*dx + dy*dy
        
        # If line is a point, return distance to the point
        if line_length_sq == 0:
            return math.sqrt((x - x1)**2 + (y - y1)**2)
        
        # Calculate projection of point onto line
        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / line_length_sq))
        
        # Closest point on line
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        # Distance to closest point
        return math.sqrt((x - proj_x)**2 + (y - proj_y)**2)

    def _calculate_min_trip_value(self):
        """Calculate minimum acceptable trip value based on provider attributes"""
        # This represents the minimum price the provider will accept for a trip
        if self.provider_type == "individual":
            # Based on opportunity cost of time and operational costs
            base_cost = self.base_price * 5  # Minimum distance equivalent
            time_value = (100 - self.capacity) * 0.1  # Providers with less capacity value their time more
            
            # Mode-specific costs
            mode_cost_factor = {
                "car": 1.2,  # Higher operational costs
                "bike": 0.8,  # Lower operational costs
                "bus": 1.5,
                "train": 2.0
            }
            
            cost_factor = mode_cost_factor.get(self.mode_type, 1.0)
            return base_cost * cost_factor + time_value
        else:
            # Fractional fleets have fixed costs to cover
            return self.base_price * 10 / self.capacity  # Per-passenger minimum

    def _choose_pricing_strategy(self):
        """Choose a personalized pricing strategy"""
        strategies = [
            "value_based",      # Charge based on perceived value to customer
            "cost_plus",        # Base cost plus markup
            "competitor_based", # Set prices relative to competitors
            "dynamic_surge",    # Highly responsive to demand fluctuations
            "time_optimized",   # Optimize for time of day
            "loyalty_focused"   # Lower prices for repeat customers
        ]
        
        # Different provider types tend toward different strategies
        if self.provider_type == "individual":
            if self.mode_type == "car":
                weights = [0.2, 0.1, 0.3, 0.3, 0.05, 0.05]
            elif self.mode_type == "bike":
                weights = [0.1, 0.3, 0.2, 0.1, 0.2, 0.1]
            else:
                weights = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
        else:  # fractional_fleet
            if self.mode_type == "bus":
                weights = [0.05, 0.3, 0.05, 0.1, 0.3, 0.2]
            elif self.mode_type == "train":
                weights = [0.05, 0.4, 0.05, 0.0, 0.3, 0.2]
            else:
                weights = [0.1, 0.3, 0.1, 0.1, 0.3, 0.1]
                
        return random.choices(strategies, weights=weights)[0]

    def _get_mode_speed(self):
        """Get speed based on transport mode"""
        speed_map = {
            'car': 6.0,
            'bike': 3.0,
            'bus': 2.5,
            'train': 8.0,
            'walk': 1.0
        }
        
        base_speed = speed_map.get(self.mode_type, 4.0)
        
        # Add some individual variation (±20%)
        individual_factor = random.uniform(0.8, 1.2)
        return base_speed * individual_factor

    def get_service_offers(self, origin, destination, start_time, mode_filter=None, max_offers=3):
        """
        Generate service offers for a specific route and time based on personal preferences.
        
        Args:
            origin: Origin coordinates [x, y]
            destination: Destination coordinates [x, y]
            start_time: Requested start time
            mode_filter: Optional filter for transport mode
            max_offers: Maximum number of offers to generate
            
        Returns:
            List of offer dictionaries
        """
        # Check service availability based on personal constraints
        if not self._is_available_at_time(start_time):
            return []
            
        # Check if mode matches filter
        if mode_filter and self.mode_type != mode_filter:
            return []
            
        # Check if provider is physically capable of servicing the route
        if not self._can_service_route(origin, destination):
            return []
            
        # Check if route matches location preferences
        if not self._matches_location_preferences(origin, destination):
            self.logger.debug(f"Provider {self.unique_id} skipping request due to location preferences")
            return []
            
        # Check if capacity is available
        if self.available_capacity <= 0:
            self.logger.info(f"No capacity available for provider {self.unique_id}")
            return []
            
        # Calculate base price based on personal pricing strategy
        base_price = self._calculate_personalized_price({
            'origin': origin, 
            'destination': destination, 
            'start_time': start_time
        })
        
        # Check if trip value meets minimum threshold
        route_distance = self._calculate_distance(origin, destination)
        trip_value = base_price * route_distance
        if trip_value < self._calculate_min_trip_value():
            self.logger.debug(f"Trip value {trip_value} below minimum threshold for provider {self.unique_id}")
            return []
            
        # Generate primary offer
        offers = []
        primary_offer = {
            'provider_id': self.unique_id,
            'provider_name': self.name,
            'price': base_price,
            'mode': self.mode_type,
            'departure_time': start_time,
            'duration': self._calculate_estimated_time({'origin': origin, 'destination': destination, 'start_time': start_time}),
            'route': self._generate_route(origin, destination),
            'comfort': min(95, self.quality_score + random.randint(-5, 5)),
            'reliability': self.reliability,
            'provider_type': self.provider_type,
            'reputation': self.reputation_score
        }
        offers.append(primary_offer)
        
        # For individual providers, generate personalized variants
        if self.provider_type == "individual" and max_offers > 1:
            # Create variants based on personal strategy
            if self.price_strategy == "time_optimized":
                # Offer different time slots with price variations
                for time_shift in [-900, 900]:  # ±15 minutes
                    alt_time = start_time + time_shift
                    if self._is_available_at_time(alt_time):
                        # Adjust price based on preferred times
                        time_factor = 1.0
                        if self._is_preferred_time(alt_time):
                            time_factor = 0.95  # Discount for preferred times
                        else:
                            time_factor = 1.05  # Premium for less preferred times
                            
                        alt_offer = primary_offer.copy()
                        alt_offer['departure_time'] = alt_time
                        alt_offer['price'] = primary_offer['price'] * time_factor
                        offers.append(alt_offer)
            
            elif self.price_strategy == "value_based":
                # Offer premium and economy variants
                premium_offer = primary_offer.copy()
                premium_offer['price'] = primary_offer['price'] * 1.3
                premium_offer['comfort'] = min(100, primary_offer['comfort'] + 10)
                premium_offer['reliability'] = min(100, primary_offer['reliability'] + 5)
                offers.append(premium_offer)
                
            elif self.price_strategy == "loyalty_focused" and max_offers > 2:
                # Special discount for repeat customers
                loyalty_offer = primary_offer.copy()
                loyalty_offer['price'] = primary_offer['price'] * 0.9
                loyalty_offer['loyalty_required'] = True
                offers.append(loyalty_offer)
        
        # For fractional fleets, create standardized variants
        elif self.provider_type == "fractional_fleet" and max_offers > 1:
            # Public transport tends to have standardized options
            if self.mode_type in ["bus", "train"]:
                # Standard and express options
                express_offer = primary_offer.copy()
                express_offer['price'] = primary_offer['price'] * 1.2
                express_offer['duration'] = primary_offer['duration'] * 0.8  # 20% faster
                express_offer['service_class'] = "express"
                offers.append(express_offer)
                
                # Off-peak discount if applicable
                current_hour = (start_time // 3600) % 24
                if not (7 <= current_hour <= 9 or 16 <= current_hour <= 18):  # Not rush hour
                    offpeak_offer = primary_offer.copy()
                    offpeak_offer['price'] = primary_offer['price'] * 0.8
                    offpeak_offer['service_class'] = "off_peak"
                    offers.append(offpeak_offer)
        
        # Limit to requested number of offers
        return offers[:max_offers]

    def _is_available_at_time(self, timestamp):
        """Check if provider is available at the given time"""
        if not self.is_active:
            return False
            
        # Convert timestamp to day of week and hour
        # For simplicity, assume timestamp is in seconds since start
        # and each day is 86400 seconds
        day_of_week = (timestamp // 86400) % 7
        hour_of_day = (timestamp % 86400) // 3600
        
        # Check working hours for this day
        if day_of_week not in self.working_hours:
            return False
            
        # Check if the hour falls within any scheduled shift
        for shift_start, shift_end in self.working_hours[day_of_week]:
            if shift_start <= hour_of_day < shift_end:
                return True
                
        return False

    def _is_preferred_time(self, timestamp):
        """Check if the given time is preferred by the provider"""
        # Convert timestamp to day and hour
        day_of_week = (timestamp // 86400) % 7
        hour_of_day = (timestamp % 86400) // 3600
        
        # For individuals, mid-shift times are often preferred
        if self.provider_type == "individual":
            if day_of_week in self.working_hours:
                for shift_start, shift_end in self.working_hours[day_of_week]:
                    shift_middle = (shift_start + shift_end) / 2
                    # Preferred time is in the middle of shift ±1 hour
                    if abs(hour_of_day - shift_middle) <= 1:
                        return True
        
        # For fractional fleets, non-rush hours might be preferred
        elif self.provider_type == "fractional_fleet":
            # Not rush hour
            return not (7 <= hour_of_day <= 9 or 16 <= hour_of_day <= 18)
            
        return False

    def _matches_location_preferences(self, origin, destination):
        """Check if route matches provider's location preferences"""
        # If provider doesn't have location preferences, always return true
        if not hasattr(self, 'location_preferences') or not self.location_preferences:
            return True
            
        # For fractional fleets with route corridors, check if route is in corridor
        if self.provider_type == "fractional_fleet" and "route_corridor" in self.location_preferences:
            corridor = self.location_preferences["route_corridor"]
            
            # Check if origin and destination are near the route corridor
            origin_dist = self._point_to_line_distance(origin, corridor["start"], corridor["end"])
            dest_dist = self._point_to_line_distance(destination, corridor["start"], corridor["end"])
            
            return origin_dist <= corridor["width"] and dest_dist <= corridor["width"]
        
        # For individual providers, check preferences
        if "max_distance_from_home" in self.location_preferences:
            max_distance = self.location_preferences["max_distance_from_home"]
            
            # Calculate distances from home
            origin_dist = self._calculate_distance(origin, self.home_location)
            dest_dist = self._calculate_distance(destination, self.home_location)
            
            # If either origin or destination is too far, reject
            if max_distance and (origin_dist > max_distance or dest_dist > max_distance):
                return False
        
        # Check preferred and avoided regions
        origin_preferred = False
        dest_preferred = False
        
        # Origin check
        for region in self.location_preferences.get("preferred_regions", []):
            if self._calculate_distance(origin, region["center"]) <= region["radius"]:
                origin_preferred = True
                break
                
        # Destination check
        for region in self.location_preferences.get("preferred_regions", []):
            if self._calculate_distance(destination, region["center"]) <= region["radius"]:
                dest_preferred = True
                break
        
        # Check avoided regions
        for region in self.location_preferences.get("avoid_regions", []):
            # If both origin and destination are in avoided regions, reject
            if (self._calculate_distance(origin, region["center"]) <= region["radius"] and
                self._calculate_distance(destination, region["center"]) <= region["radius"]):
                return False
        
        # If no preferred regions defined, or either origin or destination is preferred, accept
        if not self.location_preferences.get("preferred_regions") or origin_preferred or dest_preferred:
            return True
            
        # Default case - if we have preferred regions but neither point is in them, reject
        return False

    def _can_service_route(self, origin, destination):
        """
        Check if this provider can service a route from origin to destination.
        
        Args:
            origin: Origin coordinates [x, y]
            destination: Destination coordinates [x, y]
            
        Returns:
            Boolean indicating if route can be serviced
        """
        # Calculate distances
        origin_distance = self._calculate_distance(origin, self.current_location)
        route_distance = self._calculate_distance(origin, destination)
        
        # For individual providers, check if they can reach the origin
        if self.provider_type == "individual":
            # If provider is already busy and far from the origin, decline
            if self.is_moving and origin_distance > self.service_area:
                return False
                
            # If route is too long for this mode, decline
            max_route_distance = {
                'car': self.service_area * 2,
                'bike': self.service_area,
                'walk': self.service_area * 0.5
            }.get(self.mode_type, self.service_area)
            
            if route_distance > max_route_distance:
                return False
        
        # For fractional fleets, check corridor constraints
        elif self.provider_type == "fractional_fleet":
            if "route_corridor" in self.location_preferences:
                corridor = self.location_preferences["route_corridor"]
                
                # Check if both origin and destination are within service corridor
                origin_dist = self._point_to_line_distance(origin, corridor["start"], corridor["end"])
                dest_dist = self._point_to_line_distance(destination, corridor["start"], corridor["end"])
                
                if origin_dist > corridor["width"] or dest_dist > corridor["width"]:
                    return False
        
        # Additional checks based on capacity and mode
        if self.available_capacity <= 0:
            return False
            
        # In buses/trains, we need at least some minimum capacity
        if self.mode_type in ["bus", "train"] and self.available_capacity < 5:
            return False
            
        return True

    def _calculate_personalized_price(self, request):
        """
        Calculate personalized price based on provider's strategy and attributes.
        
        Args:
            request: Request details including origin, destination, time
            
        Returns:
            Calculated price
        """
        # Start with baseline distance-based calculation
        origin = request['origin']
        destination = request['destination']
        distance = self._calculate_distance(origin, destination)
        
        # Base price depends on mode and individual provider's base rate
        base_price = self.base_price * distance
        
        # Apply strategy-specific adjustments
        strategy_factor = 1.0
        
        if self.price_strategy == "value_based":
            # Higher value for longer distances and premium service
            distance_factor = min(1.5, 1.0 + (distance / 50) * 0.1)
            quality_factor = 1.0 + (self.quality_score - 70) / 100
            strategy_factor = distance_factor * quality_factor
            
        elif self.price_strategy == "cost_plus":
            # Fixed markup over operational costs
            op_cost = self._calculate_operational_cost(distance)
            markup = 1.2 if self.provider_type == "individual" else 1.1
            base_price = op_cost * markup
            
        elif self.price_strategy == "competitor_based":
            # Match or undercut competitors slightly
            market_price = self.model.marketplace.get_market_price({
                'origin': origin,
                'destination': destination,
                'service_time': request['start_time']
            })
            
            if market_price:
                # Adjust relative to market price
                if self.reputation_score > 80:
                    base_price = market_price * 1.05  # Premium for high reputation
                else:
                    base_price = market_price * 0.95  # Slight discount to compete
                    
        elif self.price_strategy == "dynamic_surge":
            # Highly responsive to demand
            is_peak = self.model.check_is_peak(request['start_time'])
            demand_factor = self.model.get_demand_factor(request['start_time'], self.mode_type)
            
            # Individual providers react more strongly to demand changes
            surge_multiplier = 1.0 + (demand_factor - 1.0) * self.surge_sensitivity
            
            if is_peak:
                surge_multiplier *= 1.2  # Additional peak multiplier
                
            strategy_factor = surge_multiplier
            
        elif self.price_strategy == "time_optimized":
            # Price varies by time of day based on provider preferences
            current_hour = (request['start_time'] // 3600) % 24
            
            if 7 <= current_hour <= 9:  # Morning rush
                strategy_factor = 1.25
            elif 16 <= current_hour <= 18:  # Evening rush 
                strategy_factor = 1.2
            elif 22 <= current_hour or current_hour <= 5:  # Late night
                strategy_factor = 1.3  # Premium for late hours
            elif self._is_preferred_time(request['start_time']):
                strategy_factor = 0.9  # Discount for preferred hours
        
        elif self.price_strategy == "loyalty_focused":
            # Base prices generally lower, focus on repeat business
            strategy_factor = 0.9  # Generally lower base price
        
        # Apply provider type adjustments
        if self.provider_type == "fractional_fleet":
            # Public transport tends to have more standardized, lower per-distance pricing
            if self.mode_type == "bus":
                base_price = min(base_price, self.base_price * 2 + (distance * 0.1))
            elif self.mode_type == "train":
                base_price = min(base_price, self.base_price * 3 + (distance * 0.15))
                
            # Apply group discount for larger capacity
            base_price = base_price * (1 - min(0.3, self.capacity * 0.01))
        
        # Apply time-based adjustments
        time_factor = self._calculate_time_decay_factor(
            request['start_time'] - self.model.schedule.time)
        
        # Apply all factors
        final_price = base_price * strategy_factor * time_factor
        
        # Ensure price stays within reasonable bounds
        min_price = base_price * self.price_floor
        max_price = base_price * self.price_ceiling
        final_price = max(min_price, min(max_price, final_price))
        
        return final_price

    def _calculate_operational_cost(self, distance):
        """Calculate operational cost for a given distance"""
        # Different modes have different cost structures
        if self.mode_type == "car":
            # Car has higher per-distance costs (fuel, maintenance)
            return self.base_price * 0.6 * distance + 5  # Fixed + variable costs
        elif self.mode_type == "bike":
            # Bike has mostly fixed costs plus personal effort
            return self.base_price * 0.3 * distance + 3
        elif self.mode_type == "bus":
            # Bus has high fixed costs, lower marginal costs per passenger
            return (self.base_price * 0.4 * distance + 20) / max(1, self.capacity)
        elif self.mode_type == "train":
            # Train has very high fixed costs, very low marginal costs
            return (self.base_price * 0.3 * distance + 50) / max(1, self.capacity)
        else:
            # Default case
            return self.base_price * 0.5 * distance + 4

    def _calculate_time_decay_factor(self, time_to_service):
        """
        Calculate the time decay factor for pricing.
        
        Implements a time-decay pricing model where:
        1. Services far in the future have discounted prices
        2. Last-minute services have premium prices
        3. The pricing curve follows a U-shape with minimum at optimal booking time
        
        Args:
            time_to_service: Time in seconds until the service starts
            
        Returns:
            Time decay price factor (multiplier)
        """
        # Express time to service in hours for more intuitive calculations
        hours_to_service = time_to_service / 3600
        
        # Parameters defining the time-decay curve - now personalized
        if self.provider_type == "individual":
            # Individual providers have more personalized and varied curves
            optimal_booking_time = random.uniform(12, 36)  # 12-36 hours in advance
            min_price_factor = random.uniform(0.8, 0.9)  # 10-20% discount at optimal time
            max_distant_factor = random.uniform(1.0, 1.2)  # 0-20% premium for distant bookings
            max_immediate_factor = random.uniform(1.3, 1.7)  # 30-70% premium for immediate
            
            # Apply strategy adjustments
            if self.price_strategy == "dynamic_surge":
                max_immediate_factor += 0.2  # More premium for immediate service
            elif self.price_strategy == "loyalty_focused":
                min_price_factor -= 0.05  # Deeper discounts at optimal time
                max_immediate_factor -= 0.1  # Less premium for immediate service
        else:
            # Fractional fleets have more standardized pricing
            optimal_booking_time = 24  # 24 hours
            min_price_factor = 0.9  # 10% discount
            max_distant_factor = 1.05  # 5% premium for distant bookings
            max_immediate_factor = 1.2  # 20% premium for immediate
        
        if hours_to_service < 1:  # Very immediate service (less than 1 hour)
            # Sharp increase for last-minute bookings
            return max_immediate_factor
            
        elif hours_to_service < optimal_booking_time:
            # Decreasing curve from immediate to optimal time
            # Use quadratic decay for smooth curve
            progress = hours_to_service / optimal_booking_time
            return max_immediate_factor - (max_immediate_factor - min_price_factor) * (progress ** 0.8)
            
        elif hours_to_service < 72:  # Up to 3 days in advance
            # Increasing curve from optimal time to distant future
            # Slower increase compared to immediate bookings
            excess_hours = hours_to_service - optimal_booking_time
            max_excess = 72 - optimal_booking_time
            progress = excess_hours / max_excess
            return min_price_factor + (max_distant_factor - min_price_factor) * (progress ** 1.5)
            
        else:  # Far future bookings (more than 3 days)
            # Plateau at maximum distant booking factor
            return max_distant_factor

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

    def _calculate_estimated_time(self, request):
        """
        Calculate estimated service time based on distance, mode, and conditions.
        
        Args:
            request: Request dictionary with origin and destination
            
        Returns:
            Estimated time in minutes
        """
        origin = request['origin']
        destination = request['destination']

        # Calculate direct distance
        distance = self._calculate_distance(origin, destination)

        # Get current speed based on conditions
        current_speed = self._get_current_speed(request['start_time'])

        # Calculate time based on distance and speed
        estimated_time = distance / current_speed

        # Add mode-specific overhead time
        overhead_map = {
            'car': 3.0,
            'bike': 2.0,
            'bus': 8.0,
            'train': 5.0,
            'walk': 0.0
        }
        
        overhead = overhead_map.get(self.mode_type, 2.0)
        
        # Individual providers might add more variability
        if self.provider_type == "individual":
            # Add some random variation (±10%)
            estimated_time *= random.uniform(0.9, 1.1)
        else:
            # Fractional fleets (public transport) tend to have more fixed schedules
            # Round to nearest 5 minutes for buses, 10 minutes for trains
            if self.mode_type == "bus":
                estimated_time = round(estimated_time / 5) * 5
            elif self.mode_type == "train":
                estimated_time = round(estimated_time / 10) * 10
        
        return estimated_time + overhead

    def _get_current_speed(self, time_of_day):
        """Get current speed based on time of day and conditions"""
        base_speed = self._get_mode_speed()
        
        # Time-based adjustments
        hour = (time_of_day // 3600) % 24
        
        # Rush hour speed reductions
        if 7 <= hour <= 9 or 16 <= hour <= 18:
            if self.mode_type == "car":
                return base_speed * 0.7  # Cars most affected by rush hour
            elif self.mode_type == "bike":
                return base_speed * 0.9  # Bikes less affected
            elif self.mode_type == "bus":
                return base_speed * 0.8  # Buses somewhat affected
            elif self.mode_type == "train":
                return base_speed * 0.95  # Trains least affected
        
        # Night time speed increases for road vehicles
        elif 22 <= hour or hour <= 5:
            if self.mode_type in ["car", "bike"]:
                return base_speed * 1.2  # Faster at night due to less traffic
        
        # Default case - normal speed
        return base_speed

    def _generate_route(self, origin, destination):
        """
        Generate a realistic route from origin to destination.
        
        Args:
            origin: Origin coordinates [x, y]
            destination: Destination coordinates [x, y]
            
        Returns:
            List of route points [[x1,y1], [x2,y2], ...]
        """
        # Different route generation based on provider type and mode
        if self.provider_type == "fractional_fleet" and "route_corridor" in self.location_preferences:
            # For public transport with fixed corridors, route follows the corridor
            corridor = self.location_preferences["route_corridor"]
            
            # Find best points along the corridor
            origin_proj = self._project_point_to_line(origin, corridor["start"], corridor["end"])
            dest_proj = self._project_point_to_line(destination, corridor["start"], corridor["end"])
            
            # Calculate progress along the corridor line
            start_to_end_dist = self._calculate_distance(corridor["start"], corridor["end"])
            origin_prog = self._calculate_distance(corridor["start"], origin_proj) / start_to_end_dist
            dest_prog = self._calculate_distance(corridor["start"], dest_proj) / start_to_end_dist
            
            # Generate intermediate stops (more for bus, fewer for train)
            stops_count = 6 if self.mode_type == "bus" else 3
            route = [origin]  # Start with actual origin
            
            # Add first corridor point
            route.append(origin_proj)
            
            # Calculate direction from start to end
            min_prog = min(origin_prog, dest_prog)
            max_prog = max(origin_prog, dest_prog)
            
            # Add intermediate corridor points (stops)
            for i in range(1, stops_count + 1):
                progress = min_prog + (max_prog - min_prog) * (i / (stops_count + 1))
                x = corridor["start"][0] + progress * (corridor["end"][0] - corridor["start"][0])
                y = corridor["start"][1] + progress * (corridor["end"][1] - corridor["start"][1])
                route.append([x, y])
            
            # Add final corridor point
            route.append(dest_proj)
            
            # End with actual destination
            route.append(destination)
        else:
            # For individual providers or non-corridor services, generate more direct routes
            # Create grid-based routes with some randomness to mimic street patterns
            
            # Direct vector from origin to destination
            dx = destination[0] - origin[0]
            dy = destination[1] - origin[1]
            
            # Determine if route should prioritize horizontal first or vertical first
            if abs(dx) > abs(dy):
                # Horizontal first, then vertical
                midpoint_1 = [origin[0] + dx * 0.7, origin[1] + dy * 0.3]
                midpoint_2 = [origin[0] + dx * 0.3, origin[1] + dy * 0.7]
            else:
                # Vertical first, then horizontal
                midpoint_1 = [origin[0] + dx * 0.3, origin[1] + dy * 0.7]
                midpoint_2 = [origin[0] + dx * 0.7, origin[1] + dy * 0.3]
            
            # Add some randomness to midpoints
            deviation = min(abs(dx), abs(dy)) * 0.2
            midpoint_1[0] += random.uniform(-deviation, deviation)
            midpoint_1[1] += random.uniform(-deviation, deviation)
            midpoint_2[0] += random.uniform(-deviation, deviation)
            midpoint_2[1] += random.uniform(-deviation, deviation)
            
            # Create route with intermediate points
            route = [origin, midpoint_1, midpoint_2, destination]
            
            # For longer routes, add more detail
            if self._calculate_distance(origin, destination) > 30:
                detailed_route = [route[0]]
                
                # Insert additional points between each segment
                for i in range(len(route) - 1):
                    start = route[i]
                    end = route[i+1]
                    
                    # Add 1-2 points between each segment
                    num_extra = random.randint(1, 2)
                    for j in range(1, num_extra + 1):
                        t = j / (num_extra + 1)
                        x = start[0] + t * (end[0] - start[0])
                        y = start[1] + t * (end[1] - start[1])
                        
                        # Add some random deviation to mimic real streets
                        deviation = min(abs(dx), abs(dy)) * 0.1
                        x += random.uniform(-deviation, deviation)
                        y += random.uniform(-deviation, deviation)
                        
                        detailed_route.append([x, y])
                        
                    detailed_route.append(end)
                    
                route = detailed_route

        return route

    def _project_point_to_line(self, point, line_start, line_end):
        """Project a point onto a line segment and return the projected point"""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Line vector
        dx = x2 - x1
        dy = y2 - y1
        
        # Line length squared
        line_length_sq = dx*dx + dy*dy
        
        # If line is a point, return the start point
        if line_length_sq == 0:
            return line_start
        
        # Calculate projection parameter
        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / line_length_sq))
        
        # Calculate projected point
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        return [proj_x, proj_y]

    def generate_offer(self, request):
        """
        Generate an offer for a travel request based on provider's capabilities.
        
        Args:
            request: The travel request
            
        Returns:
            Offer details or None if cannot service
        """
        # Check if provider is available at the requested time
        start_time = request.get('start_time', self.model.current_step + 1)
        if not self._is_available_at_time(start_time):
            return None
        
        # Check if request can be serviced
        origin = request.get('origin', [0, 0])
        destination = request.get('destination', [0, 0])
        
        if not self._can_service_route(origin, destination):
            return None
            
        # Check location preferences
        if not self._matches_location_preferences(origin, destination):
            return None
        
        # Check capacity
        if self.available_capacity <= 0:
            return None
        
        # Calculate distance and price
        distance = self._calculate_distance(origin, destination)
        
        # Use personalized pricing strategy
        price = self._calculate_personalized_price({
            'origin': origin,
            'destination': destination,
            'start_time': start_time
        })
        
        # Check if trip meets minimum value criteria
        trip_value = price * distance
        if trip_value < self._calculate_min_trip_value():
            return None
        
        # Calculate route
        route = self._generate_route(origin, destination)
        
        # Calculate estimated time
        estimated_time = self._calculate_estimated_time({
            'origin': origin,
            'destination': destination,
            'start_time': start_time
        })
        
        # Generate offer with personalized attributes
        offer = {
            'provider_id': self.unique_id,
            'provider_type': self.provider_type,
            'provider_name': self.name,
            'request_id': request.get('request_id', 0),
            'price': price,
            'route': route,
            'estimated_time': estimated_time,
            'start_time': start_time,
            'created_at': self.model.current_step if hasattr(self.model, 'current_step') else 0,
            'mode': self.mode_type,
            'reliability': self.reliability,
            'quality_score': self.quality_score,
            'reputation': self.reputation_score,
            'available_capacity': self.available_capacity
        }
        
        # For fractional fleets, add special attributes
        if self.provider_type == "fractional_fleet":
            offer['token_holder_count'] = len(self.fleet_token_holders)
            
            # Add service class for public transport
            if self.mode_type in ["bus", "train"]:
                offer['service_class'] = "standard"
                
                # Express service if longer distance and higher price
                if distance > 20 and price > self.base_price * distance * 1.1:
                    offer['service_class'] = "express"
                    offer['estimated_time'] *= 0.8  # Express is faster
                    
        # For individual providers, add personal preferences
        else:
            # Flag if this is a preferred time for the provider
            offer['is_preferred_time'] = self._is_preferred_time(start_time)
            
            # Add driver details for transparency
            offer['driver_details'] = {
                'experience': random.randint(1, 10),  # Years of experience
                'rating': self.reputation_score / 20  # Convert to 0-5 scale
            }
            
        return offer

    def mint_service_nft(self, request_id, offer_id, commuter_id):
        """
        Create NFT for service when offer is accepted.
        
        Args:
            request_id: ID of the accepted request
            offer_id: ID of the accepted offer
            commuter_id: ID of the commuter who accepted the offer
            
        Returns:
            NFT ID or None if failed
        """
        if offer_id not in self.active_bids:
            self.logger.error(f"Offer {offer_id} not found for minting NFT")
            return None
            
        offer = self.active_bids[offer_id]

        # Generate NFT metadata with more personalized information
        nft_metadata = {
            'service_type': self.mode_type,
            'provider_name': self.name,
            'provider_id': self.unique_id,
            'provider_type': self.provider_type,
            'start_time': offer['start_time'],
            'duration': offer['estimated_time'],
            'route': offer['route'],
            'price': offer['price'],
            'request_id': request_id,
            'offer_id': offer_id,
            'creation_time': self.model.schedule.time,
            'valid_until': offer['start_time'] + offer['estimated_time'] * 2,  # Double service time
            'reputation_score': self.reputation_score,
            'terms': self._generate_service_terms(offer)
        }
        
        # Add provider-specific metadata
        if self.provider_type == "individual":
            nft_metadata['driver_details'] = {
                'name': self.name,
                'experience': random.randint(1, 10),
                'languages': random.sample(["English", "Chinese", "Spanish", "Arabic", "Hindi"], 
                                         random.randint(1, 2))
            }
        else:  # fractional_fleet
            nft_metadata['fleet_details'] = {
                'token_holders': len(self.fleet_token_holders),
                'service_class': offer.get('service_class', 'standard'),
                'amenities': random.sample(["WiFi", "USB Charging", "Air Conditioning", "Accessibility"], 
                                         random.randint(1, 3))
            }

        # Create service details for blockchain
        service_details = {
            'request_id': request_id,
            'price': offer['price'],
            'start_time': offer['start_time'],
            'duration': offer['estimated_time'] * 60,  # Convert to seconds
            'route_details': {
                'route': offer['route'],
                'estimated_time': offer['estimated_time'],
                'distance': self._calculate_route_distance(offer['route']),
                'mode': self.mode_type,
                'provider_type': self.provider_type
            }
        }

        # Add commuter to preferred customers list if not already there
        if commuter_id not in self.preferred_customers:
            self.preferred_customers.add(commuter_id)

        # Mint NFT on blockchain
        success, nft_id = self.blockchain_interface.create_nft(
            service_details, self.unique_id, commuter_id)

        if success:
            # Update local records
            self.issued_nfts[nft_id] = {
                'metadata': nft_metadata,
                'status': 'active',
                'request_id': request_id,
                'offer_id': offer_id,
                'commuter_id': commuter_id,
                'creation_time': self.model.schedule.time
            }

            # Update offer status
            self.active_bids[offer_id]['status'] = 'accepted'

            # Update capacity
            self.available_capacity = max(0, self.available_capacity - 1)
            
            # Record win for learning
            self.win_rate_history.append(1)
            
            # Update physical state - set target to customer pickup location
            self.target_location = offer['route'][0]  # Origin point
            self.is_moving = True
            self.current_route = self._generate_route(self.current_location, self.target_location)
            
            self.logger.info(f"Minted NFT {nft_id} for offer {offer_id}, moving to pickup location")
            
            return nft_id
        else:
            self.logger.error(f"Failed to mint NFT for offer {offer_id}")
            return None

    def _generate_service_terms(self, offer):
        """Generate service terms based on provider type and attributes"""
        # Different terms based on provider type and mode
        if self.provider_type == "individual":
            # Individual providers have more varied terms
            cancellation_policy = random.choice([
                "Free cancellation up to 1 hour before service",
                "80% refund if canceled at least 2 hours before service",
                "90% refund if canceled at least 12 hours before service"
            ])
            
            # Better terms for higher reputation providers
            if self.reputation_score > 85:
                service_guarantee = True
                min_service_level = 85
            else:
                service_guarantee = self.reputation_score > 75
                min_service_level = max(60, self.reputation_score - 10)
                
        else:  # fractional_fleet
            # Public transport typically has standardized terms
            if self.mode_type == "bus":
                cancellation_policy = "90% refund if canceled at least 2 hours before service"
            elif self.mode_type == "train":
                cancellation_policy = "Full refund if canceled at least 24 hours before service"
            else:
                cancellation_policy = "85% refund if canceled at least 6 hours before service"
                
            service_guarantee = True
            min_service_level = 75
            
        return {
            'cancellation_policy': cancellation_policy,
            'refund_policy': cancellation_policy,  # Same as cancellation for simplicity
            'transfer_rights': 'Transferable until 1 hour before service time',
            'service_guarantee': service_guarantee,
            'min_service_level': min_service_level
        }

    def _calculate_route_distance(self, route):
        """
        Calculate the total distance of a route.
        
        Args:
            route: List of route points [[x1,y1], [x2,y2], ...]
            
        Returns:
            Total distance
        """
        if not route or len(route) < 2:
            return 0
            
        total_distance = 0
        for i in range(len(route) - 1):
            point1 = route[i]
            point2 = route[i + 1]
            
            # Calculate segment distance
            segment_distance = self._calculate_distance(point1, point2)
            
            total_distance += segment_distance
            
        return total_distance

    def accept_booking(self, commuter_id, request_id, price, start_time, route=None):
        """
        Accept a direct booking from a commuter (bypassing auction).
        
        Args:
            commuter_id: Commuter making the booking
            request_id: Associated request ID
            price: Agreed price
            start_time: Service start time
            route: Optional route information
            
        Returns:
            Boolean indicating success
        """
        # Check if provider is available at this time
        if not self._is_available_at_time(start_time):
            self.logger.info(f"Booking rejected: Provider {self.unique_id} not available at requested time")
            return False
            
        # Check if capacity is available
        if self.available_capacity <= 0:
            self.logger.info(f"Booking rejected: No capacity available for provider {self.unique_id}")
            return False
        
        # For individual providers, check if already committed elsewhere
        if self.provider_type == "individual" and self.is_moving:
            # Calculate time to complete current service and get back
            current_service_end = 0
            for nft_id, nft in self.issued_nfts.items():
                if nft['status'] == 'active':
                    service_time = nft['metadata']['start_time']
                    service_duration = nft['metadata']['duration']
                    service_end = service_time + service_duration
                    if service_end > current_service_end:
                        current_service_end = service_end
            
            # Add travel time back to service area
            travel_time = 30  # Assume 30 minutes to get back
            
            # If already committed during requested time, reject
            if start_time < current_service_end + travel_time:
                self.logger.info(f"Booking rejected: Provider {self.unique_id} already committed during requested time")
                return False
            
        # Generate a synthetic offer
        offer_id = str(uuid.uuid4())
        estimated_time = 30  # Default 30 minutes if no route provided
        
        # Create or enhance route if needed
        if not route or len(route) < 2:
            origin = self.current_location
            destination = [random.randint(0, self.model.grid_width-1), 
                          random.randint(0, self.model.grid_height-1)]
            
            if 'origin' in self.model.requests.get(request_id, {}):
                origin = self.model.requests[request_id]['origin']
            if 'destination' in self.model.requests.get(request_id, {}):
                destination = self.model.requests[request_id]['destination']
                
            route = self._generate_route(origin, destination)
        
        # Calculate estimated time based on route
        distance = self._calculate_route_distance(route)
        estimated_time = distance / self._get_current_speed(start_time)
        
        # Check if the commuter is in preferred customers list
        is_preferred = commuter_id in self.preferred_customers
        
        # Apply discounts for preferred customers
        if is_preferred and self.price_strategy == "loyalty_focused":
            price *= 0.9  # 10% discount for returning customers
        
        # Create offer object with more attributes
        offer = {
            'offer_id': offer_id,
            'provider_id': self.unique_id,
            'provider_type': self.provider_type,
            'provider_name': self.name,
            'request_id': request_id,
            'price': price,
            'mode': self.mode_type,
            'start_time': start_time,
            'estimated_time': estimated_time,
            'route': route,
            'status': 'accepted',
            'creation_time': self.model.schedule.time,
            'reliability': self.reliability,
            'quality_score': self.quality_score,
            'reputation': self.reputation_score,
            'is_preferred_customer': is_preferred
        }
        
        # For fractional fleets, add governance records
        if self.provider_type == "fractional_fleet" and self.fleet_token_holders:
            # Record trip for profit distribution
            trip_record = {
                'commuter_id': commuter_id,
                'price': price,
                'start_time': start_time,
                'distance': distance
            }
            
            if not hasattr(self, 'fleet_trip_records'):
                self.fleet_trip_records = []
                
            self.fleet_trip_records.append(trip_record)
        
        # Store offer
        self.active_bids[offer_id] = offer
        
        # Create NFT for the service
        nft_id = self.mint_service_nft(request_id, offer_id, commuter_id)
        
        # Booking is successful if NFT was created
        if nft_id:
            self.logger.info(f"Direct booking accepted from commuter {commuter_id}, NFT {nft_id} created")
            return True
        else:
            self.logger.error(f"Failed to complete direct booking from commuter {commuter_id}")
            return False

    def generate_bundle_component_offer(self, bundle_request, segment):
        """
        Generate offer for a specific segment of a bundle.
        
        Args:
            bundle_request: Bundle request information
            segment: Segment information (origin, destination, time)
            
        Returns:
            Offer dictionary with digital signature
        """
        # Check if provider is available at the requested time
        if not self._is_available_at_time(segment['start_time']):
            return None
            
        # Check if can service the route based on location and preferences
        if not self._can_service_route(segment['origin'], segment['destination']):
            return None
            
        if not self._matches_location_preferences(segment['origin'], segment['destination']):
            return None
            
        # Calculate personalized price with bundle discount
        base_price = self._calculate_personalized_price({
            'origin': segment['origin'],
            'destination': segment['destination'],
            'start_time': segment['start_time']
        })

        # Apply bundle-specific discount (more aggressive for fractional fleets)
        if self.provider_type == "individual":
            bundle_discount = random.uniform(0.05, 0.15)  # 5-15% discount
        else:
            bundle_discount = random.uniform(0.1, 0.2)  # 10-20% discount for fleets
            
        discounted_price = base_price * (1 - bundle_discount)
        
        # Check if the trip meets minimum value criteria
        distance = self._calculate_distance(segment['origin'], segment['destination'])
        trip_value = discounted_price * distance
        if trip_value < self._calculate_min_trip_value() * 0.8:  # Lower threshold for bundles
            return None

        # Create detailed offer with provider characteristics
        offer = {
            'provider_id': self.unique_id,
            'provider_type': self.provider_type,
            'provider_name': self.name,
            'bundle_id': bundle_request['bundle_id'],
            'segment_id': segment['segment_id'],
            'price': discounted_price,
            'base_price': base_price,
            'bundle_discount': bundle_discount,
            'start_time': segment['start_time'],
            'route': self._generate_route(segment['origin'], segment['destination']),
            'estimated_time': self._calculate_estimated_time({
                'origin': segment['origin'],
                'destination': segment['destination'],
                'start_time': segment['start_time']
            }),
            'mode': self.mode_type,
            'quality_score': self.quality_score,
            'reliability': self.reliability,
            'reputation': self.reputation_score,
            'creation_time': self.model.schedule.time,
            'valid_until': segment['start_time'] - 3600,  # Valid until 1 hour before service
            'available_capacity': self.available_capacity
        }
        
        # Add provider-specific details
        if self.provider_type == "individual":
            offer['experience_level'] = random.randint(1, 10)
            
            # Individual providers might have special conditions
            if random.random() < 0.3:
                offer['special_conditions'] = random.choice([
                    "No more than 2 passengers",
                    "No large luggage",
                    "Prefer cash payment"
                ])
        else:
            # Fractional fleets add community benefits
            offer['community_owned'] = True
            offer['token_holders'] = len(self.fleet_token_holders)
            
            if self.mode_type in ["bus", "train"]:
                offer['service_class'] = random.choice(["standard", "express"])
                
                if offer['service_class'] == "express":
                    offer['estimated_time'] *= 0.8  # Express is faster
                    offer['price'] *= 1.1  # And slightly more expensive

        # Sign the offer (via blockchain interface)
        offer['signature'] = self.blockchain_interface.sign_bundle_offer(self, offer)
        
        self.logger.info(f"Generated bundle component offer for segment {segment['segment_id']}")

        return offer

    def update(self):
        """
        Main update method called on each simulation step.
        """
        # Update physical position if moving
        if self.is_moving:
            self._update_position()

        # Check for accepted offers and update capacity
        self._process_accepted_offers()
        self._update_capacity()

        # Update reputation based on service quality
        self._update_reputation()
        
        # Update learning parameters and pricing strategy
        self._update_learning_parameters()
        self._update_pricing_strategy()
        
        # Apply time-decay adjustments to active offers
        self._update_offer_time_decay()
        
        # For fractional fleets, process governance votes and distribute earnings
        if self.provider_type == "fractional_fleet" and self.fleet_token_holders:
            self._process_governance()
            self._distribute_earnings()

    def _update_position(self):
        """Update provider's physical position on the grid"""
        if not self.is_moving or not self.target_location:
            return
            
        # Calculate distance to target
        distance_to_target = self._calculate_distance(self.current_location, self.target_location)
        
        # Get movement speed
        movement_speed = self._get_mode_speed() / 10  # Scale down for movement per step
        
        if distance_to_target <= movement_speed:
            # Reached target
            self.current_location = self.target_location
            
            # Check if we've reached the end of current route
            if self.current_route:
                if len(self.current_route) > 1:
                    # Move to next point in route
                    self.current_route.pop(0)
                    self.target_location = self.current_route[0]
                else:
                    # End of route
                    self.is_moving = False
                    self.current_route = []
                    self.target_location = None
        else:
            # Move toward target
            direction_x = self.target_location[0] - self.current_location[0]
            direction_y = self.target_location[1] - self.current_location[1]
            
            # Normalize direction
            distance = math.sqrt(direction_x**2 + direction_y**2)
            direction_x /= distance
            direction_y /= distance
            
            # Move in that direction
            self.current_location[0] += direction_x * movement_speed
            self.current_location[1] += direction_y * movement_speed

    def _process_accepted_offers(self):
        """
        Check for offers that have been accepted on the blockchain.
        """
        for offer_id, offer in list(self.active_bids.items()):
            if offer['status'] == 'pending':
                # Check blockchain status
                blockchain_status = self.blockchain_interface.check_offer_status(offer_id)
                if blockchain_status == 'accepted':
                    # Offer accepted, mint NFT
                    self.logger.info(f"Offer {offer_id} accepted on blockchain")
                    
                    # Get commuter ID from blockchain
                    commuter_id = self.blockchain_interface.get_offer_commuter(offer_id)
                    
                    # Mint NFT
                    if commuter_id:
                        self.mint_service_nft(offer['request_id'], offer_id, commuter_id)
                        
                        # Add commuter to preferred customers list
                        if commuter_id not in self.preferred_customers:
                            self.preferred_customers.add(commuter_id)
                    else:
                        self.logger.error(f"Could not get commuter ID for accepted offer {offer_id}")
                    
                elif blockchain_status == 'rejected':
                    # Offer rejected
                    self.active_bids[offer_id]['status'] = 'rejected'
                    
                    # Record loss for learning
                    self.win_rate_history.append(0)
                    self.logger.info(f"Offer {offer_id} rejected")
                    
                # Clean up old pending offers
                current_time = self.model.schedule.time
                if current_time - offer.get('creation_time', 0) > 24 * 3600:  # 24 hours old
                    self.logger.info(f"Removing old pending offer {offer_id}")
                    self.active_bids.pop(offer_id)

    def _update_capacity(self):
        """
        Update available capacity as services complete.
        """
        current_time = self.model.schedule.time

        # Check NFTs for completed services
        for nft_id, nft in list(self.issued_nfts.items()):
            if nft['status'] == 'active':
                service_start_time = nft['metadata']['start_time']
                service_duration = nft['metadata']['duration']
                service_end_time = service_start_time + service_duration
                
                if current_time > service_end_time:
                    # Service completed
                    self.issued_nfts[nft_id]['status'] = 'completed'

                    # Restore capacity
                    self.available_capacity = min(self.capacity, self.available_capacity + 1)

                    # Record service completion
                    self.service_history.append({
                        'nft_id': nft_id,
                        'request_id': nft['request_id'],
                        'commuter_id': nft['commuter_id'],
                        'start_time': service_start_time,
                        'end_time': service_end_time,
                        'price': nft['metadata']['price'],
                        'completion_time': current_time
                    })
                    
                    self.logger.info(f"Service for NFT {nft_id} completed, capacity restored to {self.available_capacity}")
                    
                    # For individual providers, move back to home location if no other services
                    if self.provider_type == "individual" and not self.is_moving:
                        active_count = sum(1 for nft in self.issued_nfts.values() if nft['status'] == 'active')
                        
                        if active_count == 0:
                            # Head back home
                            self.target_location = self.home_location
                            self.is_moving = True
                            self.current_route = self._generate_route(self.current_location, self.target_location)
                            self.logger.info(f"Provider {self.unique_id} returning to home location")
                    
                    # Update quality score based on performance
                    # This is a simplified model - in reality this would come from customer ratings
                    on_time_probability = min(95, self.reliability) / 100
                    was_on_time = random.random() < on_time_probability
                    
                    if was_on_time:
                        self.reliability = min(100, self.reliability + 0.5)
                    else:
                        self.reliability = max(50, self.reliability - 1.0)
                        
                    # Notify model of service completion if it has the method
                    if hasattr(self.model, 'on_service_completed'):
                        self.model.on_service_completed(
                            provider_id=self.unique_id,
                            commuter_id=nft['commuter_id'],
                            was_on_time=was_on_time,
                            nft_id=nft_id
                        )
                        
                    # For fractional fleets, record profits for distribution
                    if self.provider_type == "fractional_fleet" and self.fleet_token_holders:
                        if not hasattr(self, 'fleet_profits'):
                            self.fleet_profits = []
                            
                        # Calculate operating costs
                        distance = 0
                        if 'route' in nft['metadata']:
                            distance = self._calculate_route_distance(nft['metadata']['route'])
                            
                        op_cost = self._calculate_operational_cost(distance)
                        profit = nft['metadata']['price'] - op_cost
                        
                        self.fleet_profits.append({
                            'nft_id': nft_id,
                            'commuter_id': nft['commuter_id'],
                            'revenue': nft['metadata']['price'],
                            'cost': op_cost,
                            'profit': profit,
                            'timestamp': current_time
                        })

    def _update_reputation(self):
        """Update provider reputation based on service quality"""
        # In a real system, this would come from customer ratings
        # For simulation, we model it based on reliability and random factors
        
        # Skip updates sometimes to model infrequent ratings
        if random.random() > 0.1:  # Only 10% chance of update per step
            return
            
        # Get recent service completions
        recent_services = [s for s in self.service_history 
                          if self.model.schedule.time - s['completion_time'] < 86400]  # Last day
                          
        if not recent_services:
            return
            
        # Calculate reliability score (on-time percentage)
        on_time_ratings = []
        for service in recent_services:
            expected_end = service['start_time'] + service['end_time'] - service['start_time']
            actual_end = service['completion_time']
            
            # Calculate lateness
            lateness = max(0, actual_end - expected_end)
            
            # Score from 0-1 based on lateness
            if lateness == 0:
                score = 1.0  # Perfect on-time
            elif lateness < 300:  # Less than 5 minutes
                score = 0.9
            elif lateness < 900:  # Less than 15 minutes
                score = 0.7
            elif lateness < 1800:  # Less than 30 minutes
                score = 0.5
            else:
                score = 0.3
                
            on_time_ratings.append(score)
            
        # Calculate quality score based on attributes
        # In reality, this would be from customer ratings
        quality_ratings = []
        for _ in range(len(recent_services)):
            base_quality = self.quality_score / 100  # Convert to 0-1 scale
            
            # Add some randomness to model customer perception
            rating = base_quality + random.uniform(-0.1, 0.1)
            rating = max(0, min(1, rating))  # Keep in 0-1 range
            
            quality_ratings.append(rating)
            
        # Combine scores (weighted average)
        if on_time_ratings and quality_ratings:
            avg_on_time = sum(on_time_ratings) / len(on_time_ratings)
            avg_quality = sum(quality_ratings) / len(quality_ratings)
            
            new_score = (avg_on_time * 0.6 + avg_quality * 0.4) * 100  # Convert to 0-100 scale
            
            # Smooth updates to reputation (80% old, 20% new)
            self.reputation_score = 0.8 * self.reputation_score + 0.2 * new_score
            
            # Record reputation history
            self.reputation_history.append((self.model.schedule.time, self.reputation_score))
            
            # Keep history limited
            if len(self.reputation_history) > 100:
                self.reputation_history = self.reputation_history[-100:]

    def _process_governance(self):
        """Process governance votes for fractional fleet providers"""
        if self.provider_type != "fractional_fleet" or not self.fleet_token_holders:
            return
            
        # Process pending votes
        for vote_id, vote_data in list(self.governance_votes.items()):
            if vote_data['status'] == 'active' and self.model.schedule.time >= vote_data['end_time']:
                # Calculate results
                yes_votes = 0
                no_votes = 0
                
                for voter, vote in vote_data['votes'].items():
                    weight = self.fleet_token_holders.get(voter, 0)
                    if vote:
                        yes_votes += weight
                    else:
                        no_votes += weight
                
                # Determine if proposal passed
                passed = yes_votes > no_votes
                vote_data['status'] = 'passed' if passed else 'rejected'
                
                # Execute proposal if passed
                if passed:
                    proposal_type = vote_data['proposal_type']
                    
                    if proposal_type == 'pricing':
                        # Update pricing strategy or parameters
                        new_strategy = vote_data['proposal_data'].get('strategy')
                        if new_strategy in ["value_based", "cost_plus", "competitor_based", 
                                         "dynamic_surge", "time_optimized", "loyalty_focused"]:
                            self.price_strategy = new_strategy
                            
                        # Update price floor/ceiling
                        new_floor = vote_data['proposal_data'].get('price_floor')
                        if new_floor is not None:
                            self.price_floor = max(0.4, min(0.8, new_floor))
                            
                        new_ceiling = vote_data['proposal_data'].get('price_ceiling')
                        if new_ceiling is not None:
                            self.price_ceiling = max(1.2, min(2.5, new_ceiling))
                            
                    elif proposal_type == 'route':
                        # Update route preferences
                        new_route = vote_data['proposal_data'].get('route_corridor')
                        if new_route and 'start' in new_route and 'end' in new_route:
                            if 'location_preferences' not in self.__dict__:
                                self.location_preferences = {}
                                
                            self.location_preferences['route_corridor'] = new_route
                            
                    elif proposal_type == 'service':
                        # Update service parameters
                        new_quality = vote_data['proposal_data'].get('quality_target')
                        if new_quality is not None:
                            self.quality_score = max(50, min(100, new_quality))
                            
                        new_reliability = vote_data['proposal_data'].get('reliability_target')
                        if new_reliability is not None:
                            self.reliability = max(50, min(100, new_reliability))
                
                self.logger.info(f"Governance proposal {vote_id} {vote_data['status']}")
        
        # Occasionally create new governance proposals
        if random.random() < 0.01:  # 1% chance per step
            # Generate a new proposal
            proposal_types = ['pricing', 'route', 'service']
            proposal_type = random.choice(proposal_types)
            
            proposal_data = {}
            
            if proposal_type == 'pricing':
                # Pricing strategy proposal
                strategies = ["value_based", "cost_plus", "competitor_based", 
                           "dynamic_surge", "time_optimized", "loyalty_focused"]
                proposal_data['strategy'] = random.choice(strategies)
                
                # Price parameters
                adjust_by = random.uniform(-0.1, 0.1)
                proposal_data['price_floor'] = max(0.4, min(0.8, self.price_floor + adjust_by))
                proposal_data['price_ceiling'] = max(1.2, min(2.5, self.price_ceiling + adjust_by))
                
            elif proposal_type == 'route':
                # Route changes
                if 'route_corridor' in self.location_preferences:
                    current = self.location_preferences['route_corridor']
                    
                    # Small adjustment to current route
                    start_x = current['start'][0] + random.uniform(-5, 5)
                    start_y = current['start'][1] + random.uniform(-5, 5)
                    end_x = current['end'][0] + random.uniform(-5, 5)
                    end_y = current['end'][1] + random.uniform(-5, 5)
                    
                    # Keep within grid
                    start_x = max(0, min(self.model.grid_width - 1, start_x))
                    start_y = max(0, min(self.model.grid_height - 1, start_y))
                    end_x = max(0, min(self.model.grid_width - 1, end_x))
                    end_y = max(0, min(self.model.grid_height - 1, end_y))
                    
                    proposal_data['route_corridor'] = {
                        'start': [start_x, start_y],
                        'end': [end_x, end_y],
                        'width': current['width']
                    }
                
            elif proposal_type == 'service':
                # Service quality proposals
                adjust_by = random.uniform(-5, 5)
                proposal_data['quality_target'] = max(50, min(100, self.quality_score + adjust_by))
                proposal_data['reliability_target'] = max(50, min(100, self.reliability + adjust_by))
            
            # Create the vote
            vote_id = f"vote_{self.unique_id}_{self.model.schedule.time}"
            
            vote_data = {
                'proposal_type': proposal_type,
                'proposal_data': proposal_data,
                'start_time': self.model.schedule.time,
                'end_time': self.model.schedule.time + 86400,  # 1 day voting period
                'status': 'active',
                'votes': {},
                'proposer': random.choice(list(self.fleet_token_holders.keys()))
            }
            
            self.governance_votes[vote_id] = vote_data
            
            # Auto-populate some votes for simulation purposes
            for holder_id, stake in self.fleet_token_holders.items():
                if random.random() < 0.7:  # 70% participation
                    vote_data['votes'][holder_id] = random.random() < 0.6  # 60% yes bias

    def _distribute_earnings(self):
        """Distribute earnings to token holders for fractional fleet providers"""
        if self.provider_type != "fractional_fleet" or not self.fleet_token_holders:
            return
            
        # Only distribute periodically
        if self.model.schedule.time % 86400 != 0:  # Once per day
            return
            
        if not hasattr(self, 'fleet_profits') or not self.fleet_profits:
            return
            
        # Calculate total profit to distribute
        total_profit = sum(record['profit'] for record in self.fleet_profits)
        
        if total_profit <= 0:
            return
            
        # Calculate total tokens
        total_tokens = sum(self.fleet_token_holders.values())
        
        # Distribute proportionally to token holders
        distributions = {}
        for holder_id, stake in self.fleet_token_holders.items():
            share = stake / total_tokens
            distribution = total_profit * share
            distributions[holder_id] = distribution
            
        # Record distributions
        if not hasattr(self, 'earnings_distributions'):
            self.earnings_distributions = []
            
        self.earnings_distributions.append({
            'time': self.model.schedule.time,
            'total_profit': total_profit,
            'distributions': distributions
        })
        
        # Clear profits after distribution
        self.fleet_profits = []
        
        self.logger.info(f"Distributed {total_profit} in profits to {len(self.fleet_token_holders)} token holders")

    def _update_learning_parameters(self):
        """
        Update learning parameters based on outcomes.
        """
        if len(self.win_rate_history) < 10:
            return  # Not enough data
            
        # Calculate recent win rate
        recent_win_rate = np.mean(self.win_rate_history[-10:])

        # Adjust price adjustment factor
        if recent_win_rate < 0.3:
            # Low win rate, become more aggressive in pricing
            self.price_adjustment_factor = max(0.1, self.price_adjustment_factor - 0.02)
        elif recent_win_rate > 0.7:
            # High win rate, can be less aggressive
            self.price_adjustment_factor = min(0.3, self.price_adjustment_factor + 0.02)
            
        # Update time decay factor
        # If win rate is high, can increase the time decay effect (wider spread between min/max prices)
        # If win rate is low, should reduce the time decay effect to be more competitive
        if recent_win_rate > 0.7:
            self.time_decay_factor = min(0.25, self.time_decay_factor + 0.01)
        elif recent_win_rate < 0.3:
            self.time_decay_factor = max(0.05, self.time_decay_factor - 0.01)
            
        # Learn about location preferences based on successful offers
        if self.provider_type == "individual" and len(self.service_history) >= 5:
            # Analyze recent successful services
            recent_services = self.service_history[-5:]
            
            # Extract origin/destination regions
            successful_regions = []
            for service in recent_services:
                nft_id = service['nft_id']
                if nft_id in self.issued_nfts:
                    nft = self.issued_nfts[nft_id]
                    if 'route' in nft['metadata']:
                        route = nft['metadata']['route']
                        if len(route) >= 2:
                            origin = route[0]
                            destination = route[-1]
                            
                            # Use grid regions for learning
                            region_size = min(self.model.grid_width, self.model.grid_height) // 5
                            
                            origin_region = {
                                "center": [
                                    (origin[0] // region_size) * region_size + region_size // 2,
                                    (origin[1] // region_size) * region_size + region_size // 2
                                ],
                                "radius": region_size // 2
                            }
                            
                            dest_region = {
                                "center": [
                                    (destination[0] // region_size) * region_size + region_size // 2,
                                    (destination[1] // region_size) * region_size + region_size // 2
                                ],
                                "radius": region_size // 2
                            }
                            
                            successful_regions.append(origin_region)
                            successful_regions.append(dest_region)
            
            # Only update if we have regions
            if successful_regions:
                # Count region frequencies
                region_counts = {}
                for region in successful_regions:
                    key = f"{region['center'][0]}_{region['center'][1]}"
                    region_counts[key] = region_counts.get(key, 0) + 1
                
                # Find most successful regions
                sorted_regions = sorted(region_counts.items(), key=lambda x: x[1], reverse=True)
                
                # Update preferred regions (occasionally)
                if random.random() < 0.3:  # 30% chance to update
                    new_preferred = []
                    for key, _ in sorted_regions[:3]:  # Top 3 regions
                        x, y = map(int, key.split('_'))
                        new_preferred.append({
                            "center": [x, y],
                            "radius": region_size // 2
                        })
                    
                    # Update preferences
                    if hasattr(self, 'location_preferences'):
                        self.location_preferences["preferred_regions"] = new_preferred
                    
        # Track competitor pricing
        market_data = None
        if hasattr(self.model, 'get_market_data'):
            market_data = self.model.get_market_data(self.mode_type)
            
        if market_data and 'avg_price' in market_data:
            if 'competition' not in self.__dict__:
                self.competition = {
                    'avg_price': [],
                    'last_update': 0
                }
                
            # Record every hour (not every step)
            if self.model.schedule.time - self.competition['last_update'] >= 3600:
                self.competition['avg_price'].append((self.model.schedule.time, market_data['avg_price']))
                self.competition['last_update'] = self.model.schedule.time
                
                # Limit history
                if len(self.competition['avg_price']) > 24:
                    self.competition['avg_price'] = self.competition['avg_price'][-24:]

    def _update_pricing_strategy(self):
        """
        Update pricing strategy based on market conditions and learning.
        """
        # Only update occasionally to avoid rapid oscillations
        if self.model.schedule.time % 360 != 0:  # Update every ~6 simulation hours
            return
            
        # Skip if not enough history
        if len(self.price_history) < 5 or len(self.win_rate_history) < 5:
            return
            
        # Analyze recent performance
        recent_win_rate = np.mean(self.win_rate_history[-10:]) if len(self.win_rate_history) >= 10 else 0.5
        revenue_trend = self._analyze_revenue_trend()
        
        # For fractional fleets, less aggressive adaptation
        if self.provider_type == "fractional_fleet":
            # Public transport typically has more stable pricing
            if revenue_trend == 'decreasing' and recent_win_rate < 0.3:
                # Only make modest adjustments
                self.base_price *= 0.98
                self.logger.info(f"Fractional fleet {self.unique_id} decreasing base price to {self.base_price}")
            elif revenue_trend == 'increasing' and recent_win_rate > 0.8:
                # Only make modest increases
                self.base_price *= 1.02
                self.logger.info(f"Fractional fleet {self.unique_id} increasing base price to {self.base_price}")
        else:
            # Individual providers adapt more aggressively
            if revenue_trend == 'decreasing' and recent_win_rate < 0.4:
                # Revenue decreasing and low win rate - prices likely too high
                self.base_price *= 0.95
                self.logger.info(f"Decreasing base price to {self.base_price} due to poor performance")
            elif revenue_trend == 'increasing' and recent_win_rate > 0.7:
                # Revenue increasing and high win rate - could charge more
                self.base_price *= 1.05
                self.logger.info(f"Increasing base price to {self.base_price} due to strong performance")
                
            # Consider switching strategies if current one isn't working well
            if recent_win_rate < 0.2 and len(self.win_rate_history) > 20:
                # Very poor performance over extended period - try a new strategy
                current_strategy = self.price_strategy
                
                # Choose a different strategy
                strategies = ["value_based", "cost_plus", "competitor_based", 
                            "dynamic_surge", "time_optimized", "loyalty_focused"]
                strategies.remove(current_strategy)
                new_strategy = random.choice(strategies)
                
                self.price_strategy = new_strategy
                self.logger.info(f"Changing pricing strategy from {current_strategy} to {new_strategy} due to poor performance")
                
                # Reset some parameters when changing strategy
                self.price_adjustment_factor = random.uniform(0.15, 0.25)
                
                # Apply strategy-specific adjustments
                if new_strategy == "loyalty_focused":
                    # Lower price floor for loyalty strategy
                    self.price_floor = max(0.4, self.price_floor - 0.1)
                elif new_strategy == "dynamic_surge":
                    # More volatile pricing for surge strategy
                    self.surge_sensitivity = random.uniform(1.0, 1.3)
                elif new_strategy == "competitor_based":
                    # Slightly undercut market
                    self.base_price *= 0.97
                
            # Fine-tune parameters within current strategy
            elif 0.3 <= recent_win_rate <= 0.7:
                # Moderate performance - fine tune current strategy
                
                if self.price_strategy == "value_based":
                    # Adjust premium based on quality score
                    quality_premium = (self.quality_score - 70) / 100
                    self.price_ceiling = 1.8 + quality_premium
                    
                elif self.price_strategy == "time_optimized":
                    # Increase time sensitivity
                    self.time_decay_factor = min(0.25, self.time_decay_factor + 0.01)
                    
                elif self.price_strategy == "dynamic_surge":
                    # Adjust surge sensitivity based on market conditions
                    is_peak = self.model.check_is_peak() if hasattr(self.model, 'check_is_peak') else False
                    if is_peak:
                        self.surge_sensitivity = min(1.5, self.surge_sensitivity + 0.05)
                    else:
                        self.surge_sensitivity = max(0.8, self.surge_sensitivity - 0.05)
                        
            # Top performance - enhance current strategy
            elif recent_win_rate > 0.7:
                if self.price_strategy == "loyalty_focused":
                    # Increase price floor now that we have good customer base
                    self.price_floor = min(0.7, self.price_floor + 0.02)
                    
                elif self.price_strategy == "cost_plus":
                    # Can increase markup
                    self.price_ceiling = min(2.2, self.price_ceiling + 0.05)

    def _analyze_revenue_trend(self):
        """
        Analyze revenue trend from recent service history to guide pricing decisions.
        
        Returns:
            str: 'increasing', 'stable', or 'decreasing'
        """
        # Need enough history to determine a trend
        if len(self.service_history) < 5:
            return 'stable'
            
        # Get recent services
        recent_services = self.service_history[-10:]
        
        # Calculate total revenue in time windows
        if len(recent_services) < 6:
            # Not enough data for two windows, assume stable
            return 'stable'
            
        # Divide into recent and earlier window
        midpoint = len(recent_services) // 2
        recent_window = recent_services[midpoint:]
        earlier_window = recent_services[:midpoint]
        
        # Calculate revenue in each window
        recent_revenue = sum(s.get('price', 0) for s in recent_window)
        earlier_revenue = sum(s.get('price', 0) for s in earlier_window)
        
        # Calculate the difference
        if recent_revenue > earlier_revenue * 1.1:  # 10% increase
            return 'increasing'
        elif recent_revenue < earlier_revenue * 0.9:  # 10% decrease
            return 'decreasing'
        else:
            return 'stable'

    def _update_offer_time_decay(self):
        """
        Update prices of active offers based on time decay.
        
        This implements time-sensitive pricing where offers get cheaper
        as the service time approaches (if not booked in advance).
        """
        current_time = self.model.schedule.time
        
        for offer_id, offer in list(self.active_bids.items()):
            if offer['status'] != 'pending':
                continue
                
            # Only update price if using dynamic pricing
            if not self.dynamic_pricing_enabled:
                continue
                
            # Calculate time left until service
            start_time = offer['start_time']
            time_to_service = start_time - current_time
            
            # Original price
            original_price = offer.get('original_price', offer['price'])
            if 'original_price' not in offer:
                offer['original_price'] = original_price
            
            # Calculate new price using time decay
            time_factor = self._calculate_time_decay_factor(time_to_service)
            new_price = original_price * time_factor
            
            # Ensure price doesn't go below floor
            min_price = original_price * self.price_floor
            new_price = max(min_price, new_price)
            
            # Update offer price if changed by more than 1%
            if abs(new_price - offer['price']) / offer['price'] > 0.01:
                old_price = offer['price']
                offer['price'] = new_price
                
                # Log significant price changes
                self.logger.info(f"Updated offer {offer_id} price from {old_price} to {new_price} due to time decay")
                
                # If connected to marketplace, update listing
                if hasattr(self.model, 'marketplace') and offer.get('listing_id'):
                    self.model.marketplace.update_listing_price(offer['listing_id'], new_price)

    def _register_with_blockchain(self):
        """Register the provider on the blockchain with proper attributes."""
        self.logger.info(f"Registering provider {self.unique_id} ({self.name}) with blockchain")
        
        # Include appropriate agent attributes that match blockchain expectations
        success, self.blockchain_address = self.blockchain_interface.register_provider(self)
        
        if success:
            self.logger.info(f"Provider {self.unique_id} registered at address {self.blockchain_address}")
        else:
            self.logger.error(f"Failed to register provider {self.unique_id} with blockchain")

    def add_liquidity_to_amm(self, origin, destination, amount):
        """
        Add liquidity to AMM pool for a specific route.
        
        Args:
            origin: Origin coordinates [x, y]
            destination: Destination coordinates [x, y]
            amount: Amount of liquidity to add
            
        Returns:
            tuple: (success, lp_tokens_received)
        """
        # Check if AMM component is available in model
        if not hasattr(self.model, 'amm') or not self.model.amm:
            self.logger.error("AMM component not available in model")
            return False, 0
            
        # Check if provider has enough capacity to participate in AMM
        if self.available_capacity < self.capacity * 0.3:  # Require at least 30% free capacity
            self.logger.info(f"Not enough capacity to contribute to AMM pool: {self.available_capacity}/{self.capacity}")
            return False, 0
            
        # Generate route key in the format used by AMM
        if hasattr(self.model.amm, '_get_route_key'):
            route_key = self.model.amm._get_route_key(origin, destination)
        else:
            # Fallback if method not available - try to match expected format
            origin_tuple = tuple(origin)
            destination_tuple = tuple(destination)
            current_time = self.model.schedule.time
            route_key = f"{origin_tuple}_{destination_tuple}_{current_time}"
            
        # Calculate appropriate token amounts based on baseline price
        distance = self._calculate_distance(origin, destination)
        base_price = self.base_price * distance
        
        # For AMM, we need to provide both service tokens and payment tokens
        # in the correct ratio expected by the pool
        
        # Check if pool already exists
        pool_exists = False
        service_tokens = 0
        payment_tokens = 0
        
        if hasattr(self.model.amm, 'liquidity_pools') and route_key in self.model.amm.liquidity_pools:
            pool = self.model.amm.liquidity_pools[route_key]
            pool_exists = True
            
            # Calculate tokens to match current pool ratio
            if pool['service_tokens'] > 0:
                current_price = pool['payment_tokens'] / pool['service_tokens']
                
                # Contribute proportionally to maintain price
                service_tokens = amount / current_price
                payment_tokens = amount
            else:
                # Empty pool, use base calculation
                service_tokens = math.sqrt(amount / base_price)
                payment_tokens = service_tokens * base_price
        else:
            # New pool, calculate initial token amounts
            service_tokens = math.sqrt(amount / base_price)
            payment_tokens = service_tokens * base_price
        
        # Add liquidity to pool
        success, lp_tokens = self.model.amm.add_liquidity(
            route_key, 
            service_tokens, 
            payment_tokens, 
            self.unique_id
        )
        
        if success:
            self.logger.info(f"Added liquidity to AMM pool for route {route_key}: {service_tokens} service tokens, {payment_tokens} payment tokens")
            
            # Record participation in AMM
            if not hasattr(self, 'amm_positions'):
                self.amm_positions = {}
                
            if route_key not in self.amm_positions:
                self.amm_positions[route_key] = {
                    'lp_tokens': 0,
                    'service_tokens': 0,
                    'payment_tokens': 0
                }
                
            # Update position
            self.amm_positions[route_key]['lp_tokens'] += lp_tokens
            self.amm_positions[route_key]['service_tokens'] += service_tokens
            self.amm_positions[route_key]['payment_tokens'] += payment_tokens
            
            # Reduce available capacity to reflect AMM commitment
            capacity_reduction = min(1, max(1, int(service_tokens / 10)))
            self.available_capacity = max(0, self.available_capacity - capacity_reduction)
            
            return True, lp_tokens
        else:
            self.logger.error(f"Failed to add liquidity to AMM pool for route {route_key}")
            return False, 0

    def remove_liquidity_from_amm(self, route_key, lp_token_amount=None):
        """
        Remove liquidity from AMM pool.
        
        Args:
            route_key: Route key for the pool
            lp_token_amount: Amount of LP tokens to burn (None for all)
            
        Returns:
            tuple: (success, service_tokens, payment_tokens)
        """
        # Check if AMM component is available
        if not hasattr(self.model, 'amm') or not self.model.amm:
            self.logger.error("AMM component not available in model")
            return False, 0, 0
            
        # Check if provider has position in this pool
        if not hasattr(self, 'amm_positions') or route_key not in self.amm_positions:
            self.logger.error(f"No position in AMM pool {route_key}")
            return False, 0, 0
            
        # Use all LP tokens if amount not specified
        if lp_token_amount is None:
            lp_token_amount = self.amm_positions[route_key]['lp_tokens']
            
        # Check if pool exists
        if not hasattr(self.model.amm, 'liquidity_pools') or route_key not in self.model.amm.liquidity_pools:
            self.logger.error(f"AMM pool {route_key} does not exist")
            return False, 0, 0
            
        # Remove liquidity from pool
        success, service_tokens, payment_tokens = self.model.amm.remove_liquidity(
            route_key,
            lp_token_amount,
            self.unique_id
        )
        
        if success:
            self.logger.info(f"Removed liquidity from AMM pool {route_key}: {service_tokens} service tokens, {payment_tokens} payment tokens")
            
            # Update position
            self.amm_positions[route_key]['lp_tokens'] -= lp_token_amount
            
            # If all tokens removed, remove from positions
            if self.amm_positions[route_key]['lp_tokens'] <= 0:
                del self.amm_positions[route_key]
                
            # Restore some capacity
            capacity_increase = min(1, max(1, int(service_tokens / 10)))
            self.available_capacity = min(self.capacity, self.available_capacity + capacity_increase)
            
            return True, service_tokens, payment_tokens
        else:
            self.logger.error(f"Failed to remove liquidity from AMM pool {route_key}")
            return False, 0, 0

    def check_provider_arbitrage_opportunities(self):
        """
        Check for arbitrage opportunities between direct market and AMM pools.
        This is a provider-specific strategy to maximize profitability.
        
        Returns:
            list: Identified arbitrage opportunities
        """
        opportunities = []
        
        # Check if AMM component is available
        if not hasattr(self.model, 'amm') or not self.model.amm:
            return opportunities
            
        # Check if marketplace is available
        if not hasattr(self.model, 'marketplace'):
            return opportunities
            
        # Look at active routes where this provider can service
        for route_key, pool in self.model.amm.liquidity_pools.items():
            # Skip if not a popular route
            if route_key not in self.model.popular_routes:
                continue
                
            # Get origin and destination from route key
            try:
                parts = route_key.split('_')
                if len(parts) >= 3:  # Includes time window
                    origin_str = parts[0] + '_' + parts[1]
                    dest_str = parts[2] + '_' + parts[3]
                    origin = eval(origin_str)
                    destination = eval(dest_str)
                    
                    # Check if provider can service this route
                    if not self._can_service_route(origin, destination):
                        continue
                        
                    # Compare AMM price with direct market price
                    amm_price = pool['current_price']
                    
                    # Get market price for similar route
                    market_price = self.model.marketplace.get_market_price({
                        'origin': origin,
                        'destination': destination,
                        'service_time': self.model.schedule.time + 3600  # 1 hour in future
                    })
                    
                    if not market_price:
                        continue
                        
                    # Check for significant price difference (>10%)
                    price_diff = abs(market_price - amm_price) / market_price
                    
                    if price_diff > 0.1:
                        # Found an arbitrage opportunity
                        opportunity = {
                            'route_key': route_key,
                            'origin': origin,
                            'destination': destination,
                            'amm_price': amm_price,
                            'market_price': market_price,
                            'price_diff': price_diff,
                            'action': 'buy_amm_sell_market' if amm_price < market_price else 'buy_market_sell_amm'
                        }
                        
                        opportunities.append(opportunity)
            except:
                # Skip if parsing fails
                continue
                
        # Sort opportunities by potential profit
        opportunities.sort(key=lambda x: x['price_diff'], reverse=True)
        
        return opportunities

    def execute_arbitrage(self, opportunity):
        """
        Execute an arbitrage opportunity between AMM and direct market.
        
        Args:
            opportunity: Arbitrage opportunity dict
            
        Returns:
            bool: Success status
        """
        route_key = opportunity['route_key']
        action = opportunity['action']
        
        if action == 'buy_amm_sell_market':
            # Buy from AMM and sell on market
            
            # Calculate amount to buy based on available capacity
            amount = min(5, self.available_capacity) * opportunity['amm_price']
            
            # Get quote from AMM
            quote = self.model.amm.get_quote(
                opportunity['origin'],
                opportunity['destination'],
                amount,
                is_buy=True
            )
            
            if not quote:
                self.logger.error(f"Failed to get AMM quote for route {route_key}")
                return False
                
            # Execute swap to buy from AMM
            success, swap_details = self.model.amm.execute_swap(quote, self.unique_id)
            
            if not success:
                self.logger.error(f"Failed to execute AMM swap for route {route_key}")
                return False
                
            # If successful, list on marketplace at higher price
            if 'nft_id' in swap_details:
                nft_id = swap_details['nft_id']
                sale_price = opportunity['market_price'] * 0.98  # Slightly below market price for quicker sale
                
                # Create time parameters for dynamic pricing
                time_parameters = {
                    'initial_price': sale_price,
                    'final_price': sale_price * 0.7,  # Allow up to 30% discount as time approaches
                    'decay_duration': 3600  # 1 hour decay
                }
                
                list_success = self.blockchain_interface.list_nft_for_sale(
                    nft_id,
                    sale_price,
                    time_parameters
                )
                
                if list_success:
                    self.logger.info(f"Successfully arbitraged route {route_key}: Bought from AMM at {quote['price']}, listed on market at {sale_price}")
                    return True
                else:
                    self.logger.error(f"Failed to list NFT {nft_id} on marketplace")
                    return False
        
        elif action == 'buy_market_sell_amm':
            # Find suitable NFT on marketplace
            market_results = self.blockchain_interface.search_nft_market({
                'origin_area': [opportunity['origin'][0], opportunity['origin'][1], 5],
                'dest_area': [opportunity['destination'][0], opportunity['destination'][1], 5],
                'max_price': opportunity['market_price'] * 0.9  # Look for at least 10% below market price
            })
            
            if not market_results:
                self.logger.error(f"No suitable NFTs found on marketplace for route {route_key}")
                return False
                
            # Find the cheapest NFT
            cheapest_nft = min(market_results, key=lambda x: x['price'])
            
            # Purchase the NFT
            purchase_success = self.blockchain_interface.purchase_nft(
                cheapest_nft['token_id'],
                self.unique_id
            )
            
            if not purchase_success:
                self.logger.error(f"Failed to purchase NFT {cheapest_nft['token_id']} from marketplace")
                return False
                
            # Add NFT to owned list
            self.issued_nfts[cheapest_nft['token_id']] = {
                'metadata': cheapest_nft,
                'status': 'active',
                'request_id': 0,
                'offer_id': 0,
                'commuter_id': self.unique_id,
                'creation_time': self.model.schedule.time
            }
            
            # Sell to AMM at higher price
            # This is a simplified version - in reality would need to tokenize the purchased NFT
            # and then sell the service tokens to the AMM
            
            # For simulation purposes, assume the sale is successful
            self.logger.info(f"Successfully arbitraged route {route_key}: Bought from market at {cheapest_nft['price']}, will sell to AMM at {opportunity['amm_price']}")
            
            return True
            
        return False

    def check_request_status(self, request_id):
        """
        Check the status of a previously submitted request.
        
        Args:
            request_id: ID of the request to check
            
        Returns:
            str: Current status of the request
        """
        # Check blockchain for request status
        return self.blockchain_interface.check_request_status(request_id)

    def check_offer_status(self, offer_id):
        """
        Check the status of a previously submitted offer.
        
        Args:
            offer_id: ID of the offer to check
            
        Returns:
            str: Current status of the offer
        """
        return self.blockchain_interface.check_offer_status(offer_id)

    def update_commuter_satisfaction(self, commuter_id, satisfaction_score):
        """
        Update satisfaction score for a specific commuter.
        
        Args:
            commuter_id: ID of the commuter
            satisfaction_score: Score between -1 and 1
        """
        if commuter_id not in self.customer_satisfaction:
            self.customer_satisfaction[commuter_id] = []
            
        # Add new score
        self.customer_satisfaction[commuter_id].append((self.model.schedule.time, satisfaction_score))
        
        # Limit history
        if len(self.customer_satisfaction[commuter_id]) > 10:
            self.customer_satisfaction[commuter_id] = self.customer_satisfaction[commuter_id][-10:]
            
        # Update preferred/blacklisted status
        avg_score = sum(score for _, score in self.customer_satisfaction[commuter_id]) / len(self.customer_satisfaction[commuter_id])
        
        if avg_score > 0.7:
            # Add to preferred customers
            if commuter_id not in self.preferred_customers:
                self.preferred_customers.add(commuter_id)
                
            # Remove from blacklist if present
            if commuter_id in self.blacklisted_customers:
                self.blacklisted_customers.remove(commuter_id)
                
        elif avg_score < -0.3:
            # Add to blacklisted customers
            if commuter_id not in self.blacklisted_customers:
                self.blacklisted_customers.add(commuter_id)
                
            # Remove from preferred if present
            if commuter_id in self.preferred_customers:
                self.preferred_customers.remove(commuter_id)

    def modify_working_hours(self, day_of_week, new_hours):
        """
        Modify working hours for a specific day, representing provider's evolving preferences.
        
        Args:
            day_of_week: Day to modify (0=Monday, 6=Sunday)
            new_hours: List of (start_hour, end_hour) tuples
        """
        if day_of_week < 0 or day_of_week > 6:
            self.logger.error(f"Invalid day of week: {day_of_week}")
            return False
            
        # Validate hours
        for start, end in new_hours:
            if start < 0 or start > 23 or end < 0 or end > 23 or start >= end:
                self.logger.error(f"Invalid hours: {start}-{end}")
                return False
                
        # Update working hours
        self.working_hours[day_of_week] = new_hours
        self.logger.info(f"Updated working hours for day {day_of_week}: {new_hours}")
        return True

    def modify_location_preferences(self, preferred_regions=None, avoid_regions=None, max_distance=None):
        """
        Update provider location preferences based on experience and market conditions.
        
        Args:
            preferred_regions: New list of preferred regions
            avoid_regions: New list of regions to avoid
            max_distance: New maximum service distance from home
        """
        if not hasattr(self, 'location_preferences'):
            self.location_preferences = {}
            
        if preferred_regions is not None:
            self.location_preferences['preferred_regions'] = preferred_regions
            
        if avoid_regions is not None:
            self.location_preferences['avoid_regions'] = avoid_regions
            
        if max_distance is not None:
            self.location_preferences['max_distance_from_home'] = max_distance
            
        self.logger.info(f"Updated location preferences for provider {self.unique_id}")

    def get_analytics(self):
        """
        Get provider analytics for dashboard/monitoring.
        
        Returns:
            dict: Provider analytics
        """
        # Calculate various metrics
        completed_services = len([s for s in self.service_history if 'completion_time' in s])
        
        # Calculate revenue metrics if we have completed services
        total_revenue = 0
        avg_price = 0
        recent_revenue = 0
        
        if completed_services > 0:
            # Total revenue
            total_revenue = sum(s.get('price', 0) for s in self.service_history)
            avg_price = total_revenue / completed_services
            
            # Recent revenue (last 24 hours)
            current_time = self.model.schedule.time
            recent_services = [s for s in self.service_history if current_time - s.get('completion_time', 0) < 86400]
            recent_revenue = sum(s.get('price', 0) for s in recent_services)
        
        # Calculate reputation metrics
        reputation_trend = 'stable'
        if len(self.reputation_history) >= 5:
            recent_reps = [r for _, r in self.reputation_history[-5:]]
            if recent_reps[-1] > recent_reps[0] * 1.05:
                reputation_trend = 'improving'
            elif recent_reps[-1] < recent_reps[0] * 0.95:
                reputation_trend = 'declining'
        
        # Build analytics dictionary
        analytics = {
            'provider_id': self.unique_id,
            'name': self.name,
            'provider_type': self.provider_type,
            'mode_type': self.mode_type,
            'capacity': {
                'total': self.capacity,
                'available': self.available_capacity,
                'utilization': (self.capacity - self.available_capacity) / self.capacity
            },
            'pricing': {
                'base_price': self.base_price,
                'strategy': self.price_strategy,
                'time_decay_factor': self.time_decay_factor,
                'price_floor': self.price_floor,
                'price_ceiling': self.price_ceiling
            },
            'performance': {
                'completed_services': completed_services,
                'total_revenue': total_revenue,
                'avg_price': avg_price,
                'recent_revenue': recent_revenue,
                'win_rate': sum(self.win_rate_history[-10:]) / len(self.win_rate_history[-10:]) if len(self.win_rate_history) >= 10 else 0
            },
            'reputation': {
                'score': self.reputation_score,
                'reliability': self.reliability,
                'quality_score': self.quality_score,
                'trend': reputation_trend
            },
            'customer_relationships': {
                'preferred_customers': len(self.preferred_customers),
                'blacklisted_customers': len(self.blacklisted_customers)
            }
        }
        
        # Add fractional fleet specific metrics
        if self.provider_type == 'fractional_fleet':
            fleet_metrics = {
                'token_holders': len(self.fleet_token_holders),
                'governance_proposals': len(self.governance_votes),
                'active_proposals': sum(1 for v in self.governance_votes.values() if v['status'] == 'active')
            }
            
            # Add earnings distribution metrics if available
            if hasattr(self, 'earnings_distributions') and self.earnings_distributions:
                last_distribution = self.earnings_distributions[-1]
                fleet_metrics['last_distribution'] = {
                    'time': last_distribution['time'],
                    'total_profit': last_distribution['total_profit'],
                    'holder_count': len(last_distribution['distributions'])
                }
                
            analytics['fleet'] = fleet_metrics
        
        # Add AMM participation metrics
        if hasattr(self, 'amm_positions') and self.amm_positions:
            amm_metrics = {
                'pools_participated': len(self.amm_positions),
                'total_lp_tokens': sum(pos.get('lp_tokens', 0) for pos in self.amm_positions.values()),
                'pool_details': [{
                    'route_key': k,
                    'lp_tokens': v.get('lp_tokens', 0),
                    'service_tokens': v.get('service_tokens', 0),
                    'payment_tokens': v.get('payment_tokens', 0)
                } for k, v in self.amm_positions.items()]
            }
            
            analytics['amm_participation'] = amm_metrics
        
        return analytics