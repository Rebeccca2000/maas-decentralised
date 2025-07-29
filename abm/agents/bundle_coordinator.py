"""
Bundle Coordinator for Decentralized MaaS

This module coordinates the creation, optimization, and execution of 
mobility service bundles across multiple providers, supporting complex
multi-modal journeys with off-chain composition and blockchain settlement.
"""

import uuid
import math
import logging
import random
from typing import List, Dict, Any, Tuple, Optional

class BundleCoordinator:
    """
    Coordinates the creation and execution of mobility service bundles
    by managing the lifecycle of bundles, collecting offers from providers,
    and optimizing bundle combinations.
    """
    
    def __init__(self, model, blockchain_interface):
        """
        Initialize the bundle coordinator.
        
        Args:
            model: The ABM model
            blockchain_interface: Interface to the blockchain
        """
        self.model = model
        self.blockchain_interface = blockchain_interface
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("BundleCoordinator")
        
        # Bundle management
        self.bundle_registry = {}  # Bundle ID -> Bundle details
        self.bundle_offers = {}    # Bundle ID -> Segment ID -> Offers
        self.active_bundles = {}   # Currently active bundles
        self.completed_bundles = {}  # Completed bundles
        
        # Bundle pattern recognition
        self.popular_routes = {}   # Track popular bundled routes
        self.successful_patterns = {}  # Track successful bundle patterns
        
        self.logger.info("Bundle Coordinator initialized")
    
    def identify_complementary_services(self, origin, destination, start_time, commuter_preferences=None):
        """
        Identify potential service combinations that form a valid bundle
        from origin to destination.
        
        Args:
            origin: Origin coordinates [x, y]
            destination: Destination coordinates [x, y]
            start_time: Requested start time
            commuter_preferences: Optional preferences
            
        Returns:
            List of route segments that could form bundles
        """
        # Find transit stations near origin and destination
        stations_near_origin = self._find_nearby_stations(origin, 10)  # 10 unit radius
        stations_near_dest = self._find_nearby_stations(destination, 10)
        
        potential_bundles = []
        
        # Check direct route (single segment) if relatively close
        direct_distance = self._calculate_distance(origin, destination)
        if direct_distance < 20:  # Short enough for direct service
            direct_bundle = [{
                'segment_id': f"segment_direct_{uuid.uuid4()}",
                'segment_type': 'direct',
                'origin': origin,
                'destination': destination,
                'preferred_modes': ['car', 'bike'],
                'start_time': start_time,
                'estimated_duration': self._estimate_segment_duration(origin, destination, 'direct')
            }]
            potential_bundles.append(direct_bundle)
        
        # For each pair of stations, create a potential multi-segment bundle
        for origin_station in stations_near_origin:
            for dest_station in stations_near_dest:
                # Skip if stations are the same or too close
                if origin_station['station_id'] == dest_station['station_id']:
                    continue
                
                # Calculate timings
                first_mile_duration = self._estimate_segment_duration(
                    origin, origin_station['location'], 'first_mile')
                
                main_journey_duration = self._estimate_segment_duration(
                    origin_station['location'], dest_station['location'], 'main_journey')
                
                last_mile_duration = self._estimate_segment_duration(
                    dest_station['location'], destination, 'last_mile')
                
                # Create segment start times
                first_mile_start = start_time
                main_journey_start = first_mile_start + first_mile_duration
                last_mile_start = main_journey_start + main_journey_duration
                
                # Create a 3-segment bundle: first mile, main journey, last mile
                bundle_segments = [
                    {
                        'segment_id': f"segment_first_{uuid.uuid4()}",
                        'segment_type': 'first_mile',
                        'origin': origin,
                        'destination': origin_station['location'],
                        'preferred_modes': ['bike', 'car'],
                        'start_time': first_mile_start,
                        'estimated_duration': first_mile_duration,
                        'station_id': origin_station['station_id'],
                        'station_type': origin_station['type']
                    },
                    {
                        'segment_id': f"segment_main_{uuid.uuid4()}",
                        'segment_type': 'main_journey',
                        'origin': origin_station['location'],
                        'destination': dest_station['location'],
                        'preferred_modes': ['bus', 'train'],
                        'start_time': main_journey_start,
                        'estimated_duration': main_journey_duration,
                        'origin_station_id': origin_station['station_id'],
                        'dest_station_id': dest_station['station_id']
                    },
                    {
                        'segment_id': f"segment_last_{uuid.uuid4()}",
                        'segment_type': 'last_mile',
                        'origin': dest_station['location'],
                        'destination': destination,
                        'preferred_modes': ['bike', 'car', 'walk'],
                        'start_time': last_mile_start,
                        'estimated_duration': last_mile_duration,
                        'station_id': dest_station['station_id'],
                        'station_type': dest_station['type']
                    }
                ]
                
                # Apply commuter preferences if provided
                if commuter_preferences:
                    self._apply_preferences(bundle_segments, commuter_preferences)
                
                potential_bundles.append(bundle_segments)
        
        # If no public transport options, create a 2-segment bundle (e.g., bike then car)
        if not stations_near_origin and not stations_near_dest and direct_distance > 20:
            # Find midpoint
            midpoint = [
                (origin[0] + destination[0]) / 2,
                (origin[1] + destination[1]) / 2
            ]
            
            # Calculate durations
            first_half_duration = self._estimate_segment_duration(origin, midpoint, 'first_half')
            second_half_duration = self._estimate_segment_duration(midpoint, destination, 'second_half')
            
            # Create segment start times
            first_half_start = start_time
            second_half_start = first_half_start + first_half_duration
            
            two_segment_bundle = [
                {
                    'segment_id': f"segment_first_half_{uuid.uuid4()}",
                    'segment_type': 'first_half',
                    'origin': origin,
                    'destination': midpoint,
                    'preferred_modes': ['bike', 'car'],
                    'start_time': first_half_start,
                    'estimated_duration': first_half_duration
                },
                {
                    'segment_id': f"segment_second_half_{uuid.uuid4()}",
                    'segment_type': 'second_half',
                    'origin': midpoint,
                    'destination': destination,
                    'preferred_modes': ['car'],
                    'start_time': second_half_start,
                    'estimated_duration': second_half_duration
                }
            ]
            
            potential_bundles.append(two_segment_bundle)
        
        # Log results
        self.logger.info(f"Identified {len(potential_bundles)} potential bundle options")
        return potential_bundles
    
    def _find_nearby_stations(self, location, radius):
        """
        Find transit stations near a location.
        
        Args:
            location: [x, y] coordinates
            radius: Search radius
            
        Returns:
            List of nearby stations with metadata
        """
        nearby_stations = []
        
        # Check if model has stations
        if not hasattr(self.model, 'stations') or not self.model.stations:
            return nearby_stations
        
        # Process each station type
        for station_type, stations in self.model.stations.items():
            for station_id, station_location in stations.items():
                # Calculate distance
                distance = self._calculate_distance(location, station_location)
                
                if distance <= radius:
                    nearby_stations.append({
                        'station_id': station_id,
                        'type': station_type,
                        'location': station_location,
                        'distance': distance
                    })
        
        # Sort by distance
        nearby_stations.sort(key=lambda x: x['distance'])
        
        return nearby_stations
    
    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt(
            (point2[0] - point1[0])**2 +
            (point2[1] - point1[1])**2
        )
    
    def _estimate_segment_duration(self, origin, destination, segment_type):
        """
        Estimate duration for a segment.
        
        Args:
            origin: Origin coordinates
            destination: Destination coordinates
            segment_type: Type of segment
            
        Returns:
            Estimated duration in simulation ticks
        """
        # Calculate distance
        distance = self._calculate_distance(origin, destination)
        
        # Base speed depends on segment type (units per tick)
        if segment_type == 'main_journey':
            # Public transport is faster for main journeys
            base_speed = 1.2
        elif segment_type in ['first_mile', 'last_mile']:
            # First/last mile is typically slower
            base_speed = 0.8
        elif segment_type == 'direct':
            # Direct journeys (e.g., by car)
            base_speed = 1.0
        else:
            # Default for other types
            base_speed = 0.9
        
        # Calculate raw duration
        raw_duration = distance / base_speed
        
        # Add buffer time and transfer time
        buffer_factor = 1.2  # 20% buffer
        if segment_type in ['first_mile', 'last_mile']:
            # Add transfer time at stations
            transfer_time = 2  # 2 ticks for transfers
            return (raw_duration * buffer_factor) + transfer_time
        else:
            return raw_duration * buffer_factor
    
    def _apply_preferences(self, bundle_segments, preferences):
        """
        Apply commuter preferences to bundle segments.
        
        Args:
            bundle_segments: List of segments to modify
            preferences: Dict of commuter preferences
        """
        # Preferred modes
        if 'preferred_modes' in preferences:
            for segment in bundle_segments:
                # Prioritize preferred modes but keep others as fallback
                modes = segment['preferred_modes']
                preferred = [m for m in preferences['preferred_modes'] if m in modes]
                
                if preferred:
                    # Put preferred modes first
                    other_modes = [m for m in modes if m not in preferred]
                    segment['preferred_modes'] = preferred + other_modes
        
        # Comfort preference
        if 'comfort_preference' in preferences and preferences['comfort_preference'] > 0.7:
            # High comfort preference - adjust mode preferences
            for segment in bundle_segments:
                if 'car' in segment['preferred_modes']:
                    # Prioritize car for comfort-focused travelers
                    segment['preferred_modes'] = ['car'] + [m for m in segment['preferred_modes'] if m != 'car']
    
    def create_bundle_request(self, commuter_id, origin, destination, start_time, preferences=None):
        """
        Create a request for a service bundle.
        
        Args:
            commuter_id: ID of the commuter
            origin: Origin coordinates
            destination: Destination coordinates
            start_time: Requested start time
            preferences: Optional commuter preferences
            
        Returns:
            Bundle ID if successful, None otherwise
        """
        # Generate unique bundle ID
        bundle_id = str(uuid.uuid4())
        
        # Create basic bundle request
        bundle_request = {
            'bundle_id': bundle_id,
            'commuter_id': commuter_id,
            'origin': origin,
            'destination': destination,
            'start_time': start_time,
            'preferences': preferences,
            'status': 'initiated',
            'creation_time': self.model.schedule.time,
            'max_price': self._estimate_max_price(origin, destination),
            'total_distance': self._calculate_distance(origin, destination)
        }
        
        # Generate potential bundle options
        potential_bundles = self.identify_complementary_services(
            origin, destination, start_time, preferences)
        
        if not potential_bundles:
            self.logger.warning(f"No viable bundle options for commuter {commuter_id}")
            return None
        
        # Store potential options
        bundle_request['bundle_options'] = potential_bundles
        bundle_request['status'] = 'seeking_offers'
        
        # Register bundle
        self.bundle_registry[bundle_id] = bundle_request
        
        self.logger.info(f"Created bundle request {bundle_id} for commuter {commuter_id} with {len(potential_bundles)} options")
        return bundle_id
    
    def _estimate_max_price(self, origin, destination):
        """
        Estimate maximum reasonable price for a bundle.
        
        Args:
            origin: Origin coordinates
            destination: Destination coordinates
            
        Returns:
            Maximum acceptable price
        """
        # Calculate direct distance
        distance = self._calculate_distance(origin, destination)
        
        # Base price per unit distance
        base_price = 2.0
        
        # Apply non-linear scaling for longer distances
        if distance > 20:
            # Discount for longer trips
            return base_price * (20 + (distance - 20) * 0.8)
        else:
            return base_price * distance
    
    def collect_offers_for_bundle(self, bundle_id):
        """
        Collect offers from providers for each segment of the bundle.
        
        Args:
            bundle_id: Bundle ID
            
        Returns:
            Dictionary of offers by segment if successful, None otherwise
        """
        if bundle_id not in self.bundle_registry:
            self.logger.warning(f"Bundle {bundle_id} not found")
            return None
        
        bundle = self.bundle_registry[bundle_id]
        
        if bundle['status'] != 'seeking_offers':
            self.logger.warning(f"Bundle {bundle_id} not in seeking offers state")
            return None
        
        # Initialize offers storage
        if bundle_id not in self.bundle_offers:
            self.bundle_offers[bundle_id] = {}
        
        all_bundle_offers = {}
        
        # Process each bundle option
        for option_index, bundle_option in enumerate(bundle['bundle_options']):
            option_offers = {}
            
            # For each segment, collect offers from suitable providers
            for segment in bundle_option:
                segment_id = segment['segment_id']
                
                # Skip if we already have offers for this segment
                if segment_id in self.bundle_offers.get(bundle_id, {}):
                    option_offers[segment_id] = self.bundle_offers[bundle_id][segment_id]
                    continue
                
                # Find providers suitable for this segment
                suitable_providers = self._find_suitable_providers(segment)
                
                # Request offers from each provider
                segment_offers = []
                for provider_id in suitable_providers:
                    if provider_id not in self.model.providers:
                        continue
                        
                    provider = self.model.providers[provider_id]
                    
                    # Generate offer for this segment
                    if hasattr(provider, 'generate_bundle_component_offer'):
                        offer = provider.generate_bundle_component_offer(
                            {'bundle_id': bundle_id, 'segment': segment}, segment)
                        
                        if offer:
                            segment_offers.append(offer)
                
                # Store offers for this segment
                if segment_offers:
                    option_offers[segment_id] = segment_offers
                    self.bundle_offers[bundle_id][segment_id] = segment_offers
            
            # Store offers for this bundle option
            all_bundle_offers[f"option_{option_index}"] = option_offers
        
        # Update bundle status
        bundle['status'] = 'offers_collected'
        bundle['collected_offers'] = all_bundle_offers
        
        self.logger.info(f"Collected offers for bundle {bundle_id}")
        return all_bundle_offers
    
    def _find_suitable_providers(self, segment):
        """
        Find providers suitable for a segment.
        
        Args:
            segment: Segment specification
            
        Returns:
            List of provider IDs suitable for this segment
        """
        suitable_providers = []
        
        # Get all providers
        for provider_id, provider in self.model.providers.items():
            # Check if provider mode is preferred for this segment
            if provider.mode_type not in segment['preferred_modes']:
                continue
            
            # Check if provider can service this route
            if hasattr(provider, '_can_service_route'):
                if not provider._can_service_route(segment['origin'], segment['destination']):
                    continue
            
            # Check if provider has available capacity
            if provider.available_capacity <= 0:
                continue
            
            suitable_providers.append(provider_id)
        
        return suitable_providers
    
    def create_optimal_bundle(self, bundle_id):
        """
        Create the optimal bundle from collected offers.
        
        Args:
            bundle_id: Bundle ID
            
        Returns:
            Optimized bundle if successful, None otherwise
        """
        if bundle_id not in self.bundle_registry:
            self.logger.warning(f"Bundle {bundle_id} not found")
            return None
        
        bundle = self.bundle_registry[bundle_id]
        
        if bundle['status'] != 'offers_collected' or 'collected_offers' not in bundle:
            self.logger.warning(f"Bundle {bundle_id} not ready for optimization")
            return None
        
        # Evaluate each bundle option
        option_scores = []
        
        for option_key, option_offers in bundle['collected_offers'].items():
            # Skip options with missing offers for any segment
            if not self._has_complete_offers(option_key, bundle):
                continue
            
            # Find best offer combination for this option
            best_combination, score, total_price = self._find_best_offer_combination(
                option_key, bundle)
            
            if best_combination:
                option_scores.append({
                    'option_key': option_key,
                    'score': score,
                    'best_combination': best_combination,
                    'total_price': total_price
                })
        
        if not option_scores:
            self.logger.warning(f"No viable offer combinations for bundle {bundle_id}")
            bundle['status'] = 'failed'
            return None
        
        # Select highest scoring option
        best_option = max(option_scores, key=lambda x: x['score'])
        
        # Create optimized bundle
        optimized_bundle = {
            'bundle_id': bundle_id,
            'commuter_id': bundle['commuter_id'],
            'origin': bundle['origin'],
            'destination': bundle['destination'],
            'start_time': bundle['start_time'],
            'option_key': best_option['option_key'],
            'selected_offers': best_option['best_combination'],
            'total_price': best_option['total_price'],
            'score': best_option['score'],
            'status': 'optimized',
            'optimization_time': self.model.schedule.time
        }
        
        # Update bundle status
        bundle['status'] = 'optimized'
        bundle['optimized_bundle'] = optimized_bundle
        
        self.logger.info(f"Created optimal bundle for {bundle_id} with score {best_option['score']}")
        return optimized_bundle
    
    def _has_complete_offers(self, option_key, bundle):
        """
        Check if a bundle option has offers for all segments.
        
        Args:
            option_key: Option identifier
            bundle: Bundle details
            
        Returns:
            Boolean indicating completeness
        """
        option_index = int(option_key.split('_')[1])
        
        if option_index >= len(bundle['bundle_options']):
            return False
        
        option_segments = bundle['bundle_options'][option_index]
        option_offers = bundle['collected_offers'][option_key]
        
        # Check if all segments have at least one offer
        for segment in option_segments:
            if segment['segment_id'] not in option_offers:
                return False
            if not option_offers[segment['segment_id']]:
                return False
        
        return True
    
    def _find_best_offer_combination(self, option_key, bundle):
        """
        Find the best combination of offers for a bundle option.
        
        Args:
            option_key: Option identifier
            bundle: Bundle details
            
        Returns:
            (best_combination, score, total_price) or (None, 0, 0) if no valid combination
        """
        option_index = int(option_key.split('_')[1])
        option_segments = bundle['bundle_options'][option_index]
        option_offers = bundle['collected_offers'][option_key]
        
        # For extremely large combinations, we can use heuristics or sampling
        # For simplicity, we'll use a greedy approach for each segment
        
        best_combination = {}
        total_price = 0
        total_score = 0
        commuter_preferences = bundle.get('preferences', {})
        
        for segment in option_segments:
            segment_id = segment['segment_id']
            
            if segment_id not in option_offers:
                return None, 0, 0
                
            segment_offers = option_offers[segment_id]
            
            if not segment_offers:
                return None, 0, 0
            
            # Calculate score for each offer based on price, time match, and provider quality
            scored_offers = []
            for offer in segment_offers:
                offer_score = self._calculate_offer_score(offer, segment, commuter_preferences)
                scored_offers.append((offer_score, offer))
            
            # Select best offer for this segment
            best_offer_score, best_offer = max(scored_offers, key=lambda x: x[0])
            
            best_combination[segment_id] = best_offer
            total_price += best_offer['price']
            total_score += best_offer_score
        
        # Check if total price is within budget
        if total_price > bundle['max_price']:
            return None, 0, 0
        
        # Calculate final score, taking into account total price relative to max price
        price_ratio = total_price / bundle['max_price']
        price_score = 1 - price_ratio  # Lower price gets higher score
        
        final_score = total_score + (price_score * 50)  # Weight price in overall score
        
        return best_combination, final_score, total_price
    
    def _calculate_offer_score(self, offer, segment, preferences):
        """
        Calculate score for an offer based on multiple factors.
        
        Args:
            offer: Provider offer
            segment: Segment details
            preferences: Commuter preferences
            
        Returns:
            Offer score
        """
        # Base score components
        price_score = 0
        time_score = 0
        quality_score = 0
        reliability_score = 0
        mode_score = 0
        
        # Price score - lower is better
        max_reasonable_price = self._estimate_segment_price(segment)
        price_ratio = offer['price'] / max_reasonable_price
        price_score = max(0, 100 * (1 - price_ratio))
        
        # Time match score - closer to requested time is better
        requested_time = segment['start_time']
        offered_time = offer['start_time']
        time_diff = abs(offered_time - requested_time)
        time_score = 100 * math.exp(-0.1 * time_diff)  # Exponential decay
        
        # Provider quality and reliability
        provider_id = offer['provider_id']
        if provider_id in self.model.providers:
            provider = self.model.providers[provider_id]
            quality_score = getattr(provider, 'quality_score', 70)
            reliability_score = getattr(provider, 'reliability', 70)
        else:
            # Default scores if provider not found
            quality_score = 70
            reliability_score = 70
        
        # Mode preference score
        mode_score = 0
        if 'mode' in offer:
            if preferences and 'mode_preference' in preferences:
                mode_pref = preferences['mode_preference'].get(offer['mode'], 0.5)
                mode_score = 100 * mode_pref
            else:
                # Default mode score if no preferences
                mode_score = 50
        
        # Apply weights to different factors
        weights = {
            'price': 0.35,
            'time': 0.25,
            'quality': 0.15,
            'reliability': 0.15,
            'mode': 0.10
        }
        
        # Customize weights based on commuter preferences
        if preferences:
            if 'price_sensitivity' in preferences:
                # Adjust price weight based on sensitivity
                price_sensitivity = preferences['price_sensitivity']
                weights['price'] = 0.2 + (price_sensitivity * 0.3)  # 0.2 to 0.5 range
                
                # Normalize other weights
                weight_sum = sum(w for k, w in weights.items() if k != 'price')
                remaining_weight = 1 - weights['price']
                for k in weights:
                    if k != 'price':
                        weights[k] = weights[k] * remaining_weight / weight_sum
        
        # Calculate final weighted score
        final_score = (
            weights['price'] * price_score +
            weights['time'] * time_score +
            weights['quality'] * quality_score +
            weights['reliability'] * reliability_score +
            weights['mode'] * mode_score
        )
        
        return final_score
    
    def _estimate_segment_price(self, segment):
        """
        Estimate a reasonable price for a segment.
        
        Args:
            segment: Segment details
            
        Returns:
            Estimated price
        """
        # Calculate distance
        distance = self._calculate_distance(segment['origin'], segment['destination'])
        
        # Base price depends on segment type
        if segment['segment_type'] == 'main_journey':
            base_price = 1.0  # Lower per-unit for public transit
        elif segment['segment_type'] in ['first_mile', 'last_mile']:
            base_price = 2.0  # Higher for first/last mile
        else:
            base_price = 1.5  # Default for other segments
        
        return base_price * distance
    
    def execute_bundle_purchase(self, commuter_id, bundle_id):
        """
        Execute the purchase of a bundle.
        
        Args:
            commuter_id: Commuter ID
            bundle_id: Bundle ID
            
        Returns:
            Boolean indicating success
        """
        if bundle_id not in self.bundle_registry:
            self.logger.warning(f"Bundle {bundle_id} not found")
            return False
        
        bundle = self.bundle_registry[bundle_id]
        
        if bundle['status'] != 'optimized' or 'optimized_bundle' not in bundle:
            self.logger.warning(f"Bundle {bundle_id} not ready for purchase")
            return False
        
        optimized_bundle = bundle['optimized_bundle']
        
        # Format bundle for blockchain
        bundle_details = {
            'components': {},
            'total_price': optimized_bundle['total_price'],
            'name': f"Bundle {bundle_id[:8]}"
        }
        
        # Add component details
        for segment_id, offer in optimized_bundle['selected_offers'].items():
            bundle_details['components'][segment_id] = {
                'provider_id': offer['provider_id'],
                'price': offer['price'],
                'start_time': offer['start_time'],
                'duration': offer.get('estimated_time', 1800),
                'offer_signature': offer.get('signature')
            }
        
        # Execute on blockchain
        success, blockchain_bundle_id = self.blockchain_interface.execute_bundle_purchase(
            bundle_details, commuter_id)
        
        if success:
            # Update bundle status
            bundle['status'] = 'purchased'
            bundle['blockchain_bundle_id'] = blockchain_bundle_id
            
            # Move to active bundles
            self.active_bundles[bundle_id] = bundle
            
            # Update provider capacity
            self._update_provider_capacity(optimized_bundle)
            
            # Log success
            self.logger.info(f"Successfully purchased bundle {bundle_id} for commuter {commuter_id}")
            return True
        else:
            self.logger.warning(f"Failed to purchase bundle {bundle_id}")
            return False
    
    def _update_provider_capacity(self, bundle):
        """
        Update provider capacity after bundle purchase.
        
        Args:
            bundle: The purchased bundle
        """
        for segment_id, offer in bundle['selected_offers'].items():
            provider_id = offer['provider_id']
            
            if provider_id in self.model.providers:
                provider = self.model.providers[provider_id]
                
                # Reduce available capacity
                if provider.available_capacity > 0:
                    provider.available_capacity -= 1
                    self.logger.debug(f"Updated provider {provider_id} capacity to {provider.available_capacity}")
    
    def track_bundle_progress(self, bundle_id):
        """
        Track the progress of an active bundle.
        
        Args:
            bundle_id: Bundle ID
            
        Returns:
            Dict with bundle status details
        """
        if bundle_id not in self.active_bundles:
            self.logger.warning(f"Bundle {bundle_id} not in active bundles")
            return None
        
        bundle = self.active_bundles[bundle_id]
        optimized_bundle = bundle.get('optimized_bundle')
        
        if not optimized_bundle:
            return {'status': 'error', 'message': 'Bundle structure invalid'}
        
        current_time = self.model.schedule.time
        bundle_status = {
            'bundle_id': bundle_id,
            'status': 'in_progress',
            'segments': {},
            'overall_progress': 0
        }
        
        # Track each segment
        total_segments = len(optimized_bundle['selected_offers'])
        completed_segments = 0
        
        for segment_id, offer in optimized_bundle['selected_offers'].items():
            # Calculate segment times
            start_time = offer['start_time']
            estimated_duration = offer.get('estimated_time', 1800)
            estimated_end_time = start_time + estimated_duration
            
            # Determine segment status
            if current_time < start_time:
                segment_status = 'pending'
                progress = 0
            elif current_time < estimated_end_time:
                segment_status = 'in_progress'
                progress = min(100, ((current_time - start_time) / estimated_duration) * 100)
            else:
                segment_status = 'completed'
                progress = 100
                completed_segments += 1
            
            # Add segment status
            bundle_status['segments'][segment_id] = {
                'status': segment_status,
                'provider_id': offer['provider_id'],
                'progress': progress,
                'start_time': start_time,
                'estimated_end_time': estimated_end_time
            }
        
        # Calculate overall progress
        if total_segments > 0:
            # Progress is a weighted combination of completed segments and progress of in-progress segment
            bundle_status['overall_progress'] = (completed_segments * 100) / total_segments
            
            # Check if bundle is complete
            if completed_segments == total_segments:
                bundle_status['status'] = 'completed'
                
                # Move to completed bundles
                self._complete_bundle(bundle_id)
        
        return bundle_status
    
    def _complete_bundle(self, bundle_id):
        """
        Mark a bundle as completed and update statistics.
        
        Args:
            bundle_id: Bundle ID
        """
        if bundle_id not in self.active_bundles:
            return
        
        bundle = self.active_bundles[bundle_id]
        
        # Move to completed bundles
        self.completed_bundles[bundle_id] = bundle
        del self.active_bundles[bundle_id]
        
        # Update successful patterns
        self._update_pattern_statistics(bundle)
        
        # Log completion
        self.logger.info(f"Bundle {bundle_id} completed successfully")
    
    def _update_pattern_statistics(self, bundle):
        """
        Update statistics on successful bundle patterns.
        
        Args:
            bundle: The completed bundle
        """
        if 'optimized_bundle' not in bundle:
            return
        
        # Extract bundle pattern (number and types of segments)
        option_key = bundle['optimized_bundle']['option_key']
        option_index = int(option_key.split('_')[1])
        
        if option_index >= len(bundle['bundle_options']):
            return
        
        segments = bundle['bundle_options'][option_index]
        
        # Create pattern key based on segment types
        pattern_key = "_".join([segment['segment_type'] for segment in segments])
        
        # Update pattern statistics
        if pattern_key not in self.successful_patterns:
            self.successful_patterns[pattern_key] = {
                'count': 0,
                'average_price': 0,
                'total_price': 0,
                'bundles': []
            }
        
        stats = self.successful_patterns[pattern_key]
        stats['count'] += 1
        stats['total_price'] += bundle['optimized_bundle']['total_price']
        stats['average_price'] = stats['total_price'] / stats['count']
        stats['bundles'].append(bundle['id'])
        
        # Track popular routes
        route_key = f"{bundle['origin']}_{bundle['destination']}"
        
        if route_key not in self.popular_routes:
            self.popular_routes[route_key] = {
                'count': 0,
                'patterns': {},
                'average_price': 0,
                'total_price': 0
            }
        
        route_stats = self.popular_routes[route_key]
        route_stats['count'] += 1
        route_stats['total_price'] += bundle['optimized_bundle']['total_price']
        route_stats['average_price'] = route_stats['total_price'] / route_stats['count']
        
        # Track patterns for this route
        if pattern_key not in route_stats['patterns']:
            route_stats['patterns'][pattern_key] = 0
        route_stats['patterns'][pattern_key] += 1
    
    def recommend_bundles(self, origin, destination, commuter_preferences=None):
        """
        Recommend bundle patterns based on historical success.
        
        Args:
            origin: Origin coordinates
            destination: Destination coordinates
            commuter_preferences: Optional commuter preferences
            
        Returns:
            List of recommended bundle patterns
        """
        recommendations = []
        
        # Check for exact route match
        route_key = f"{origin}_{destination}"
        
        if route_key in self.popular_routes:
            # Use patterns that worked well for this exact route
            route_stats = self.popular_routes[route_key]
            
            for pattern_key, count in route_stats['patterns'].items():
                # Include patterns that have been used multiple times
                if count >= 2:
                    recommendations.append({
                        'pattern': pattern_key,
                        'confidence': min(1.0, count / 10),  # Confidence based on usage count
                        'avg_price': route_stats['average_price'],
                        'source': 'exact_route'
                    })
        
        # If no exact matches, use general patterns
        if not recommendations:
            # Find nearby routes
            nearby_routes = []
            
            for r_key, r_stats in self.popular_routes.items():
                try:
                    r_origin, r_destination = self._parse_route_key(r_key)
                    
                    # Calculate similarity
                    origin_distance = self._calculate_distance(origin, r_origin)
                    dest_distance = self._calculate_distance(destination, r_destination)
                    
                    # If route is similar enough
                    if origin_distance < 15 and dest_distance < 15:
                        nearby_routes.append((r_key, r_stats, origin_distance + dest_distance))
                except:
                    continue
            
            # Sort by similarity
            nearby_routes.sort(key=lambda x: x[2])
            
            # Use patterns from similar routes
            for r_key, r_stats, _ in nearby_routes[:3]:  # Top 3 similar routes
                for pattern_key, count in r_stats['patterns'].items():
                    if count >= 2:
                        recommendations.append({
                            'pattern': pattern_key,
                            'confidence': min(0.8, count / 15),  # Lower confidence for similar routes
                            'avg_price': r_stats['average_price'],
                            'source': 'similar_route'
                        })
        
        # If still no recommendations, use global patterns
        if not recommendations:
            for pattern_key, stats in self.successful_patterns.items():
                if stats['count'] >= 3:  # Only include well-established patterns
                    recommendations.append({
                        'pattern': pattern_key,
                        'confidence': min(0.7, stats['count'] / 20),  # Even lower confidence
                        'avg_price': stats['average_price'],
                        'source': 'global_pattern'
                    })
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Apply preferences filter if provided
        if commuter_preferences:
            recommendations = self._apply_preference_filter(recommendations, commuter_preferences)
        
        return recommendations
    
    def _apply_preference_filter(self, recommendations, preferences):
        """
        Filter and sort recommendations based on commuter preferences.
        
        Args:
            recommendations: List of recommendations
            preferences: Commuter preferences
            
        Returns:
            Filtered and sorted recommendations
        """
        # Check for price sensitivity
        if 'price_sensitivity' in preferences:
            price_sensitivity = preferences['price_sensitivity']
            
            if price_sensitivity > 0.7:  # Highly price sensitive
                # Emphasize cheaper options
                recommendations.sort(key=lambda x: (x['avg_price'], -x['confidence']))
                # Limit to cheaper half
                if len(recommendations) > 2:
                    mid_point = len(recommendations) // 2
                    recommendations = recommendations[:mid_point]
            elif price_sensitivity < 0.3:  # Low price sensitivity
                # Emphasize higher confidence options regardless of price
                recommendations.sort(key=lambda x: (-x['confidence'], x['avg_price']))
        
        # Check for time flexibility
        if 'time_flexibility' in preferences:
            time_flexibility = preferences['time_flexibility']
            
            if time_flexibility < 0.3:  # Low flexibility
                # Filter out complex patterns which might have more transfer points
                recommendations = [r for r in recommendations if r['pattern'].count('_') <= 2]
        
        # Check for mode preferences
        if 'preferred_modes' in preferences:
            preferred = preferences['preferred_modes']
            
            # Boost recommendations with preferred modes
            for r in recommendations:
                pattern = r['pattern']
                if 'car' in pattern and 'car' in preferred:
                    r['confidence'] += 0.1
                if 'bike' in pattern and 'bike' in preferred:
                    r['confidence'] += 0.1
                    
            # Resort after adjustments
            recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return recommendations
    
    def _parse_route_key(self, route_key):
        """
        Parse route key into origin and destination.
        
        Args:
            route_key: Route key string
            
        Returns:
            (origin, destination) tuple
        """
        try:
            parts = route_key.split('_')
            
            # Format could be "[x, y]_[x, y]" or similar
            if len(parts) >= 2:
                origin_str = parts[0].replace('[', '').replace(']', '')
                dest_str = parts[1].replace('[', '').replace(']', '')
                
                # Parse coordinates
                origin_coords = [float(x.strip()) for x in origin_str.split(',')]
                dest_coords = [float(x.strip()) for x in dest_str.split(',')]
                
                return origin_coords, dest_coords
        except:
            # Return default coordinates if parsing fails
            return [0, 0], [0, 0]
    
    def get_bundle_statistics(self):
        """
        Get statistics about bundle patterns and success rates.
        
        Returns:
            Dict with bundle statistics
        """
        stats = {
            'active_bundles': len(self.active_bundles),
            'completed_bundles': len(self.completed_bundles),
            'total_bundles_created': len(self.bundle_registry),
            'successful_patterns': {},
            'popular_routes': {},
            'average_bundle_segments': 0,
            'average_price': 0
        }
        
        # Calculate average segments and price
        total_segments = 0
        total_price = 0
        bundle_count = 0
        
        for bundle_id, bundle in self.completed_bundles.items():
            if 'optimized_bundle' in bundle:
                bundle_count += 1
                total_price += bundle['optimized_bundle']['total_price']
                total_segments += len(bundle['optimized_bundle']['selected_offers'])
        
        if bundle_count > 0:
            stats['average_bundle_segments'] = total_segments / bundle_count
            stats['average_price'] = total_price / bundle_count
        
        # Get top patterns
        sorted_patterns = sorted(
            self.successful_patterns.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        # Include top 5 patterns
        for pattern_key, pattern_stats in sorted_patterns[:5]:
            stats['successful_patterns'][pattern_key] = {
                'count': pattern_stats['count'],
                'average_price': pattern_stats['average_price'],
                'success_rate': pattern_stats['count'] / max(1, len(self.bundle_registry))
            }
        
        # Get top routes
        sorted_routes = sorted(
            self.popular_routes.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        # Include top 5 routes
        for route_key, route_stats in sorted_routes[:5]:
            stats['popular_routes'][route_key] = {
                'count': route_stats['count'],
                'average_price': route_stats['average_price'],
                'popular_patterns': sorted(
                    route_stats['patterns'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]  # Top 3 patterns for this route
            }
        
        return stats
    
    def update(self):
        """
        Update the state of all active bundles.
        Called on each model step.
        """
        # Process each active bundle
        for bundle_id in list(self.active_bundles.keys()):
            # Update bundle progress
            status = self.track_bundle_progress(bundle_id)
            
            # Log bundle completion
            if status and status['status'] == 'completed':
                self.logger.info(f"Bundle {bundle_id} completed during update")
        
        # Clean up old pending bundles
        self._clean_up_old_bundles()
    
    def _clean_up_old_bundles(self):
        """Clean up old pending bundles that were never completed"""
        current_time = self.model.schedule.time
        
        for bundle_id, bundle in list(self.bundle_registry.items()):
            # Skip active or completed bundles
            if bundle_id in self.active_bundles or bundle_id in self.completed_bundles:
                continue
                
            creation_time = bundle.get('creation_time', 0)
            age = current_time - creation_time
            
            # If bundle is stuck in early states for too long, mark as failed
            if bundle['status'] in ['initiated', 'seeking_offers', 'offers_collected'] and age > 100:
                bundle['status'] = 'failed'
                self.logger.info(f"Marked old bundle {bundle_id} as failed (age: {age})")