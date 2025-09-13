# File: abm/agents/decentralized_provider.py
# SIMPLIFIED VERSION - Uses marketplace API for all operations

import logging
import random
import math
from mesa import Agent

class DecentralizedProvider(Agent):
    """
    Simplified provider agent that uses marketplace API
    """
    
    def __init__(self, unique_id, model, pos, company_name, mode_type,
                 capacity, base_price, blockchain_interface=None):
        super().__init__(unique_id, model)
        
        # Basic attributes
        self.pos = pos
        self.company_name = company_name
        self.mode_type = mode_type
        self.capacity = capacity
        self.available_capacity = capacity
        self.base_price = base_price
        
        # Marketplace interface
        self.marketplace = blockchain_interface  # This is actually the marketplace API
        
        # Service tracking
        self.active_offers = {}
        self.completed_services = 0
        self.total_revenue = 0
        
        # Quality metrics
        self.quality_score = random.randint(60, 90)
        self.reliability = random.randint(70, 95)
        self.service_center = list(pos)
        
        # Logging
        self.logger = logging.getLogger(f"Provider-{unique_id}-{company_name}")
        
    def step(self):
        """Main step function - simplified flow"""
        # Register with marketplace if not registered
        if not hasattr(self, 'registered'):
            success, address = self.marketplace.register_provider(self)
            if success:
                self.registered = True
                self.address = address
        
        # Check for notifications from marketplace
        if hasattr(self, 'registered') and self.registered:
            self.check_marketplace_notifications()
    
    def check_marketplace_notifications(self):
        """Check for request notifications from marketplace"""
        notifications = self.marketplace.get_provider_notifications(self.unique_id)
        
        for notification in notifications[-5:]:  # Process last 5 notifications
            request_id = notification['request_id']
            if request_id not in self.active_offers:
                self.submit_offer_for_request(request_id)
    
    def submit_offer_for_request(self, request_id):
        """Submit an offer through marketplace API"""
        # Get request details from marketplace
        requests = self.marketplace.get_marketplace_requests()
        request = next((r for r in requests if r['request_id'] == request_id), None)
        
        if not request:
            return False
        
        # Calculate price based on distance
        distance = self._calculate_distance(
            self.service_center,
            request['origin']
        )
        
        price = self.base_price + (distance * 2)  # Simple pricing
        
        # Add some randomness
        price *= random.uniform(0.8, 1.2)
        
        # Prepare offer details
        offer_details = {
            'route': [request['origin'], request['destination']],
            'time': int(distance * 3),  # Estimated time
            'mode': self.mode_type
        }
        
        # Submit offer through marketplace API (not blockchain directly!)
        success = self.marketplace.submit_offer(
            self,
            request_id,
            price,
            offer_details
        )
        
        if success:
            self.active_offers[request_id] = {
                'price': price,
                'details': offer_details,
                'submitted_at': self.model.schedule.time
            }
            self.logger.info(f"Submitted offer for request {request_id} at price {price:.2f}")
        
        return success
    
    def complete_service(self, request_id, price):
        """Complete a service and update metrics"""
        self.completed_services += 1
        self.total_revenue += price
        
        # Remove from active offers
        if request_id in self.active_offers:
            del self.active_offers[request_id]
        
        # Update capacity
        self.available_capacity = self.capacity
        
        self.logger.info(f"Completed service for request {request_id}, earned {price:.2f}")
        return True
    
    def _calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def get_service_offers(self, origin, destination, start_time):
        """Get service offers for a route - used by marketplace"""
        distance = self._calculate_distance(origin, destination)
        price = self.base_price + (distance * 2)
        
        return [{
            'provider_id': self.unique_id,
            'price': price,
            'time': int(distance * 3),
            'mode': self.mode_type,
            'quality': self.quality_score
        }]
    
