"""
NFT Marketplace for Decentralized MaaS

This module implements a marketplace for NFTs representing mobility services,
with time-sensitive pricing, order book and AMM functionality, and market analytics.
"""

import numpy as np
import math
import json
import logging
from collections import defaultdict
import uuid

class NFTMarketplace:
    """
    NFT marketplace handling both order book and AMM-based market mechanisms
    with time-sensitive pricing for mobility services.
    """
    def __init__(self, model, blockchain_interface, market_type="hybrid"):
        """
        Initialize the NFT marketplace.
        
        Args:
            model: The model containing this marketplace
            blockchain_interface: Interface to the blockchain
            market_type: Type of market ("order_book", "amm", or "hybrid")
        """
        self.model = model
        self.blockchain_interface = blockchain_interface
        self.market_type = market_type  # "order_book", "amm", or "hybrid"
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("NFTMarketplace")
        
        # For order book model
        self.bid_book = {}  # Dictionary of bids by price
        self.ask_book = {}  # Dictionary of asks by price
        self.bids = {}      # Stores bid objects
        
        # For AMM model
        self.amm_pools = {}  # Liquidity pools for popular routes
        self.amm_params = {
            'time_decay_factor': 0.1,
            'demand_sensitivity': 0.2,
            'min_price_ratio': 0.5,  # Minimum price as ratio of original
            'volume_threshold': 5,   # Minimum transactions to consider a route popular
            'price_impact_factor': 0.05  # Price impact per volume unit
        }
        
        # Market metadata
        self.listings = {}
        self.transaction_history = []
        
        # Market analytics
        self.volume_by_route = {}
        self.price_history = {}
        self.market_depth = {}
        self.volatility_by_route = {}
        
        # Sorted order book prices (for faster matching)
        self.sorted_bids = []
        self.sorted_asks = []
        
        self.logger.info(f"NFT Marketplace initialized with {market_type} market type")

    def list_nft(self, owner_id, nft_id, nft_details, listing_params):
        """
        List an NFT for sale on the marketplace.
        
        Args:
            owner_id: ID of the NFT owner
            nft_id: ID of the NFT
            nft_details: Details of the NFT (service, route, time, etc.)
            listing_params: Parameters for the listing (price, pricing model, etc.)
            
        Returns:
            NFT ID if listing successful, None otherwise
        """
        # Extract listing parameters with defaults
        initial_price = listing_params.get('price', 0)
        dynamic_pricing = listing_params.get('dynamic_pricing', True)
        min_price_ratio = listing_params.get('min_price_ratio', 0.5)
        decay_rate = listing_params.get('decay_rate', 0.05)
        
        # Validate input
        if initial_price <= 0:
            self.logger.error(f"Invalid initial price: {initial_price}")
            return None
            
        # Create listing entry
        listing = {
            'nft_id': nft_id,
            'owner_id': owner_id,
            'details': nft_details,
            'initial_price': initial_price,
            'current_price': initial_price,
            'dynamic_pricing': dynamic_pricing,
            'min_price': initial_price * min_price_ratio,
            'listing_time': self.model.schedule.time,
            'decay_rate': decay_rate,
            'status': 'active'
        }
        
        # Store listing locally
        self.listings[nft_id] = listing
        
        # Add to order book if using order book model
        if self.market_type in ["order_book", "hybrid"]:
            self._add_to_order_book(nft_id, initial_price)
        
        # Add to AMM pool if using AMM model and it's a popular route
        if self.market_type in ["amm", "hybrid"] and self._is_popular_route(nft_details):
            self._add_to_amm_pool(nft_id, nft_details, initial_price)
        
        # List on blockchain
        time_parameters = {
            'initial_price': initial_price,
            'final_price': initial_price * min_price_ratio,
            'decay_duration': self._calculate_decay_duration(nft_details)
        }
        
        blockchain_success = self.blockchain_interface.list_nft_for_sale(
            nft_id, initial_price, time_parameters)
            
        if not blockchain_success:
            self.logger.warning(f"Blockchain listing failed for NFT {nft_id}")
            # Continue anyway - blockchain operations are async
        
        self.logger.info(f"NFT {nft_id} listed for {initial_price} with decay rate {decay_rate}")
        return nft_id

    def _add_to_order_book(self, nft_id, price):
        """
        Add listing to order book as an ask.
        
        Args:
            nft_id: NFT ID to add
            price: Ask price
        """
        price_key = self._get_price_key(price)
        
        if price_key not in self.ask_book:
            self.ask_book[price_key] = []
            
        self.ask_book[price_key].append(nft_id)
        
        # Sort ask prices (ascending)
        self._sort_order_book()
        
        # Update market depth
        route_key = self._get_route_key(self.listings[nft_id]['details'])
        if route_key not in self.market_depth:
            self.market_depth[route_key] = {'asks': defaultdict(float), 'bids': defaultdict(float)}
            
        self.market_depth[route_key]['asks'][price_key] += 1
        
        self.logger.debug(f"Added NFT {nft_id} to order book at price {price}")

    def _get_price_key(self, price):
        """
        Convert price to standard key format for order book.
        
        Args:
            price: Price value
            
        Returns:
            Rounded price key for order book
        """
        # Round to 4 decimal places for reasonable price grouping
        return round(float(price), 4)

    def _sort_order_book(self):
        """Sort bid and ask books by price for faster matching."""
        self.sorted_bids = sorted(self.bid_book.keys(), reverse=True)  # Highest first
        self.sorted_asks = sorted(self.ask_book.keys())  # Lowest first

    def _add_to_amm_pool(self, nft_id, nft_details, initial_price):
        """
        Add NFT to AMM liquidity pool for its route.
        
        Args:
            nft_id: NFT ID to add
            nft_details: NFT details with route information
            initial_price: Initial listing price
        """
        # Create route key from origin/destination
        route_key = self._get_route_key(nft_details)
        
        # Initialize pool if needed
        if route_key not in self.amm_pools:
            self.amm_pools[route_key] = {
                'nfts': [],
                'base_price': initial_price,
                'current_price': initial_price,
                'last_update_time': self.model.schedule.time,
                'demand_factor': 1.0,
                'liquidity': initial_price,
                'volume_24h': 0
            }
        
        # Add to pool
        self.amm_pools[route_key]['nfts'].append(nft_id)
        
        # If this is a new NFT, add its value to the pool liquidity
        self.amm_pools[route_key]['liquidity'] += initial_price
        
        self.logger.info(f"Added NFT {nft_id} to AMM pool for route {route_key}")

    def _get_route_key(self, nft_details):
        """
        Create a standardized key for a route.
        
        Args:
            nft_details: NFT details with origin and destination
            
        Returns:
            Route key string
        """
        origin = tuple(nft_details['origin'])
        destination = tuple(nft_details['destination'])
        
        # Handle time window (round to nearest hour)
        time_window = round(nft_details['service_time'] / 3600) * 3600
        
        return f"{origin}_{destination}_{time_window}"

    def _is_popular_route(self, nft_details):
        """
        Determine if a route is popular enough for AMM.
        
        Args:
            nft_details: NFT details with route information
            
        Returns:
            Boolean indicating if route is popular
        """
        route_key = self._get_route_key(nft_details)
        
        # Check transaction history
        transaction_count = self.volume_by_route.get(route_key, 0)
        
        # Popular routes have at least threshold transactions
        return transaction_count >= self.amm_params['volume_threshold']

    def _calculate_decay_duration(self, nft_details):
        """
        Calculate appropriate decay duration based on service time.
        
        Args:
            nft_details: NFT details including service time
            
        Returns:
            Decay duration in seconds
        """
        current_time = self.model.schedule.time
        service_time = nft_details['service_time']
        time_to_service = max(0, service_time - current_time)
        
        # Use 70% of time to service as decay period
        decay_duration = int(time_to_service * 0.7)
        
        # Ensure minimum 1 hour and maximum 24 hours
        return max(3600, min(decay_duration, 86400))

    def update_listings(self):
        """Update pricing for all listings based on time and market conditions."""
        current_time = self.model.schedule.time
        
        # Update order book listings
        for nft_id, listing in list(self.listings.items()):
            if listing['status'] != 'active':
                continue
                
            old_price = listing['current_price']
            
            # Update dynamic pricing if enabled
            if listing['dynamic_pricing']:
                # Calculate time-based price decay
                listing_time = listing['listing_time']
                time_elapsed = current_time - listing_time
                service_time = listing['details']['service_time']
                time_to_service = service_time - current_time
                
                # Price decay accelerates as service time approaches
                if time_to_service <= 0:
                    # Service time passed, set to minimum price
                    new_price = listing['min_price']
                else:
                    # Use exponential decay based on time elapsed and time to service
                    decay_factor = math.exp(-listing['decay_rate'] * time_elapsed * (1 + 1/max(1, time_to_service/3600)))
                    new_price = max(
                        listing['min_price'],
                        listing['initial_price'] * decay_factor
                    )
                
                # Update price in listing
                listing['current_price'] = new_price
                
                # Update order book if price changed significantly (>1%)
                if abs(old_price - new_price) / old_price > 0.01:
                    self._update_order_book_price(nft_id, old_price, new_price)
            
            # Check if NFT has expired (service time passed)
            if current_time > listing['details']['service_time']:
                # Mark as expired
                listing['status'] = 'expired'
                
                # Remove from order book
                if self.market_type in ["order_book", "hybrid"]:
                    self._remove_from_order_book(nft_id, listing['current_price'])
                
                # Remove from AMM pool if present
                if self.market_type in ["amm", "hybrid"]:
                    self._remove_from_amm_pool(nft_id)
                    
                self.logger.info(f"NFT {nft_id} expired at time {current_time}")
        
        # Update AMM pools
        if self.market_type in ["amm", "hybrid"]:
            self._update_amm_pools()
        
        # Match any outstanding bids
        if self.market_type in ["order_book", "hybrid"]:
            self._match_outstanding_bids()

    def _update_order_book_price(self, nft_id, old_price, new_price):
        """
        Update NFT position in order book when price changes.
        
        Args:
            nft_id: NFT ID to update
            old_price: Previous price
            new_price: New price
        """
        # No update needed if prices are effectively the same
        if round(old_price, 4) == round(new_price, 4):
            return
            
        # Remove from old price level
        old_price_key = self._get_price_key(old_price)
        if old_price_key in self.ask_book and nft_id in self.ask_book[old_price_key]:
            self.ask_book[old_price_key].remove(nft_id)
            
            # Clean up empty price levels
            if not self.ask_book[old_price_key]:
                del self.ask_book[old_price_key]
        
        # Add to new price level
        new_price_key = self._get_price_key(new_price)
        if new_price_key not in self.ask_book:
            self.ask_book[new_price_key] = []
        self.ask_book[new_price_key].append(nft_id)
        
        # Re-sort order book
        self._sort_order_book()
        
        # Update market depth
        if nft_id in self.listings:
            route_key = self._get_route_key(self.listings[nft_id]['details'])
            if route_key in self.market_depth:
                # Decrease count at old price
                self.market_depth[route_key]['asks'][old_price_key] -= 1
                if self.market_depth[route_key]['asks'][old_price_key] <= 0:
                    del self.market_depth[route_key]['asks'][old_price_key]
                
                # Increase count at new price
                self.market_depth[route_key]['asks'][new_price_key] += 1
        
        self.logger.debug(f"Updated price for NFT {nft_id} from {old_price} to {new_price}")

    def _remove_from_order_book(self, nft_id, price):
        """
        Remove NFT from order book.
        
        Args:
            nft_id: NFT ID to remove
            price: Current price of the NFT
        """
        price_key = self._get_price_key(price)
        
        if price_key in self.ask_book and nft_id in self.ask_book[price_key]:
            self.ask_book[price_key].remove(nft_id)
            
            # Clean up empty price levels
            if not self.ask_book[price_key]:
                del self.ask_book[price_key]
            
            # Re-sort order book
            self._sort_order_book()
            
            # Update market depth
            if nft_id in self.listings:
                route_key = self._get_route_key(self.listings[nft_id]['details'])
                if route_key in self.market_depth:
                    self.market_depth[route_key]['asks'][price_key] -= 1
                    if self.market_depth[route_key]['asks'][price_key] <= 0:
                        del self.market_depth[route_key]['asks'][price_key]
            
            self.logger.debug(f"Removed NFT {nft_id} from order book at price {price}")

    def _remove_from_amm_pool(self, nft_id):
        """
        Remove NFT from AMM pool.
        
        Args:
            nft_id: NFT ID to remove
        """
        # Find the pool containing this NFT
        for route_key, pool in list(self.amm_pools.items()):
            if nft_id in pool['nfts']:
                pool['nfts'].remove(nft_id)
                
                # Remove pool if empty
                if not pool['nfts']:
                    del self.amm_pools[route_key]
                    self.logger.info(f"AMM pool for route {route_key} removed (no NFTs)")
                else:
                    self.logger.debug(f"Removed NFT {nft_id} from AMM pool {route_key}")
                break

    def _update_amm_pools(self):
        """Update AMM pool pricing based on time decay and market conditions."""
        current_time = self.model.schedule.time
        
        for route_key, pool in list(self.amm_pools.items()):
            if not pool['nfts']:
                continue
                
            # Get first NFT to determine service time
            sample_nft_id = pool['nfts'][0]
            if sample_nft_id not in self.listings:
                continue
                
            sample_nft = self.listings[sample_nft_id]
            service_time = sample_nft['details']['service_time']
            
            # Calculate time to service
            time_to_service = service_time - current_time
            
            # Update demand factor based on market activity
            transaction_count = self.volume_by_route.get(route_key, 0)
            recent_transactions = sum(1 for tx in self.transaction_history[-20:] 
                                    if self._get_route_key(self.listings.get(tx['nft_id'], {}).get('details', {})) == route_key)
            
            # Demand factor increases with more recent transactions
            if recent_transactions > 3:
                # High recent demand
                pool['demand_factor'] = min(1.5, pool['demand_factor'] * 1.05)
            elif recent_transactions == 0 and transaction_count > 0:
                # No recent demand for popular route
                pool['demand_factor'] = max(0.7, pool['demand_factor'] * 0.95)
            
            # Calculate new price based on time decay and demand
            if time_to_service <= 0:
                # Service time passed, set minimum price
                new_price = pool['base_price'] * self.amm_params['min_price_ratio']
            else:
                # Apply time decay formula with sharper decay as service time approaches
                time_factor = math.exp(-self.amm_params['time_decay_factor'] * 
                                     (1.0 / max(1, time_to_service/3600)))
                
                # Apply demand adjustment
                demand_adjustment = 1.0 + (pool['demand_factor'] - 1.0) * self.amm_params['demand_sensitivity']
                
                # Calculate new price
                new_price = max(
                    pool['base_price'] * self.amm_params['min_price_ratio'],
                    pool['base_price'] * time_factor * demand_adjustment
                )
            
            # Update pool price
            old_price = pool['current_price']
            pool['current_price'] = new_price
            pool['last_update_time'] = current_time
            
            # Update individual listing prices to match AMM price
            for nft_id in pool['nfts']:
                if nft_id in self.listings and self.listings[nft_id]['status'] == 'active':
                    self.listings[nft_id]['current_price'] = new_price
                    
                    # Update order book if using hybrid model and price changed significantly
                    if self.market_type == "hybrid" and abs(old_price - new_price) / old_price > 0.01:
                        self._update_order_book_price(nft_id, old_price, new_price)
            
            # Record price change in history
            if route_key not in self.price_history:
                self.price_history[route_key] = []
            self.price_history[route_key].append((current_time, new_price))
            
            # Limit history size
            if len(self.price_history[route_key]) > 100:
                self.price_history[route_key] = self.price_history[route_key][-100:]
            
            # Calculate volatility
            if len(self.price_history[route_key]) >= 10:
                # Use last 10 price points to calculate volatility
                recent_prices = [p for _, p in self.price_history[route_key][-10:]]
                self.volatility_by_route[route_key] = np.std(recent_prices) / np.mean(recent_prices)
            
            self.logger.debug(f"Updated AMM pool price for route {route_key}: {old_price} -> {new_price}")

    def place_bid(self, buyer_id, route_params, max_price):
        """
        Place a bid in the order book.
        
        Args:
            buyer_id: ID of the buyer
            route_params: Dict with origin, destination, and time window
            max_price: Maximum bid price
            
        Returns:
            Matched NFT ID if successful, None otherwise
        """
        if self.market_type not in ["order_book", "hybrid"]:
            return None
        
        # Generate unique bid ID
        bid_id = str(uuid.uuid4())
        
        # Create bid entry
        bid = {
            'bid_id': bid_id,
            'buyer_id': buyer_id,
            'route_params': route_params,
            'max_price': max_price,
            'bid_time': self.model.schedule.time,
            'status': 'active'
        }
        
        # Add to bid book
        price_key = self._get_price_key(max_price)
        if price_key not in self.bid_book:
            self.bid_book[price_key] = []
        self.bid_book[price_key].append(bid_id)
        
        # Store bid details
        self.bids[bid_id] = bid
        
        # Update market depth
        route_key = f"{route_params['origin']}_{route_params['destination']}"
        if route_key not in self.market_depth:
            self.market_depth[route_key] = {'asks': defaultdict(float), 'bids': defaultdict(float)}
        self.market_depth[route_key]['bids'][price_key] += 1
        
        # Resort order book
        self._sort_order_book()
        
        self.logger.info(f"Bid {bid_id} placed by {buyer_id} at max price {max_price}")
        
        # Try to match immediately
        matched_nft_id = self._match_bid(bid_id)
        
        return matched_nft_id

    def _match_bid(self, bid_id):
        """
        Try to match a bid with available asks.
        
        Args:
            bid_id: ID of the bid to match
            
        Returns:
            Matched NFT ID if successful, None otherwise
        """
        if bid_id not in self.bids:
            return None
            
        bid = self.bids[bid_id]
        max_price = bid['max_price']
        route_params = bid['route_params']
        
        # Find matching NFTs within price range
        matching_nfts = self.search_nfts({
            'origin_area': route_params.get('origin', [0, 0, 999999]),
            'destination_area': route_params.get('destination', [0, 0, 999999]),
            'time_window': route_params.get('time_window', [0, 999999]),
            'max_price': max_price
        })
        
        if not matching_nfts:
            self.logger.debug(f"No matching NFTs found for bid {bid_id}")
            return None
        
        # Find best match (lowest price)
        best_match = min(matching_nfts, key=lambda x: x['price'])
        
        # Execute purchase
        purchase_success = self.purchase_nft(bid['buyer_id'], best_match['nft_id'], best_match['price'])
        
        if purchase_success:
            # Update bid status
            bid['status'] = 'matched'
            
            # Remove from bid book
            price_key = self._get_price_key(max_price)
            if price_key in self.bid_book and bid_id in self.bid_book[price_key]:
                self.bid_book[price_key].remove(bid_id)
                
                # Clean up empty price levels
                if not self.bid_book[price_key]:
                    del self.bid_book[price_key]
                
                # Update market depth
                route_key = f"{route_params['origin']}_{route_params['destination']}"
                if route_key in self.market_depth:
                    self.market_depth[route_key]['bids'][price_key] -= 1
                    if self.market_depth[route_key]['bids'][price_key] <= 0:
                        del self.market_depth[route_key]['bids'][price_key]
            
            self.logger.info(f"Bid {bid_id} matched with NFT {best_match['nft_id']} at price {best_match['price']}")
            return best_match['nft_id']
        
        return None

    def _match_outstanding_bids(self):
        """Try to match any outstanding bids with available asks."""
        # Process bids from highest to lowest price
        for price_key in sorted(self.bid_book.keys(), reverse=True):
            bid_ids = self.bid_book.get(price_key, [])[:]  # Make a copy
            
            for bid_id in bid_ids:
                if bid_id in self.bids and self.bids[bid_id]['status'] == 'active':
                    # Try to match this bid
                    matched_nft_id = self._match_bid(bid_id)
                    
                    if matched_nft_id:
                        # Successfully matched, move to next bid
                        continue

    def search_nfts(self, search_params):
        """
        Search for NFTs matching criteria.
        
        Args:
            search_params: Dict with search criteria
                - origin_area: [center_coordinates, radius]
                - destination_area: [center_coordinates, radius]
                - time_window: [min_time, max_time]
                - max_price: Maximum price
                
        Returns:
            List of matching NFT details
        """
        results = []
        
        # Extract search parameters with defaults
        origin_center = search_params.get('origin_area', [[0, 0], 999999])[0]
        origin_radius = search_params.get('origin_area', [[0, 0], 999999])[1]
        
        dest_center = search_params.get('destination_area', [[0, 0], 999999])[0]
        dest_radius = search_params.get('destination_area', [[0, 0], 999999])[1]
        
        min_time = search_params.get('time_window', [0, 999999])[0]
        max_time = search_params.get('time_window', [0, 999999])[1]
        
        max_price = search_params.get('max_price', 999999)
        
        # Search active listings
        for nft_id, listing in self.listings.items():
            if listing['status'] != 'active':
                continue
                
            # Check price constraint
            if listing['current_price'] > max_price:
                continue
                
            # Get NFT details
            nft_details = listing['details']
            
            # Check origin constraint
            origin_dist = self._calculate_distance(origin_center, nft_details['origin'])
            if origin_dist > origin_radius:
                continue
                
            # Check destination constraint
            dest_dist = self._calculate_distance(dest_center, nft_details['destination'])
            if dest_dist > dest_radius:
                continue
                
            # Check time window constraint
            service_time = nft_details['service_time']
            if service_time < min_time or service_time > max_time:
                continue
                
            # NFT matches all constraints, add to results
            results.append({
                'nft_id': nft_id,
                'price': listing['current_price'],
                'owner_id': listing['owner_id'],
                'details': nft_details
            })
        
        return results

    def _calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point [x, y]
            point2: Second point [x, y]
            
        Returns:
            Euclidean distance
        """
        return math.sqrt(
            (point2[0] - point1[0])**2 +
            (point2[1] - point1[1])**2
        )

    def purchase_nft(self, buyer_id, nft_id, offer_price=None):
        """
        Purchase an NFT from the marketplace.
        
        Args:
            buyer_id: ID of the buyer
            nft_id: ID of the NFT to purchase
            offer_price: Optional offer price (for order book model)
            
        Returns:
            Boolean indicating success
        """
        # Check if NFT exists and is active
        if nft_id not in self.listings or self.listings[nft_id]['status'] != 'active':
            self.logger.warning(f"NFT {nft_id} not available for purchase")
            return False
            
        listing = self.listings[nft_id]
        
        # Check price (for order book model)
        if self.market_type in ["order_book", "hybrid"] and offer_price is not None:
            if offer_price < listing['current_price']:
                self.logger.warning(f"Offer price {offer_price} too low for NFT {nft_id}")
                return False
        
        # Get price from appropriate source
        if self.market_type == "amm":
            # Use AMM price for this route
            route_key = self._get_route_key(listing['details'])
            if route_key in self.amm_pools:
                price = self.amm_pools[route_key]['current_price']
            else:
                price = listing['current_price']
        else:
            # Use listing price
            price = listing['current_price']
        
        # Execute purchase via blockchain
        transaction_success = self.blockchain_interface.purchase_nft(
            nft_id, buyer_id)
            
        if transaction_success:
            # Update listing status
            listing['status'] = 'sold'
            
            # Remove from order book
            if self.market_type in ["order_book", "hybrid"]:
                self._remove_from_order_book(nft_id, listing['current_price'])
            
            # Remove from AMM pool
            if self.market_type in ["amm", "hybrid"]:
                self._remove_from_amm_pool(nft_id)
            
            # Record transaction
            transaction = {
                'nft_id': nft_id,
                'buyer_id': buyer_id,
                'seller_id': listing['owner_id'],
                'price': price,
                'time': self.model.schedule.time
            }
            self.transaction_history.append(transaction)
            
            # Update market analytics
            route_key = self._get_route_key(listing['details'])
            self.volume_by_route[route_key] = self.volume_by_route.get(route_key, 0) + 1
            
            # Update price history
            if route_key not in self.price_history:
                self.price_history[route_key] = []
            self.price_history[route_key].append((self.model.schedule.time, price))
            
            # Update AMM pool volume_24h
            if route_key in self.amm_pools:
                self.amm_pools[route_key]['volume_24h'] += price
                
                # Decay older volume (simple approximation of 24h window)
                # Assumes each update is ~1 time unit
                self.amm_pools[route_key]['volume_24h'] *= 0.99
            
            # Update model tracking properties
            if hasattr(self.model, 'transaction_count'):
                self.model.transaction_count += 1
            
            self.logger.info(f"NFT {nft_id} purchased by {buyer_id} at price {price}")
            return True
            
        self.logger.warning(f"Blockchain purchase failed for NFT {nft_id}")
        return False

    def get_market_price(self, route_details):
        """
        Get current market price for a specific route and time.
        
        Args:
            route_details: Dict with origin, destination, and service time
            
        Returns:
            Current market price or None if no data
        """
        route_key = self._get_route_key(route_details)
        
        # Check AMM pools first
        if route_key in self.amm_pools:
            return self.amm_pools[route_key]['current_price']
            
        # Check recent transactions
        route_transactions = [tx for tx in self.transaction_history[-20:]
                            if self._get_route_key(self.listings.get(tx['nft_id'], {}).get('details', {})) == route_key]
        
        if route_transactions:
            # Return average of recent prices
            return np.mean([tx['price'] for tx in route_transactions])
            
        # Check order book
        matching_nfts = self.search_nfts({
            'origin_area': [route_details['origin'], 10],
            'destination_area': [route_details['destination'], 10],
            'time_window': [route_details['service_time'] - 3600, route_details['service_time'] + 3600]
        })
        
        if matching_nfts:
            # Return average price of similar listings
            return np.mean([nft['price'] for nft in matching_nfts])
            
        return None

    def get_market_analytics(self):
        """
        Get analytics data for the marketplace.
        
        Returns:
            Dict containing market analytics data
        """
        # Calculate aggregate statistics
        analytics = {
            'active_listings': sum(1 for listing in self.listings.values() if listing['status'] == 'active'),
            'transaction_volume': len(self.transaction_history),
            'active_bids': sum(len(bids) for bids in self.bid_book.values()),
            'amm_pools': len(self.amm_pools),
            'average_price': np.mean([tx['price'] for tx in self.transaction_history[-100:]]) if self.transaction_history else 0,
            'median_price': np.median([tx['price'] for tx in self.transaction_history[-100:]]) if self.transaction_history else 0,
            'popular_routes': sorted(self.volume_by_route.items(), key=lambda x: x[1], reverse=True)[:5],
            'price_trends': {},
            'volatility': {},
            'spread': {},
            'last_updated': self.model.schedule.time
        }
        
        # Calculate price trends for popular routes
        for route, count in self.volume_by_route.items():
            if count >= 5 and route in self.price_history and len(self.price_history[route]) >= 5:
                analytics['price_trends'][route] = self._calculate_price_trend(self.price_history[route])
                
                if route in self.volatility_by_route:
                    analytics['volatility'][route] = self.volatility_by_route[route]
                
                # Calculate bid-ask spread for routes with order book data
                if route in self.market_depth:
                    ask_prices = sorted(self.market_depth[route]['asks'].keys())
                    bid_prices = sorted(self.market_depth[route]['bids'].keys(), reverse=True)
                    
                    if ask_prices and bid_prices:
                        analytics['spread'][route] = ask_prices[0] - bid_prices[0]
        
        return analytics

    def _calculate_price_trend(self, price_history):
        """
        Calculate price trend from history.
        
        Args:
            price_history: List of (time, price) tuples
            
        Returns:
            Price trend coefficient
        """
        if len(price_history) < 5:
            return 0
        
        # Extract times and prices
        times = np.array([t for t, _ in price_history])
        prices = np.array([p for _, p in price_history])
        
        # Normalize times to avoid numerical issues
        times_norm = times - times[0]
        
        # Avoid division by zero
        if np.sum(times_norm**2) == 0:
            return 0
        
        # Calculate slope of best fit line
        slope = np.sum(times_norm * prices) / np.sum(times_norm**2)
        
        # Normalize by average price
        avg_price = np.mean(prices)
        if avg_price == 0:
            return 0
            
        normalized_slope = slope / avg_price
        
        return normalized_slope