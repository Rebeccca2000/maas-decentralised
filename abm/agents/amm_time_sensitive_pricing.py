import logging
import random
import numpy as np
import logging
from collections import defaultdict
import math

class MobilityAMM:
    """
    Automated Market Maker for mobility services on popular routes.
    
    This AMM establishes liquidity pools for popular routes and provides
    more efficient price discovery than direct provider offers.
    """
    
    def __init__(self, blockchain_interface, model, initial_routes=None):
        """
        Initialize the Mobility AMM.
        
        Args:
            blockchain_interface: Interface to the blockchain
            model: The model containing the AMM
            initial_routes: List of initial popular routes to create pools for
        """
        self.blockchain_interface = blockchain_interface
        self.model = model
        self.logger = logging.getLogger("MobilityAMM")
        self.logger.setLevel(logging.INFO)
        
        # Liquidity pools for popular routes
        # Each pool has a constant product formula (x * y = k)
        self.liquidity_pools = {}
        
        # Price history for routes
        self.price_history = {}
        
        # Initialize pools for initial routes
        if initial_routes:
            for route in initial_routes:
                self.create_liquidity_pool(route)
    
    def create_liquidity_pool(self, route_info):
        """
        Create a liquidity pool for a popular route.
        
        Args:
            route_info: Dictionary with route details including:
                - origin, destination coordinates
                - initial_price: Starting price for the route
                - initial_liquidity: Amount of initial liquidity
                
        Returns:
            Pool ID
        """
        # Generate a unique ID for this route
        origin = tuple(route_info['origin'])
        destination = tuple(route_info['destination'])
        route_key = f"{origin}_{destination}"
        
        # Check if pool already exists
        if route_key in self.liquidity_pools:
            self.logger.info(f"Pool already exists for route {route_key}")
            return route_key
        
        # Get initial parameters
        initial_price = route_info.get('initial_price', 10)
        initial_liquidity = route_info.get('initial_liquidity', 1000)
        
        # Calculate token reserves (x * y = k)
        # For AMM, we use "service tokens" and "payment tokens"
        service_tokens = math.sqrt(initial_liquidity / initial_price)
        payment_tokens = service_tokens * initial_price
        
        # Create pool
        self.liquidity_pools[route_key] = {
            'route_key': route_key,
            'origin': origin,
            'destination': destination,
            'service_tokens': service_tokens,
            'payment_tokens': payment_tokens,
            'k': service_tokens * payment_tokens,  # Constant product
            'last_price': initial_price,
            'created_at': self.model.schedule.time,
            'trades': [],
            'providers': [],  # Providers contributing to this pool
            'fees': {
                'provider_fee': 0.005,  # 0.5% to providers
                'protocol_fee': 0.002   # 0.2% to protocol
            }
        }
        
        # Initialize price history
        self.price_history[route_key] = [
            (self.model.schedule.time, initial_price)
        ]
        
        self.logger.info(f"Created liquidity pool for route {route_key} with initial price {initial_price}")
        return route_key
        
    def get_quote(self, origin, destination, amount, is_buy):
        """
        Get a price quote for buying or selling a mobility service.
        
        Args:
            origin: Origin coordinates
            destination: Destination coordinates
            amount: Amount of service tokens to buy or payment tokens to spend
            is_buy: True if buying service, False if selling
            
        Returns:
            Dict with quote details or None if no pool
        """
        # Find the closest matching pool
        route_key = self._find_closest_pool(origin, destination)
        
        if not route_key:
            return None
            
        pool = self.liquidity_pools[route_key]
        
        # Calculate price based on constant product formula
        if is_buy:
            # Buying service tokens with payment tokens
            # Calculate how many service tokens user will receive
            # New reserves after trade: (x - Δx) * (y + Δy) = k
            # Where Δy = amount (payment tokens in), Δx = service tokens out
            
            # Apply fee
            fee = amount * (pool['fees']['provider_fee'] + pool['fees']['protocol_fee'])
            amount_with_fee = amount - fee
            
            service_out = pool['service_tokens'] - (pool['k'] / (pool['payment_tokens'] + amount_with_fee))
            
            # Calculate effective price
            price = amount / service_out
            
            return {
                'route_key': route_key,
                'type': 'buy',
                'input_amount': amount,
                'output_amount': service_out,
                'price': price,
                'fee': fee,
                'origin': pool['origin'],
                'destination': pool['destination'],
                'slippage': self._calculate_slippage(pool, price, is_buy)
            }
        else:
            # Selling service tokens for payment tokens
            # Calculate how many payment tokens user will receive
            # New reserves after trade: (x + Δx) * (y - Δy) = k
            # Where Δx = amount (service tokens in), Δy = payment tokens out
            
            # Apply fee
            fee_multiplier = 1 - (pool['fees']['provider_fee'] + pool['fees']['protocol_fee'])
            
            payment_out = (pool['payment_tokens'] - (pool['k'] / (pool['service_tokens'] + amount))) * fee_multiplier
            
            # Calculate effective price
            price = payment_out / amount
            
            return {
                'route_key': route_key,
                'type': 'sell',
                'input_amount': amount,
                'output_amount': payment_out,
                'price': price,
                'fee': payment_out * (1 - fee_multiplier) / fee_multiplier,
                'origin': pool['origin'],
                'destination': pool['destination'],
                'slippage': self._calculate_slippage(pool, price, is_buy)
            }
    
    def execute_swap(self, quote, agent_id):
        """
        Execute a swap based on a quote.
        
        Args:
            quote: Quote from get_quote
            agent_id: ID of the agent executing the swap
            
        Returns:
            Success status and swap details
        """
        if not quote:
            return False, None
            
        route_key = quote['route_key']
        
        if route_key not in self.liquidity_pools:
            return False, None
            
        pool = self.liquidity_pools[route_key]
        
        # Check if quote is still valid (price hasn't moved too much)
        current_quote = self.get_quote(
            quote['origin'], 
            quote['destination'], 
            quote['input_amount'], 
            quote['type'] == 'buy'
        )
        
        if not current_quote:
            return False, None
            
        # Check for excessive price movement
        price_difference = abs(current_quote['price'] - quote['price']) / quote['price']
        if price_difference > 0.05:  # 5% price movement
            self.logger.warning(f"Price moved {price_difference:.2%}, rejecting swap")
            return False, None
        
        # Execute the swap
        if quote['type'] == 'buy':
            # Agent is buying service tokens with payment tokens
            new_service_tokens = pool['service_tokens'] - quote['output_amount']
            new_payment_tokens = pool['payment_tokens'] + quote['input_amount']
            
            # Update pool
            pool['service_tokens'] = new_service_tokens
            pool['payment_tokens'] = new_payment_tokens
            pool['k'] = new_service_tokens * new_payment_tokens  # Update k slightly due to fees
            pool['last_price'] = quote['price']
            
            # Create NFT for the service
            service_details = {
                'request_id': int(random.randint(1000000, 9999999)),  # Synthetic request ID
                'price': quote['input_amount'],
                'start_time': self.model.schedule.time + 3600,  # 1 hour from now
                'duration': 1800,  # 30 minutes
                'route_details': {
                    'route': [list(pool['origin']), list(pool['destination'])],
                    'amm_trade': True,
                    'price': quote['price']
                }
            }
            
            # Determine provider
            if pool['providers']:
                provider_id = random.choice(pool['providers'])
            else:
                # Create a synthetic provider if none available
                provider_id = -1  # AMM pool itself is the provider
            
            # Mint NFT
            success, nft_id = self.blockchain_interface.create_nft(
                service_details, provider_id, agent_id)
            
            swap_details = {
                'success': success,
                'route_key': route_key,
                'type': 'buy',
                'agent_id': agent_id,
                'input_amount': quote['input_amount'],
                'output_amount': quote['output_amount'],
                'price': quote['price'],
                'fee': quote['fee'],
                'timestamp': self.model.schedule.time,
                'nft_id': nft_id if success else None
            }
            
            # Record trade
            pool['trades'].append(swap_details)
            
            # Update price history
            self.price_history[route_key].append(
                (self.model.schedule.time, quote['price'])
            )
            
            # Limit history length
            if len(self.price_history[route_key]) > 100:
                self.price_history[route_key].pop(0)
            
            return success, swap_details
        else:
            # Agent is selling service tokens for payment tokens
            new_service_tokens = pool['service_tokens'] + quote['input_amount']
            new_payment_tokens = pool['payment_tokens'] - quote['output_amount']
            
            # Update pool
            pool['service_tokens'] = new_service_tokens
            pool['payment_tokens'] = new_payment_tokens
            pool['k'] = new_service_tokens * new_payment_tokens  # Update k slightly due to fees
            pool['last_price'] = quote['price']
            
            swap_details = {
                'success': True,
                'route_key': route_key,
                'type': 'sell',
                'agent_id': agent_id,
                'input_amount': quote['input_amount'],
                'output_amount': quote['output_amount'],
                'price': quote['price'],
                'fee': quote['fee'],
                'timestamp': self.model.schedule.time
            }
            
            # Record trade
            pool['trades'].append(swap_details)
            
            # Update price history
            self.price_history[route_key].append(
                (self.model.schedule.time, quote['price'])
            )
            
            # Limit history length
            if len(self.price_history[route_key]) > 100:
                self.price_history[route_key].pop(0)
            
            return True, swap_details
    
    def add_liquidity(self, route_key, service_tokens, payment_tokens, provider_id=None):
        """
        Add liquidity to a pool.
        
        Args:
            route_key: Route key for the pool
            service_tokens: Amount of service tokens to add
            payment_tokens: Amount of payment tokens to add
            provider_id: ID of provider adding liquidity (optional)
            
        Returns:
            Success status and LP token amount
        """
        if route_key not in self.liquidity_pools:
            return False, 0
            
        pool = self.liquidity_pools[route_key]
        
        # Calculate the correct ratio
        current_ratio = pool['payment_tokens'] / pool['service_tokens']
        
        # Ensure tokens are added in the current ratio
        correct_payment = service_tokens * current_ratio
        
        if abs(payment_tokens - correct_payment) / correct_payment > 0.05:
            # More than 5% off, reject
            self.logger.warning(f"Liquidity ratio mismatch: expected {correct_payment}, got {payment_tokens}")
            return False, 0
            
        # Add liquidity
        pool['service_tokens'] += service_tokens
        pool['payment_tokens'] += payment_tokens
        pool['k'] = pool['service_tokens'] * pool['payment_tokens']
        
        # Add provider to pool if specified
        if provider_id and provider_id not in pool['providers']:
            pool['providers'].append(provider_id)
            
        self.logger.info(f"Added liquidity to pool {route_key}: {service_tokens} service, {payment_tokens} payment")
        
        # Return LP tokens (simplified version - in reality would be based on share of pool)
        lp_tokens = math.sqrt(service_tokens * payment_tokens)
        return True, lp_tokens
    
    def remove_liquidity(self, route_key, lp_tokens, provider_id=None):
        """
        Remove liquidity from a pool.
        
        Args:
            route_key: Route key for the pool
            lp_tokens: LP tokens to burn
            provider_id: ID of provider removing liquidity (optional)
            
        Returns:
            Success status, service tokens, and payment tokens
        """
        if route_key not in self.liquidity_pools:
            return False, 0, 0
            
        pool = self.liquidity_pools[route_key]
        
        # Calculate total LP tokens (simplified)
        total_lp = math.sqrt(pool['service_tokens'] * pool['payment_tokens'])
        
        # Calculate share of pool
        share = lp_tokens / total_lp
        
        if share > 1:
            self.logger.warning(f"Invalid LP token amount: {lp_tokens} > {total_lp}")
            return False, 0, 0
            
        # Calculate tokens to return
        service_out = pool['service_tokens'] * share
        payment_out = pool['payment_tokens'] * share
        
        # Remove liquidity
        pool['service_tokens'] -= service_out
        pool['payment_tokens'] -= payment_out
        pool['k'] = pool['service_tokens'] * pool['payment_tokens']
        
        # Remove provider if specified and no liquidity left
        if provider_id and provider_id in pool['providers']:
            # This is simplified - in reality would track provider's share
            if lp_tokens > total_lp * 0.9:  # If removing more than 90% of their stake
                pool['providers'].remove(provider_id)
                
        self.logger.info(f"Removed liquidity from pool {route_key}: {service_out} service, {payment_out} payment")
        
        return True, service_out, payment_out
    
    def update_pools(self):
        """
        Perform regular updates on all pools.
        """
        current_time = self.model.schedule.time
        
        for route_key, pool in self.liquidity_pools.items():
            # Adjust prices based on external factors if needed
            self._adjust_pool_for_market_conditions(route_key, pool)
            
            # Remove stale trades
            pool['trades'] = [t for t in pool['trades'] 
                             if current_time - t['timestamp'] < 24 * 3600]  # Keep last 24h
    
    def get_popular_routes(self, limit=10):
        """
        Get the most active routes by trading volume.
        
        Args:
            limit: Maximum number of routes to return
            
        Returns:
            List of route keys sorted by trading volume
        """
        route_volumes = []
        
        for route_key, pool in self.liquidity_pools.items():
            # Calculate recent volume (last 24h)
            current_time = self.model.schedule.time
            recent_trades = [t for t in pool['trades'] 
                           if current_time - t['timestamp'] < 24 * 3600]
            
            volume = sum(t['input_amount'] for t in recent_trades)
            route_volumes.append((route_key, volume))
            
        # Sort by volume, descending
        route_volumes.sort(key=lambda x: x[1], reverse=True)
        
        return [r[0] for r in route_volumes[:limit]]
    
    def get_pool_stats(self, route_key):
        """
        Get statistics for a specific pool.
        
        Args:
            route_key: Pool route key
            
        Returns:
            Dict with pool statistics
        """
        if route_key not in self.liquidity_pools:
            return None
            
        pool = self.liquidity_pools[route_key]
        
        # Calculate 24h trading volume
        current_time = self.model.schedule.time
        recent_trades = [t for t in pool['trades'] 
                       if current_time - t['timestamp'] < 24 * 3600]
        
        volume_24h = sum(t['input_amount'] for t in recent_trades)
        
        # Calculate price change
        price_history = self.price_history.get(route_key, [])
        if len(price_history) >= 2:
            start_price = price_history[0][1]
            current_price = price_history[-1][1]
            price_change = (current_price - start_price) / start_price * 100
        else:
            price_change = 0
            
        return {
            'route_key': route_key,
            'origin': pool['origin'],
            'destination': pool['destination'],
            'current_price': pool['last_price'],
            'price_change_percent': price_change,
            'service_tokens': pool['service_tokens'],
            'payment_tokens': pool['payment_tokens'],
            'trades_24h': len(recent_trades),
            'volume_24h': volume_24h,
            'provider_count': len(pool['providers']),
            'created_at': pool['created_at']
        }
    
    def _find_closest_pool(self, origin, destination):
        """
        Find the closest matching pool for a route.
        
        Args:
            origin: Origin coordinates
            destination: Destination coordinates
            
        Returns:
            Route key of closest pool or None
        """
        origin = tuple(origin)
        destination = tuple(destination)
        
        # First check for exact match
        route_key = f"{origin}_{destination}"
        if route_key in self.liquidity_pools:
            return route_key
            
        # Search for close matches
        best_match = None
        best_score = float('inf')
        
        for pool_key, pool in self.liquidity_pools.items():
            # Calculate distance between origins
            origin_distance = math.sqrt(
                (origin[0] - pool['origin'][0])**2 + 
                (origin[1] - pool['origin'][1])**2
            )
            
            # Calculate distance between destinations
            dest_distance = math.sqrt(
                (destination[0] - pool['destination'][0])**2 + 
                (destination[1] - pool['destination'][1])**2
            )
            
            # Combined score (lower is better)
            score = origin_distance + dest_distance
            
            # Only consider reasonably close matches
            if score < 10 and score < best_score:
                best_score = score
                best_match = pool_key
                
        return best_match
    
    def _calculate_slippage(self, pool, execution_price, is_buy):
        """
        Calculate expected slippage for a trade.
        
        Args:
            pool: Pool data
            execution_price: Execution price
            is_buy: Whether buying or selling
            
        Returns:
            Slippage percentage
        """
        # Calculate spot price from current pool state
        spot_price = pool['payment_tokens'] / pool['service_tokens']
        
        # Calculate slippage
        slippage = (execution_price - spot_price) / spot_price
        
        # For sell orders, slippage is negative
        if not is_buy:
            slippage = -slippage
            
        return slippage * 100  # Convert to percentage
    
    def _adjust_pool_for_market_conditions(self, route_key, pool):
        """
        Adjust pool based on external market conditions.
        
        Args:
            route_key: Route key
            pool: Pool data
        """
        # Check if we should simulate market forces
        # This is where you would integrate with external price data
        # or other market simulation factors
        
        # For simplicity, add small random fluctuations
        if random.random() < 0.2:  # 20% chance of adjustment
            # Small random adjustment (-2% to +2%)
            adjustment = 1 + (random.random() * 0.04 - 0.02)
            
            # Adjust both tokens to maintain k
            old_service = pool['service_tokens']
            old_payment = pool['payment_tokens']
            
            # Adjust service tokens, recalculate payment tokens to maintain k
            new_service = old_service * adjustment
            new_payment = pool['k'] / new_service
            
            pool['service_tokens'] = new_service
            pool['payment_tokens'] = new_payment
            
            # Record price change
            new_price = new_payment / new_service
            self.price_history[route_key].append(
                (self.model.schedule.time, new_price)
            )
            
            self.logger.debug(f"Adjusted pool {route_key} by factor {adjustment:.4f}")

"""
AMM Route Analyzer for Decentralized MaaS

This module analyzes route data to identify which routes are good candidates
for AMM liquidity pools, helping to optimize computational resources.
"""

class AMMRouteAnalyzer:
    """
    Analyzes route patterns to identify good candidates for AMM liquidity pools,
    optimizing computational resources by focusing on popular routes.
    """
    
    def __init__(self, model, marketplace_integration, min_transactions=5):
        """
        Initialize the AMM route analyzer.
        
        Args:
            model: The ABM model
            marketplace_integration: MarketplaceIntegration instance
            min_transactions: Minimum transactions for a route to be considered popular
        """
        self.model = model
        self.marketplace = marketplace_integration
        self.min_transactions = min_transactions
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("AMMRouteAnalyzer")
        
        # Route tracking
        self.route_stats = {}
        self.active_amm_routes = set()
        self.route_clusters = {}
        self.route_forecasts = {}
        
        # Thresholds and parameters
        self.params = {
            'popularity_threshold': 0.6,        # Minimum popularity score for AMM (0-1)
            'volatility_threshold': 0.4,        # Maximum volatility for stable AMM pools (0-1)
            'cluster_distance': 5.0,            # Maximum distance to consider routes part of same cluster
            'min_route_efficiency': 0.7,        # Minimum efficiency score for AMM (0-1)
            'computational_budget': 20,          # Maximum number of AMM pools to maintain
            'update_interval': 86400,           # How often to update analysis (in simulation time)
            'forecast_horizon': 7 * 86400        # How far ahead to forecast (default 7 days)
        }
        
        self.logger.info(f"AMM Route Analyzer initialized with {min_transactions} min transactions")
    
    def update(self):
        """
        Update route analysis based on market activity.
        Only runs full analysis periodically to conserve computational resources.
        """
        current_time = self.model.schedule.time
        
        # Run full analysis periodically
        if current_time % self.params['update_interval'] == 0:
            self.analyze_routes()
            self.logger.info("Completed periodic route analysis")
        
        # Check for inactive AMM pools that should be removed
        self._check_inactive_pools()
    
    def analyze_routes(self):
        """
        Perform comprehensive analysis of all routes to identify AMM candidates.
        """
        # Get market analytics
        analytics = self.marketplace.nft_marketplace.get_market_analytics()
        
        # Extract transaction data by route
        route_transactions = self._get_route_transactions()
        
        # Calculate statistics for each route
        for route_key, transactions in route_transactions.items():
            if len(transactions) < self.min_transactions:
                continue
                
            # Extract locations from route key
            try:
                parts = route_key.split('_')
                if len(parts) >= 3:  # Includes time window
                    origin = eval(parts[0])
                    destination = eval(parts[1])
                else:
                    continue
            except:
                continue
            
            # Calculate statistics
            stats = self._calculate_route_statistics(route_key, transactions, origin, destination)
            
            # Store route statistics
            self.route_stats[route_key] = stats
            
            self.logger.debug(f"Analyzed route {route_key}: "
                            f"Pop: {stats['popularity']:.2f}, "
                            f"Vol: {stats['volatility']:.2f}, "
                            f"Eff: {stats['efficiency']:.2f}")
        
        # Cluster similar routes
        self._cluster_routes()
        
        # Forecast future demand
        self._forecast_route_demand()
        
        # Identify optimal AMM pool candidates
        recommended_routes = self.recommend_amm_routes()
        
        # Log recommendations
        self.logger.info(f"Recommended {len(recommended_routes)} routes for AMM pools")
        for route in recommended_routes[:5]:  # Log top 5
            self.logger.info(f"  Route {route['route_key']}: score {route['score']:.2f}")
        
        # Track active AMM routes
        self.active_amm_routes = set(route['route_key'] for route in recommended_routes)
    
    def _get_route_transactions(self):
        """
        Extract and organize transaction data by route.
        
        Returns:
            Dict mapping route keys to lists of transactions
        """
        route_transactions = defaultdict(list)
        
        # Get transactions from marketplace
        for tx in self.marketplace.nft_marketplace.transaction_history:
            nft_id = tx.get('nft_id')
            if nft_id not in self.marketplace.nft_marketplace.listings:
                continue
                
            listing = self.marketplace.nft_marketplace.listings[nft_id]
            if 'details' not in listing:
                continue
                
            nft_details = listing['details']
            route_key = self.marketplace.nft_marketplace._get_route_key(nft_details)
            
            # Add transaction to route
            route_transactions[route_key].append({
                'nft_id': nft_id,
                'price': tx.get('price', 0),
                'time': tx.get('time', 0),
                'buyer_id': tx.get('buyer_id'),
                'seller_id': tx.get('seller_id'),
                'origin': nft_details.get('origin'),
                'destination': nft_details.get('destination'),
                'service_time': nft_details.get('service_time')
            })
        
        return route_transactions
    
    def _calculate_route_statistics(self, route_key, transactions, origin, destination):
        """
        Calculate detailed statistics for a route.
        
        Args:
            route_key: Route key
            transactions: List of transactions for this route
            origin: Origin coordinates
            destination: Destination coordinates
            
        Returns:
            Dict with route statistics
        """
        # Basic transaction statistics
        tx_count = len(transactions)
        current_time = self.model.schedule.time
        
        # Calculate time distributions
        listing_times = [tx.get('listing_time', tx.get('time', 0) - 86400) for tx in transactions]
        service_times = [tx.get('service_time', tx.get('time', 0) + 86400) for tx in transactions]
        
        # Calculate price statistics
        prices = [tx.get('price', 0) for tx in transactions]
        avg_price = np.mean(prices) if prices else 0
        price_volatility = np.std(prices) / avg_price if avg_price > 0 else 0
        
        # Calculate temporal distribution
        if service_times:
            min_time = min(service_times)
            max_time = max(service_times)
            time_span = max(1, max_time - min_time)
            
            # Create time buckets (8 buckets across the time span)
            bucket_size = time_span / 8
            buckets = [0] * 8
            
            for t in service_times:
                bucket = min(7, int((t - min_time) / bucket_size))
                buckets[bucket] += 1
            
            # Calculate time distribution evenness (0-1, higher is more even)
            total = sum(buckets)
            if total > 0:
                proportions = [b/total for b in buckets]
                time_evenness = 1 - np.std(proportions)
            else:
                time_evenness = 0
        else:
            time_evenness = 0
        
        # Calculate spatial statistics
        distance = math.sqrt(
            (destination[0] - origin[0])**2 +
            (destination[1] - origin[1])**2
        )
        
        # Calculate recency factor (more weight to recent transactions)
        if transactions:
            transaction_times = [tx.get('time', 0) for tx in transactions]
            latest_tx = max(transaction_times)
            recency = math.exp(-0.1 * max(0, (current_time - latest_tx) / 86400))  # Days since last tx
        else:
            recency = 0
        
        # Calculate popularity score (0-1)
        popularity = min(1.0, tx_count / 20)  # Scale: 20+ transactions = 1.0
        popularity = popularity * (0.7 + 0.3 * recency)  # Weight by recency
        
        # Calculate efficiency score (0-1)
        # Higher for routes with consistent pricing and even distribution
        efficiency = (0.6 * (1 - price_volatility) + 0.4 * time_evenness)
        
        return {
            'route_key': route_key,
            'origin': origin,
            'destination': destination,
            'tx_count': tx_count,
            'avg_price': avg_price,
            'price_volatility': price_volatility,
            'time_evenness': time_evenness,
            'distance': distance,
            'recency': recency,
            'popularity': popularity,
            'efficiency': efficiency,
            'last_transaction': max(t.get('time', 0) for t in transactions) if transactions else 0
        }
    
    def _cluster_routes(self):
        """
        Cluster similar routes to reduce computational overhead.
        Routes that are very close in origin/destination can share an AMM pool.
        """
        # Reset clusters
        self.route_clusters = {}
        
        # Sort routes by popularity for deterministic clustering
        sorted_routes = sorted(
            self.route_stats.items(),
            key=lambda x: x[1]['popularity'],
            reverse=True
        )
        
        # Assign each route to a cluster
        for route_key, stats in sorted_routes:
            # Check if this route should be the center of a new cluster
            new_cluster = True
            
            for cluster_key, cluster in self.route_clusters.items():
                cluster_center = cluster['center']
                
                # Calculate distance from this route to cluster center
                origin_distance = math.sqrt(
                    (stats['origin'][0] - cluster_center['origin'][0])**2 +
                    (stats['origin'][1] - cluster_center['origin'][1])**2
                )
                
                dest_distance = math.sqrt(
                    (stats['destination'][0] - cluster_center['destination'][0])**2 +
                    (stats['destination'][1] - cluster_center['destination'][1])**2
                )
                
                # If close enough to existing cluster, add to it
                if origin_distance <= self.params['cluster_distance'] and \
                   dest_distance <= self.params['cluster_distance']:
                    cluster['routes'].append(route_key)
                    new_cluster = False
                    break
            
            # If not assigned to existing cluster, create new one
            if new_cluster:
                self.route_clusters[route_key] = {
                    'center': stats,
                    'routes': [route_key]
                }
        
        # Calculate aggregate statistics for each cluster
        for cluster_key, cluster in self.route_clusters.items():
            routes = cluster['routes']
            
            # Aggregate popularity (weighted average)
            total_weight = 0
            weighted_popularity = 0
            weighted_efficiency = 0
            
            for route in routes:
                if route in self.route_stats:
                    stats = self.route_stats[route]
                    weight = stats['tx_count']
                    weighted_popularity += stats['popularity'] * weight
                    weighted_efficiency += stats['efficiency'] * weight
                    total_weight += weight
            
            if total_weight > 0:
                cluster['popularity'] = weighted_popularity / total_weight
                cluster['efficiency'] = weighted_efficiency / total_weight
                cluster['tx_count'] = total_weight
            else:
                cluster['popularity'] = 0
                cluster['efficiency'] = 0
                cluster['tx_count'] = 0
        
        self.logger.info(f"Clustered {len(self.route_stats)} routes into {len(self.route_clusters)} groups")
    
    def _forecast_route_demand(self):
        """
        Forecast future demand for routes based on historical patterns.
        Uses simple time series analysis.
        """
        current_time = self.model.schedule.time
        
        # Prepare forecast data
        self.route_forecasts = {}
        
        # Get transaction history from marketplace
        transaction_history = self.marketplace.nft_marketplace.transaction_history
        
        # Group transactions by route and time window
        route_time_series = defaultdict(lambda: defaultdict(list))
        
        for tx in transaction_history:
            nft_id = tx.get('nft_id')
            tx_time = tx.get('time', 0)
            
            # Skip very old transactions
            if current_time - tx_time > 30 * 86400:  # Older than 30 days
                continue
                
            if nft_id in self.marketplace.nft_marketplace.listings:
                listing = self.marketplace.nft_marketplace.listings[nft_id]
                if 'details' in listing:
                    nft_details = listing['details']
                    route_key = self.marketplace.nft_marketplace._get_route_key(nft_details)
                    
                    # Group by day for time series
                    day = int(tx_time / 86400)
                    route_time_series[route_key][day].append(tx.get('price', 0))
        
        # Calculate forecasts for each route
        for route_key, time_series in route_time_series.items():
            if len(time_series) < 3:  # Need at least 3 days of data
                continue
                
            # Convert to ordered time series
            days = sorted(time_series.keys())
            
            # Count transactions per day
            counts = [len(time_series[day]) for day in days]
            
            # Simple forecast using weighted moving average
            if len(counts) >= 3:
                weights = [0.1, 0.3, 0.6]  # More weight to recent days
                while len(counts) > len(weights):
                    weights.insert(0, weights[0] * 0.5)  # Add weights for older days
                
                weights = weights[-len(counts):]  # Trim if needed
                
                # Normalize weights
                weights = [w / sum(weights) for w in weights]
                
                # Calculate weighted forecast
                forecast = sum(w * c for w, c in zip(weights, counts))
                
                # Store forecast
                self.route_forecasts[route_key] = {
                    'current_volume': counts[-1] if counts else 0,
                    'forecast_volume': forecast,
                    'trend': forecast - counts[-1] if counts else 0,
                    'days_observed': len(days)
                }
        
        self.logger.info(f"Generated forecasts for {len(self.route_forecasts)} routes")
    
    def _check_inactive_pools(self):
        """
        Check for inactive AMM pools that should be removed to free up resources.
        """
        # Get current AMM pools
        amm_pools = self.marketplace.amm.liquidity_pools
        current_time = self.model.schedule.time
        
        # Check each pool for inactivity
        for route_key, pool in list(amm_pools.items()):
            # Skip if this is a recommended route
            if route_key in self.active_amm_routes:
                continue
                
            # Check last transaction
            last_update = pool.get('last_update_time', 0)
            inactive_time = current_time - last_update
            
            # If inactive for more than 3 days and not in recommended routes
            if inactive_time > 3 * 86400 and len(pool.get('nfts', [])) == 0:
                self.logger.info(f"Marking inactive AMM pool {route_key} for removal")
                
                # Remove from AMM pools (would be handled by AMM's update method)
                # This just marks it as a candidate for removal
                pool['marked_for_removal'] = True
    
    def recommend_amm_routes(self):
        """
        Recommend routes for AMM liquidity pools based on analysis.
        
        Returns:
            List of recommended routes with scores
        """
        # Generate candidate list based on individual routes
        candidates = []
        
        # First add routes from active clusters
        for cluster_key, cluster in self.route_clusters.items():
            # Skip if doesn't meet minimum criteria
            if cluster['popularity'] < self.params['popularity_threshold']:
                continue
                
            if cluster['tx_count'] < self.min_transactions:
                continue
                
            # Calculate route score
            efficiency = cluster['efficiency']
            popularity = cluster['popularity']
            
            # Get forecast data if available
            forecast_bonus = 0
            if cluster_key in self.route_forecasts:
                forecast = self.route_forecasts[cluster_key]
                trend = forecast['trend']
                
                # Give bonus for positive trends
                if trend > 0:
                    forecast_bonus = min(0.2, trend / 5)  # Cap at 0.2
            
            # Calculate final score
            score = (0.6 * popularity) + (0.3 * efficiency) + (0.1 * forecast_bonus)
            
            # Add to candidates
            candidates.append({
                'route_key': cluster_key,
                'score': score,
                'origin': cluster['center']['origin'],
                'destination': cluster['center']['destination'],
                'popularity': popularity,
                'efficiency': efficiency,
                'forecast_bonus': forecast_bonus,
                'tx_count': cluster['tx_count']
            })
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit to computational budget
        recommended = candidates[:self.params['computational_budget']]
        
        return recommended
    
    def should_use_amm(self, origin, destination, service_time):
        """
        Determine if a specific route should use AMM.
        
        Args:
            origin: Origin coordinates
            destination: Destination coordinates
            service_time: Service time
            
        Returns:
            Boolean indicating if AMM should be used
        """
        # Create route details
        route_details = {
            'origin': origin,
            'destination': destination,
            'service_time': service_time
        }
        
        # Get route key
        route_key = self.marketplace.nft_marketplace._get_route_key(route_details)
        
        # Check if in active AMM routes
        if route_key in self.active_amm_routes:
            return True
        
        # Check if in AMM pool
        if route_key in self.marketplace.amm.liquidity_pools:
            return True
        
        # Check if part of an active cluster
        for cluster_key, cluster in self.route_clusters.items():
            if cluster_key in self.active_amm_routes:
                # Check if this route is close to cluster center
                center = cluster['center']
                
                # Calculate distances
                origin_distance = math.sqrt(
                    (origin[0] - center['origin'][0])**2 +
                    (origin[1] - center['origin'][1])**2
                )
                
                dest_distance = math.sqrt(
                    (destination[0] - center['destination'][0])**2 +
                    (destination[1] - center['destination'][1])**2
                )
                
                # If close enough to cluster center, use AMM
                if origin_distance <= self.params['cluster_distance'] and \
                   dest_distance <= self.params['cluster_distance']:
                    return True
        
        return False
    
    def suggest_liquidity_providers(self, route_key):
        """
        Suggest providers who should add liquidity to a route.
        
        Args:
            route_key: Route key
            
        Returns:
            List of provider IDs who would benefit from adding liquidity
        """
        # Get route data
        if route_key not in self.route_stats:
            return []
            
        route_data = self.route_stats[route_key]
        
        # Identify providers who serviced this route
        providers = set()
        
        # Get transactions for this route
        for tx in self.marketplace.nft_marketplace.transaction_history:
            nft_id = tx.get('nft_id')
            if nft_id not in self.marketplace.nft_marketplace.listings:
                continue
                
            listing = self.marketplace.nft_marketplace.listings[nft_id]
            if 'details' not in listing:
                continue
                
            nft_details = listing['details']
            tx_route_key = self.marketplace.nft_marketplace._get_route_key(nft_details)
            
            if tx_route_key == route_key:
                # Find provider who created this NFT
                provider_id = listing.get('details', {}).get('provider_id')
                if provider_id:
                    providers.add(provider_id)
        
        # Sort providers by activity on this route
        provider_count = {}
        for provider_id in providers:
            provider_count[provider_id] = 0
            
            # Count transactions by this provider
            for tx in self.marketplace.nft_marketplace.transaction_history:
                nft_id = tx.get('nft_id')
                if nft_id in self.marketplace.nft_marketplace.listings:
                    listing = self.marketplace.nft_marketplace.listings[nft_id]
                    if listing.get('details', {}).get('provider_id') == provider_id:
                        tx_route_key = self.marketplace.nft_marketplace._get_route_key(listing.get('details', {}))
                        if tx_route_key == route_key:
                            provider_count[provider_id] += 1
        
        # Sort by count
        sorted_providers = sorted(provider_count.items(), key=lambda x: x[1], reverse=True)
        
        # Return provider IDs
        return [p[0] for p in sorted_providers]
    
    def get_route_recommendations(self):
        """
        Get recommendations for providers and routes.
        
        Returns:
            Dict with recommendations
        """
        # Recommend routes for AMM pools
        recommended_routes = self.recommend_amm_routes()
        
        # For each recommended route, suggest providers
        for route in recommended_routes:
            route_key = route['route_key']
            route['suggested_providers'] = self.suggest_liquidity_providers(route_key)
        
        return {
            'recommended_routes': recommended_routes,
            'route_clusters': self.route_clusters,
            'route_forecasts': self.route_forecasts,
            'amm_routes': list(self.active_amm_routes),
            'route_stats': self.route_stats
        }

"""
Time-Sensitive Pricing Module for Decentralized MaaS

This module implements time-sensitive pricing models for mobility services,
providing mathematical functions to model value decay as service time approaches.
"""

class TimeSensitivePricing:
    """
    Handles different pricing models for time-sensitive mobility assets,
    where asset value changes as the service time approaches.
    """
    
    def __init__(self, base_pricing_model="exponential_decay"):
        """
        Initialize the time-sensitive pricing model.
        
        Args:
            base_pricing_model: Default pricing model to use
                ("exponential_decay", "linear_decay", "threshold_decay", "reverse_j")
        """
        self.base_pricing_model = base_pricing_model
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("TimeSensitivePricing")
        
        # Default parameters for different pricing models
        self.pricing_params = {
            "exponential_decay": {
                "decay_rate": 0.1,           # Base decay rate
                "time_sensitivity": 1.5,      # How much time remaining affects decay
                "min_price_ratio": 0.5,       # Minimum price as ratio of initial
                "emergency_threshold": 900    # Time threshold (seconds) for emergency pricing
            },
            "linear_decay": {
                "start_decay_ratio": 0.7,     # When decay starts as ratio of total time
                "min_price_ratio": 0.6        # Minimum price as ratio of initial
            },
            "threshold_decay": {
                "thresholds": [0.8, 0.5, 0.25, 0.1],  # Remaining time thresholds
                "price_ratios": [0.9, 0.7, 0.5, 0.3]  # Price ratios at each threshold
            },
            "reverse_j": {
                "plateau_end": 0.6,          # When price starts to drop, as ratio of total time
                "cliff_start": 0.2,          # When price drops rapidly, as ratio of total time
                "min_price_ratio": 0.4       # Minimum price as ratio of initial
            }
        }
        
        self.logger.info(f"Time-sensitive pricing initialized with {base_pricing_model} model")
    
    def calculate_price(self, initial_price, listing_time, service_time, current_time, 
                        model=None, custom_params=None):
        """
        Calculate the current price based on time remaining until service.
        
        Args:
            initial_price: Initial listing price
            listing_time: When the service was listed
            service_time: When the service will be provided
            current_time: Current time
            model: Pricing model to use (defaults to base model)
            custom_params: Optional custom parameters for the model
            
        Returns:
            Current calculated price
        """
        # Use default model if none specified
        if model is None:
            model = self.base_pricing_model
            
        # Get model parameters (custom or default)
        params = custom_params if custom_params else self.pricing_params[model].copy()
        
        # Calculate time values
        time_elapsed = current_time - listing_time
        total_listing_time = service_time - listing_time
        time_remaining = service_time - current_time
        
        # Handle expired services
        if time_remaining <= 0:
            return initial_price * params.get("min_price_ratio", 0.5)
            
        # Time remaining as a ratio of total listing time
        time_ratio = time_remaining / total_listing_time
        
        # Select pricing model
        if model == "exponential_decay":
            return self._exponential_decay_price(initial_price, time_elapsed, time_ratio, params)
        elif model == "linear_decay":
            return self._linear_decay_price(initial_price, time_ratio, params)
        elif model == "threshold_decay":
            return self._threshold_decay_price(initial_price, time_ratio, params)
        elif model == "reverse_j":
            return self._reverse_j_price(initial_price, time_ratio, params)
        else:
            self.logger.warning(f"Unknown pricing model: {model}, using exponential_decay")
            return self._exponential_decay_price(initial_price, time_elapsed, time_ratio, 
                                               self.pricing_params["exponential_decay"])
    
    def _exponential_decay_price(self, initial_price, time_elapsed, time_ratio, params):
        """
        Calculate price using exponential decay model.
        
        Price decays exponentially with elapsed time, with decay accelerating 
        as service time approaches.
        
        Args:
            initial_price: Initial price
            time_elapsed: Time elapsed since listing (in seconds)
            time_ratio: Ratio of time remaining to total listing time
            params: Model parameters
            
        Returns:
            Current price
        """
        decay_rate = params["decay_rate"]
        time_sensitivity = params["time_sensitivity"]
        min_price_ratio = params["min_price_ratio"]
        
        # Scale decay by time remaining (faster decay as time approaches)
        # Use max to avoid division by zero or negative values
        time_factor = 1.0 + time_sensitivity * (1.0 - time_ratio)
        
        # When time is very short, apply emergency pricing
        emergency_threshold = params.get("emergency_threshold", 900)  # 15 minutes default
        if time_ratio < 0.1 and time_elapsed < emergency_threshold:
            # Accelerate decay for last-minute services
            time_factor *= 1.5
        
        # Calculate decay factor
        decay = math.exp(-decay_rate * time_elapsed * time_factor / 86400)  # Normalize to days
        
        # Calculate price with floor
        price = max(initial_price * min_price_ratio, initial_price * decay)
        
        return price
    
    def _linear_decay_price(self, initial_price, time_ratio, params):
        """
        Calculate price using linear decay model.
        
        Price remains stable until a threshold, then decreases linearly.
        
        Args:
            initial_price: Initial price
            time_ratio: Ratio of time remaining to total listing time
            params: Model parameters
            
        Returns:
            Current price
        """
        start_decay_ratio = params["start_decay_ratio"]
        min_price_ratio = params["min_price_ratio"]
        
        if time_ratio >= start_decay_ratio:
            # Before decay starts, price is unchanged
            return initial_price
        elif time_ratio <= 0:
            # Service time passed, minimum price
            return initial_price * min_price_ratio
        else:
            # Linear decay between start_decay_ratio and 0
            decay_progress = (start_decay_ratio - time_ratio) / start_decay_ratio
            price_ratio = 1.0 - decay_progress * (1.0 - min_price_ratio)
            return initial_price * price_ratio
    
    def _threshold_decay_price(self, initial_price, time_ratio, params):
        """
        Calculate price using threshold decay model.
        
        Price drops at specific time thresholds, creating a step function.
        
        Args:
            initial_price: Initial price
            time_ratio: Ratio of time remaining to total listing time
            params: Model parameters
            
        Returns:
            Current price
        """
        thresholds = params["thresholds"]
        price_ratios = params["price_ratios"]
        
        # Default to initial price
        price_ratio = 1.0
        
        # Find applicable threshold
        for i, threshold in enumerate(thresholds):
            if time_ratio <= threshold:
                price_ratio = price_ratios[i]
                break
                
        return initial_price * price_ratio
    
    def _reverse_j_price(self, initial_price, time_ratio, params):
        """
        Calculate price using reverse J-curve model.
        
        Price remains stable for a while, then drops gradually,
        and finally drops rapidly near the service time (like a reverse J-curve).
        
        Args:
            initial_price: Initial price
            time_ratio: Ratio of time remaining to total listing time
            params: Model parameters
            
        Returns:
            Current price
        """
        plateau_end = params["plateau_end"]
        cliff_start = params["cliff_start"]
        min_price_ratio = params["min_price_ratio"]
        
        if time_ratio >= plateau_end:
            # On the plateau, price is unchanged
            return initial_price
        elif time_ratio <= 0:
            # Service time passed, minimum price
            return initial_price * min_price_ratio
        elif time_ratio <= cliff_start:
            # In the cliff region, price drops rapidly (quadratic)
            progress = time_ratio / cliff_start
            price_ratio = min_price_ratio + (1.0 - min_price_ratio) * (progress ** 2)
            return initial_price * price_ratio
        else:
            # In the gradual descent region
            progress = (time_ratio - cliff_start) / (plateau_end - cliff_start)
            price_ratio = 1.0 - (1.0 - progress) * (1.0 - cliff_start) / (plateau_end - cliff_start)
            return initial_price * price_ratio
    
    def optimize_parameters(self, transaction_data, model=None):
        """
        Optimize pricing model parameters based on historical transaction data.
        
        Args:
            transaction_data: List of dicts with transaction details
                Each dict should contain: initial_price, listing_time, service_time, sale_time, sale_price
            model: Pricing model to optimize (defaults to base model)
            
        Returns:
            Optimized parameters dict
        """
        if model is None:
            model = self.base_pricing_model
            
        if len(transaction_data) < 10:
            self.logger.warning("Not enough transaction data for optimization")
            return self.pricing_params[model]
            
        # Simple optimization: try different parameter values and pick the best
        best_params = self.pricing_params[model].copy()
        best_error = float('inf')
        
        # Parameter search ranges depend on the model
        if model == "exponential_decay":
            param_ranges = {
                "decay_rate": np.linspace(0.05, 0.3, 6),
                "time_sensitivity": np.linspace(0.5, 2.5, 5),
                "min_price_ratio": np.linspace(0.3, 0.7, 5)
            }
            
            # Test combinations of parameters
            for decay_rate in param_ranges["decay_rate"]:
                for time_sensitivity in param_ranges["time_sensitivity"]:
                    for min_price_ratio in param_ranges["min_price_ratio"]:
                        test_params = {
                            "decay_rate": decay_rate,
                            "time_sensitivity": time_sensitivity,
                            "min_price_ratio": min_price_ratio
                        }
                        
                        # Calculate error for these parameters
                        error = self._calculate_prediction_error(transaction_data, model, test_params)
                        
                        if error < best_error:
                            best_error = error
                            best_params = test_params
        
        elif model == "linear_decay":
            param_ranges = {
                "start_decay_ratio": np.linspace(0.5, 0.9, 5),
                "min_price_ratio": np.linspace(0.3, 0.8, 6)
            }
            
            for start_decay_ratio in param_ranges["start_decay_ratio"]:
                for min_price_ratio in param_ranges["min_price_ratio"]:
                    test_params = {
                        "start_decay_ratio": start_decay_ratio,
                        "min_price_ratio": min_price_ratio
                    }
                    
                    error = self._calculate_prediction_error(transaction_data, model, test_params)
                    
                    if error < best_error:
                        best_error = error
                        best_params = test_params
        
        # Add similar optimizations for other models as needed
        
        self.logger.info(f"Optimized {model} parameters with error {best_error:.3f}")
        return best_params
    
    def _calculate_prediction_error(self, transaction_data, model, params):
        """
        Calculate mean squared error for a set of parameters.
        
        Args:
            transaction_data: List of transaction dicts
            model: Pricing model
            params: Parameters to test
            
        Returns:
            Mean squared error
        """
        total_error = 0.0
        
        for tx in transaction_data:
            # Predict price at sale time
            predicted_price = self.calculate_price(
                tx["initial_price"], 
                tx["listing_time"], 
                tx["service_time"], 
                tx["sale_time"], 
                model, 
                params
            )
            
            # Calculate squared error
            error = (predicted_price - tx["sale_price"]) ** 2
            total_error += error
            
        return total_error / len(transaction_data)
    
    def suggest_pricing_model(self, route_characteristics):
        """
        Suggest an appropriate pricing model based on route characteristics.
        
        Args:
            route_characteristics: Dict with route information
                - popularity: Route popularity score (0-1)
                - volatility: Price volatility score (0-1)
                - advance_booking: Typical advance booking time (seconds)
                - demand_pattern: "steady", "peaky", or "variable"
            
        Returns:
            Tuple of (recommended_model, params, explanation)
        """
        popularity = route_characteristics.get("popularity", 0.5)
        volatility = route_characteristics.get("volatility", 0.5)
        advance_booking = route_characteristics.get("advance_booking", 86400)  # Default 1 day
        demand_pattern = route_characteristics.get("demand_pattern", "steady")
        
        # Select model based on characteristics
        if popularity > 0.8 and volatility > 0.7:
            # High popularity, high volatility -> exponential decay with fast rate
            model = "exponential_decay"
            params = self.pricing_params[model].copy()
            params["decay_rate"] = 0.2  # Faster decay
            params["time_sensitivity"] = 2.0  # More time sensitivity
            explanation = ("High popularity and volatility routes benefit from exponential decay "
                         "with faster rates to quickly respond to changing market conditions.")
        
        elif popularity > 0.7 and demand_pattern == "peaky":
            # Popular route with peaky demand -> threshold decay
            model = "threshold_decay"
            params = self.pricing_params[model].copy()
            # Adjust thresholds to match typical booking patterns
            explanation = ("Routes with peaky demand benefit from threshold-based pricing "
                         "to capture value during high-demand periods while ensuring "
                         "availability during low-demand periods.")
        
        elif advance_booking > 7 * 86400:  # > 7 days
            # Long advance booking -> reverse J curve
            model = "reverse_j"
            params = self.pricing_params[model].copy()
            params["plateau_end"] = 0.7  # Longer plateau
            explanation = ("Routes with long advance booking periods benefit from stable "
                         "pricing initially, followed by accelerating discounts as "
                         "service time approaches.")
        
        elif volatility < 0.3 and demand_pattern == "steady":
            # Low volatility, steady demand -> linear decay
            model = "linear_decay"
            params = self.pricing_params[model].copy()
            params["start_decay_ratio"] = 0.6  # Start decay earlier
            explanation = ("Routes with steady demand and low volatility benefit from "
                         "simple linear decay models that are predictable for users.")
        
        else:
            # Default to standard exponential decay
            model = "exponential_decay"
            params = self.pricing_params[model].copy()
            explanation = ("Standard exponential decay provides a good balance of "
                         "time sensitivity and predictable pricing behavior.")
        
        return model, params, explanation
    
    def get_pricing_curve(self, initial_price, listing_time, service_time, model=None, params=None):
        """
        Generate pricing curve data for visualization.
        
        Args:
            initial_price: Initial price
            listing_time: Listing timestamp
            service_time: Service timestamp
            model: Pricing model (defaults to base model)
            params: Model parameters (defaults to standard parameters)
            
        Returns:
            Dict with time points and prices
        """
        if model is None:
            model = self.base_pricing_model
            
        if params is None:
            params = self.pricing_params[model].copy()
            
        # Generate time points
        total_time = service_time - listing_time
        time_points = []
        current = listing_time
        
        # Generate more points as we get closer to service time
        while current <= service_time:
            time_points.append(current)
            
            # Adaptive time steps - closer together near service time
            time_remaining = service_time - current
            if time_remaining < 3600:  # Less than 1 hour
                step = 300  # 5 minutes
            elif time_remaining < 86400:  # Less than 1 day
                step = 3600  # 1 hour
            else:
                step = 21600  # 6 hours
                
            current += step
            
        # Add one point after service time
        time_points.append(service_time + 3600)
        
        # Calculate prices for each time point
        prices = [self.calculate_price(initial_price, listing_time, service_time, t, model, params) 
                for t in time_points]
        
        # Format as time since listing for easier plotting
        times_since_listing = [(t - listing_time) / 3600 for t in time_points]  # Convert to hours
        
        return {
            "model": model,
            "times": times_since_listing,
            "prices": prices,
            "service_time": (service_time - listing_time) / 3600  # Service time in hours since listing
        }