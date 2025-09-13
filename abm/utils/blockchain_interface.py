"""
Enhanced BlockchainInterface for Decentralized MaaS
Provides an optimized bridge between ABM simulation and blockchain contracts
with support for NFT marketplace operations, caching, and asynchronous transactions.
"""

import json
import hashlib
import logging
import time
import uuid
from web3 import Web3
from web3.middleware import geth_poa_middleware
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import deque
import math
from eth_account import Account
from collections import deque, defaultdict
import threading

@dataclass
class TransactionData:
    """Data structure for pending transactions"""
    tx_type: str
    function_name: str
    params: dict
    sender_id: Union[int, str]
    callback: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)
    tx_hash: Optional[str] = None
    status: str = "pending"
    gas_price: int = 2000000000  # 2 gwei
    gas_limit: int = 500000


class BlockchainInterface:
    def __init__(self, config_file="blockchain_config.json", using_hardhat=True, 
                 max_workers=None, cache_ttl=300, async_mode=True):
        """
        Initialize connection to blockchain and load contracts with enhanced performance features
        
        Args:
            config_file: Path to blockchain configuration file
            using_hardhat: Whether using local Hardhat node
            max_workers: Maximum number of worker threads for async operations
            cache_ttl: Time-to-live for cached data in seconds
            async_mode: Whether to use asynchronous transaction processing
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MarketplaceAPI")  # Renamed to reflect true purpose
        
        
        # Load configuration
        self.config = self._load_config(config_file)
        self.w3 = self._connect_to_blockchain(using_hardhat)
        self.contracts = self._load_contracts()
        self.gas_limit = 1000000  # Default gas limit
        
        # Account management
        self.accounts = {}
        self.nonce_tracker = {}
        self.nonce_lock = threading.Lock()

        # Enhanced initialization
        # Calculate optimal worker count based on your system
        if max_workers is None:
            # For I/O-bound blockchain operations, can use more threads
            import os
            max_workers = min(32, (os.cpu_count() or 1) * 4)  # 4x CPU cores, cap at 32
        
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.async_mode = async_mode
        self.cache_ttl = cache_ttl
        
        # Transaction queue and processing
        self.tx_queue = deque()
        self.pending_transactions = {}
        self.tx_nonce_map = {}  # To track nonces for each account
        self.tx_count = 0
        
        # State caching
        self.state_cache = {
            'commuters': {},
            'providers': {},
            'requests': {},
            'auctions': {},
            'offers': {},
            'nfts': {},
            'marketplace': {},
            'last_updated': {}
        }
        
        # Batch processing tracking
        self.batch_size_limit = 10
        self.current_batch = {
            'registrations': [],
            'requests': [],
            'transactions':[],
            'offers': []
        }
        
        # Statistics and monitoring
        self.stats = {
            'transactions_submitted': 0,
            'transactions_confirmed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_operations': 0
        }
        
        # Start background tasks if using async mode
        if self.async_mode:
            self.running = True
            self.tx_processor_task = None
            self.cache_updater_task = None
            self._start_background_tasks()
        
        self.logger.info("BlockchainInterface initialized with enhanced performance features")
        
        # MARKETPLACE DATABASE (off-chain storage)
        self.marketplace_db = {
            'requests': {},      # Full request data
            'offers': {},        # Full offer data
            'providers': {},     # Provider profiles
            'commuters': {},     # Commuter profiles
            'matches': {},       # Matching results
            'notifications': defaultdict(list)  # Provider notifications
        }

        self.steps_per_day = 144   # 144 steps = 1 day
        self.minutes_per_step = 10  # 10 minutes per step
        
    
        self.model = None

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.running = False
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
            
    def _start_background_tasks(self):
        """Start background tasks for async processing"""
        self.thread_pool.submit(self._transaction_processor)
        self.thread_pool.submit(self._periodic_cache_update)

    def _transaction_processor(self):
        """Continuously process queued transactions in batches."""
        try:
            while getattr(self, 'running', False):
                try:
                    self._process_transaction_batch()
                except Exception as e:
                    self.logger.debug(f"Transaction processor error: {e}")
                time.sleep(0.5)
        except Exception:
            # Ensure background thread never crashes the app
            pass

    def _periodic_cache_update(self):
        """Periodic placeholder for cache updates (no-op for now)."""
        try:
            while getattr(self, 'running', False):
                # Could refresh derived stats or TTL-based invalidation
                self.state_cache['last_updated']['marketplace'] = time.time()
                time.sleep(5)
        except Exception:
            pass
        
    def _load_config(self, config_file):
        """Load blockchain configuration"""
        # Try multiple paths, starting with relative paths to root
        possible_config_paths = [
            config_file,  # As provided
            "../../" + config_file,  # Up two levels (to root)
            "../../../" + config_file,  # Just in case
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../", config_file)  # Absolute path
        ]
        
        for path in possible_config_paths:
            if os.path.exists(path):
                self.logger.info(f"Found config file at {path}")
                with open(path, 'r') as f:
                    return json.load(f)
        
        # Default configuration if no file found
        self.logger.warning(f"Config file {config_file} not found in any location, using defaults")
        default_config = {
            "rpc_url": "http://127.0.0.1:8545",  # Default Hardhat URL
            "chain_id": 31337,                   # Default Hardhat chain ID
                                                 #it ensures transactions don't accidentally get sent to mainnet Ethereum (chain ID 1) or other networks
            "deployment_info": "../../deployment-info.json",  # Point to root
            "max_batch_size": 10,
            "tx_confirmation_blocks": 1,
            "gas_price_strategy": "medium",
            "retry_count": 3,
            "retry_delay": 2  # seconds
        }
        return default_config
    
    def _connect_to_blockchain(self, using_hardhat=True):
        """Connect to the blockchain network with optimized middleware"""
        w3 = Web3(Web3.HTTPProvider(self.config["rpc_url"]))
        
        # Add POA middleware for networks like Polygon
        if not using_hardhat:
            w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
        if not w3.is_connected():
            raise ConnectionError(f"Failed to connect to blockchain at {self.config['rpc_url']}")
            
        self.logger.info(f"Connected to blockchain: {w3.is_connected()}")
        return w3
    
    def _load_contracts(self):
        """Load contract ABIs and addresses with error handling"""
        contracts = {}
        
        try:
            # Go up two directories to find deployment info
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            deployment_info_path = os.path.join(base_dir, "deployment-info.json")
            
            if not os.path.exists(deployment_info_path): 
                self.logger.error(f"Deployment info file {deployment_info_path} not found")
                return {}
                
            with open(deployment_info_path, 'r') as f:
                deployment_info = json.load(f)
                
            # Set correct path to artifacts directory
            abi_dir = os.path.join(base_dir, "artifacts", "contracts")
            
            # Define contracts to load
            contract_keys = [
                "registry", "request", "auction", 
                "nft", "market", "facade", "mockToken"
            ]
            
            # Map contract keys to ABI file patterns
            abi_patterns = {
                "registry": "MaaSRegistry.sol/MaaSRegistry.json",
                "request": "MaaSRequest.sol/MaaSRequest.json",
                "auction": "MaaSAuction.sol/MaaSAuction.json",
                "nft": "MaaSNFT.sol/MaaSNFT.json",
                "market": "MaaSMarket.sol/MaaSMarket.json",
                "facade": "MaaSFacade.sol/MaaSFacade.json",
                "mockToken": "MockERC20.sol/MockERC20.json"
            }
            
            # Load each contract
            for key in contract_keys:
                if key in deployment_info:
                    address = deployment_info[key]
                    abi_file = os.path.join(abi_dir, abi_patterns.get(key, ""))
                    
                    self.logger.info(f"Looking for ABI file at: {abi_file}")
                    
                    try:
                        if os.path.exists(abi_file):
                            with open(abi_file, 'r') as abi_f:
                                abi_data = json.load(abi_f)
                                abi = abi_data["abi"]
                                contracts[key] = self.w3.eth.contract(address=address, abi=abi)
                        else:
                            self.logger.warning(f"ABI file {abi_file} not found for {key}")
                    except Exception as e:
                        self.logger.error(f"Error loading ABI for {key}: {e}")
                else:
                    self.logger.warning(f"No address found for {key} in deployment info")
            
            return contracts
            
        except Exception as e:
            self.logger.error(f"Error loading contracts: {e}")
            return {}
    
    # ================ Enhanced Account Management ================
    
    def create_account(self, agent_id, agent_type):
        """
        Create or assign an Ethereum account for an agent with improved error handling
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent ('commuter' or 'provider')
            
        Returns:
            str: Account address or None if failed
        """
        try:
            # In production, you might load existing keys from secure storage
            # For simulation, create a NEW account
            acct = self.w3.eth.account.create() 
            
            # Store account details
            self.accounts[agent_id] = {
                "address": acct.address,
                "private_key": acct.key.hex(),
                "type": agent_type,
                "created_at": time.time(),
                "tx_count": 0
            }
            
            # Fund account with ETH for gas (in development)
            self._fund_account(acct.address)
            
            # Initialize nonce tracking
            self.tx_nonce_map[acct.address] = self.w3.eth.get_transaction_count(acct.address)
            
            self.logger.info(f"Created account {acct.address} for {agent_type} {agent_id}")
            return acct.address
            
        except Exception as e:
            self.logger.error(f"Error creating account: {e}")
            return None
    
    def _fund_account(self, address, amount=1.0):
        """Fund an account with ETH and tokens for simulation"""
        try:
            # Get the first account from the node (usually has funds)
            admin = self.w3.eth.accounts[0]
            
            # Send ETH
            tx_hash = self.w3.eth.send_transaction({
                'from': admin,
                'to': address,
                'value': Web3.to_wei(amount, 'ether')
            })
            
            # Wait for transaction
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # If using mock token, send some tokens too
            if "mockToken" in self.contracts:
                token = self.contracts["mockToken"]
                tx = token.functions.transfer(
                    address,
                    Web3.to_wei(amount, 'ether')
                ).transact({'from': admin})
                self.w3.eth.wait_for_transaction_receipt(tx)
                
            self.logger.info(f"Funded account {address} with {amount} ETH and 1000 tokens")
            return True
            
        except Exception as e:
            self.logger.error(f"Error funding account: {e}")
            return False
    
    # ================ MARKETPLACE API FUNCTIONS ================
    
    def create_travel_request_marketplace(self, commuter, request):
        """
        MARKETPLACE API: Create travel request
        1. Store full request in marketplace DB
        2. Push only hash to blockchain
        """
        try:
            request_id = request.get('request_id', int(uuid.uuid4().int & (2**64 - 1)))
            
            # Store FULL request in marketplace database
            full_request = {
                'request_id': request_id,
                'commuter_id': commuter.unique_id,
                'origin': request['origin'],
                'destination': request['destination'],
                'start_time': request.get('start_time'),
                'travel_purpose': request.get('travel_purpose', 'work'),
                'flexible_time': request.get('flexible_time', 'medium'),
                'requirement_keys': request.get('requirement_keys', []),
                'requirement_values': request.get('requirement_values', []),
                'max_price': request.get('max_price', 100),
                'created_at': time.time(),
                'status': 'active'
            }
            
            self.marketplace_db['requests'][request_id] = full_request
            
            # Generate content hash
            content_hash = self._generate_content_hash(full_request)
            
            # Push minimal data to blockchain (just hash and IDs)
            blockchain_data = {
                'request_id': request_id,
                'commuter_id': commuter.unique_id,
                'content_hash': content_hash,
                'timestamp': int(time.time())
            }
            
            # Queue minimal blockchain transaction
            if self.async_mode:
                self.queue_transaction(
                    TransactionData(
                        tx_type="request",
                        function_name="createRequestWithHash",  # Modified function
                        params=blockchain_data,
                        sender_id=commuter.unique_id
                    )
                )
            
            # Filter and notify eligible providers
            self._notify_eligible_providers(request_id, full_request)
            
            self.logger.info(f"Request {request_id} created in marketplace, hash {content_hash[:8]}... pushed to blockchain")
            return True, request_id
            
        except Exception as e:
            self.logger.error(f"Error creating marketplace request: {e}")
            return False, None
    
    def submit_offer_marketplace(self, provider, request_id, price, details=None):
        """
        MARKETPLACE API: Submit offer from provider
        1. Store full offer in marketplace DB
        2. Push only essential data to blockchain
        """
        try:
            # Generate offer ID
            offer_id = int(request_id * 1000 + provider.unique_id)
            
            # Store FULL offer in marketplace database
            full_offer = {
                'offer_id': offer_id,
                'request_id': request_id,
                'provider_id': provider.unique_id,
                'price': price,
                'mode': self._get_provider_mode(provider),
                'route_details': details.get('route', []) if details else [],
                'estimated_time': details.get('time', 30) if details else 30,
                'capacity': getattr(provider, 'available_capacity', 1),
                'quality_score': getattr(provider, 'quality_score', 70),
                'reliability': getattr(provider, 'reliability', 70),
                'created_at': time.time(),
                'status': 'submitted'
            }
            
            self.marketplace_db['offers'][offer_id] = full_offer
            
            # Generate offer hash
            offer_hash = self._generate_content_hash(full_offer)
            
            # Push minimal data to blockchain
            blockchain_data = {
                'offer_id': offer_id,
                'request_id': request_id,
                'provider_id': provider.unique_id,
                'price_hash': hashlib.sha256(str(price).encode()).hexdigest(),
                'offer_hash': offer_hash,
                'timestamp': int(time.time())
            }
            
            # Queue minimal blockchain transaction
            if self.async_mode:
                self.queue_transaction(
                    TransactionData(
                        tx_type="offer",
                        function_name="submitOfferHash",  # Modified function
                        params=blockchain_data,
                        sender_id=provider.unique_id
                    )
                )
            
            self.logger.info(f"Offer {offer_id} submitted to marketplace for request {request_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error submitting marketplace offer: {e}")
            return False
    
    def run_marketplace_matching(self, request_id):
        """
        MARKETPLACE API: Run matching algorithm off-chain
        Complex matching logic happens here, not on blockchain
        """
        try:
            # Get request from marketplace DB
            request = self.marketplace_db['requests'].get(request_id)
            if not request:
                return False, None
            
            # Get all offers for this request
            relevant_offers = [
                offer for offer in self.marketplace_db['offers'].values()
                if offer['request_id'] == request_id and offer['status'] == 'submitted'
            ]
            
            if not relevant_offers:
                return False, None
            
            # Run complex matching algorithm (off-chain)
            best_offer = self._run_matching_algorithm(request, relevant_offers)
            
            if not best_offer:
                return False, None
            
            # Store match result
            match_result = {
                'request_id': request_id,
                'winning_offer_id': best_offer['offer_id'],
                'provider_id': best_offer['provider_id'],
                'final_price': best_offer['price'],
                'match_time': time.time(),
                'match_score': best_offer.get('match_score', 0)
            }
            
            self.marketplace_db['matches'][request_id] = match_result
            
            # Push match result to blockchain for settlement
            blockchain_result = {
                'request_id': request_id,
                'winning_offer_id': best_offer['offer_id'],
                'provider_id': best_offer['provider_id'],
                'price': best_offer['price']
            }
            
            if self.async_mode:
                self.queue_transaction(
                    TransactionData(
                        tx_type="match",
                        function_name="recordMatchResult",
                        params=blockchain_result,
                        sender_id=0  # System transaction
                    )
                )
            
            self.logger.info(f"Matching completed for request {request_id}, winner: offer {best_offer['offer_id']}")
            return True, match_result
            
        except Exception as e:
            self.logger.error(f"Error in marketplace matching: {e}")
            return False, None
    
    def _notify_eligible_providers(self, request_id, request_data):
        """Filter and notify only eligible providers"""
        origin = request_data['origin']
        max_distance = 10  # Configurable
        
        for provider_id, provider_data in self.marketplace_db['providers'].items():
            # Check distance
            if 'location' in provider_data:
                distance = self._calculate_distance(origin, provider_data['location'])
                if distance <= max_distance:
                    # Add to notification queue
                    notification = {
                        'request_id': request_id,
                        'notified_at': time.time(),
                        'distance': distance
                    }
                    self.marketplace_db['notifications'][provider_id].append(notification)
                    self.logger.debug(f"Provider {provider_id} notified about request {request_id}")
    
    def _run_matching_algorithm(self, request, offers):
        """
        Complex matching algorithm (runs off-chain in marketplace)
        This replaces the on-chain auction logic
        """
        scored_offers = []
        
        for offer in offers:
            score = 0
            
            # Price score (40% weight)
            max_price = request.get('max_price', 100)
            if offer['price'] <= max_price:
                price_score = (max_price - offer['price']) / max_price * 40
                score += price_score
            
            # Quality score (30% weight)
            quality_score = offer.get('quality_score', 50) / 100 * 30
            score += quality_score
            
            # Reliability score (20% weight)
            reliability_score = offer.get('reliability', 50) / 100 * 20
            score += reliability_score
            
            # Time score (10% weight)
            time_score = 10 if offer.get('estimated_time', 60) < 45 else 5
            score += time_score
            
            offer['match_score'] = score
            scored_offers.append(offer)
        
        # Return best offer
        if scored_offers:
            return max(scored_offers, key=lambda x: x['match_score'])
        return None
    
    def _generate_content_hash(self, data):
        """Generate hash of content for blockchain storage"""
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _get_provider_mode(self, provider):
        """Get provider mode"""
        if hasattr(provider, 'mode_type'):
            return provider.mode_type
        return 'car'
    
    def _calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    # ================ BACKWARDS COMPATIBILITY ================
    # Keep these method names but redirect to marketplace functions
    
    def create_travel_request(self, commuter, request):
        """Redirect to marketplace function"""
        return self.create_travel_request_marketplace(commuter, request)
    
    def submit_offer(self, provider, request_id, price, details=None):
        """Redirect to marketplace function"""
        return self.submit_offer_marketplace(provider, request_id, price, details)
    
    def finalize_auction(self, request_id):
        """Now runs marketplace matching instead of blockchain auction"""
        return self.run_marketplace_matching(request_id)
    
    # ================ REGISTRATION FUNCTIONS (Keep as is) ================

    def is_commuter_registered(self, commuter_id):
        """Return True if a commuter account exists."""
        return commuter_id in self.accounts

    def is_provider_registered(self, provider_id):
        """Return True if a provider account exists."""
        return provider_id in self.accounts

    def register_commuter(self, commuter_agent):
        """Register commuter - stores profile in marketplace DB"""
        try:
            if commuter_agent.unique_id in self.accounts:
                return True, self.accounts[commuter_agent.unique_id]["address"]
            
            # Create account
            account = Account.create()
            address = account.address
            private_key = account.key.hex()
            
            self.accounts[commuter_agent.unique_id] = {
                "address": address,
                "private_key": private_key
            }
            
            # Store profile in marketplace DB
            self.marketplace_db['commuters'][commuter_agent.unique_id] = {
                'id': commuter_agent.unique_id,
                'age': getattr(commuter_agent, 'age', 30),
                'income_level': getattr(commuter_agent, 'income_level', 'middle'),
                'preferences': getattr(commuter_agent, 'preferences', {}),
                'registered_at': time.time()
            }
            
            self.logger.info(f"Commuter {commuter_agent.unique_id} registered in marketplace")
            return True, address
            
        except Exception as e:
            self.logger.error(f"Error registering commuter: {e}")
            return False, None
    
    def register_provider(self, provider_agent):
        """Register provider - stores profile in marketplace DB"""
        try:
            if provider_agent.unique_id in self.accounts:
                return True, self.accounts[provider_agent.unique_id]["address"]
            
            # Create account
            account = Account.create()
            address = account.address
            private_key = account.key.hex()
            
            self.accounts[provider_agent.unique_id] = {
                "address": address,
                "private_key": private_key
            }
            
            # Store profile in marketplace DB
            self.marketplace_db['providers'][provider_agent.unique_id] = {
                'id': provider_agent.unique_id,
                'company_name': getattr(provider_agent, 'company_name', f"Provider-{provider_agent.unique_id}"),
                'mode_type': getattr(provider_agent, 'mode_type', 'car'),
                'capacity': getattr(provider_agent, 'capacity', 4),
                'base_price': getattr(provider_agent, 'base_price', 10),
                'location': getattr(provider_agent, 'service_center', [0, 0]),
                'quality_score': getattr(provider_agent, 'quality_score', 70),
                'reliability': getattr(provider_agent, 'reliability', 70),
                'registered_at': time.time()
            }
            
            self.logger.info(f"Provider {provider_agent.unique_id} registered in marketplace")
            return True, address
            
        except Exception as e:
            self.logger.error(f"Error registering provider: {e}")
            return False, None
    
    # ================ TRANSACTION PROCESSING (Simplified) ================
    
    def queue_transaction(self, transaction_data):
        """Queue transaction for batch processing"""
        self.tx_queue.append(transaction_data)
        
        if len(self.tx_queue) >= self.batch_size_limit and self.async_mode:
            self.thread_pool.submit(self._process_transaction_batch)
    
    def _process_transaction_batch(self):
        """Process queued transactions"""
        if not self.tx_queue:
            return
        
        batch_size = min(self.batch_size_limit, len(self.tx_queue))
        batch = []
        for _ in range(batch_size):
            if self.tx_queue:
                batch.append(self.tx_queue.popleft())
        
        self.logger.info(f"Processing {len(batch)} transactions (would go to blockchain)")
        # In production, these would be sent to blockchain
        # For now, just log them
        for tx in batch:
            self.tx_count += 1
            self.logger.debug(f"TX: {tx.tx_type} - {tx.function_name}")
    
    # ================ QUERY FUNCTIONS ================
    
    def get_marketplace_requests(self, status='active'):
        """Get requests from marketplace DB"""
        return [r for r in self.marketplace_db['requests'].values() if r['status'] == status]
    
    def get_provider_notifications(self, provider_id):
        """Get notifications for a provider"""
        return self.marketplace_db['notifications'].get(provider_id, [])
    
    def get_request_offers(self, request_id):
        """Get all offers for a request"""
        return [o for o in self.marketplace_db['offers'].values() if o['request_id'] == request_id]
    
    # ================ REMOVE/DEPRECATE THESE FUNCTIONS ================
    # These are unnecessary for the simplified marketplace approach
    
    def create_nft(self, service_details, provider_id, commuter_id):
        """UNNECESSARY - Keep for backwards compatibility but log warning"""
        self.logger.warning("NFT creation called but not needed in marketplace flow")
        return False, None
    
    def create_bundle(self, bundle_segments):
        """UNNECESSARY - Too complex for MVP"""
        self.logger.warning("Bundle creation called but not needed in simplified flow")
        return False, None
