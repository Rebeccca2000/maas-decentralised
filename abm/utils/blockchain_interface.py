"""
Enhanced BlockchainInterface for Decentralized MaaS
Provides an optimized bridge between ABM simulation and blockchain contracts
with support for NFT marketplace operations, caching, and asynchronous transactions.
"""

import json
import asyncio
import logging
import time
import uuid
import numpy as np
from web3 import Web3
from web3.middleware import geth_poa_middleware
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import deque
import random
import math

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
        self.logger = logging.getLogger("BlockchainInterface")
        
        # Load configuration
        self.config = self._load_config(config_file)
        self.w3 = self._connect_to_blockchain(using_hardhat)
        self.contracts = self._load_contracts()
        self.accounts = {}  # Will store account addresses by agent ID
        self.gas_limit = 1000000  # Default gas limit
        
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
    
    # ================ Core ABM-Blockchain Operations ================
    
    def register_commuter(self, commuter_agent):
        """
        Register a commuter on the blockchain with minimal data
        """
        try:
            # Create account if not exists
            if 'registry' not in self.contracts:
                self.logger.error("Registry contract not loaded. Check ABI paths.")
                return False, None
                
            if commuter_agent.unique_id not in self.accounts:
                address = self.create_account(commuter_agent.unique_id, "commuter")
                if not address:
                    return False, None
            else:
                address = self.accounts[commuter_agent.unique_id]["address"]
                
            private_key = self.accounts[commuter_agent.unique_id]["private_key"]
            
            # Process synchronously with minimal data
            registry = self.contracts["registry"]
            nonce = self._get_next_nonce(address)
            
            # Build transaction with only commuter ID
            transaction = registry.functions.addCommuter(
                commuter_agent.unique_id
            ).build_transaction({
                'chainId': self.config["chain_id"],
                'gas': self.gas_limit,
                'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                'nonce': nonce,
            })
            
            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(transaction, private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Update cache
            self.state_cache['commuters'][commuter_agent.unique_id] = {
                'address': address,
                'registered': True,
                'last_updated': time.time()
            }
            
            self.logger.info(f"Commuter {commuter_agent.unique_id} registered on blockchain. Status: {receipt['status']}")
            return receipt['status'] == 1, address
            
        except Exception as e:
            self.logger.error(f"Error registering commuter: {e}")
            return False, None


    def register_provider(self, provider_agent):
        try:
            # Create account if not exists
            if provider_agent.unique_id not in self.accounts:
                address = self.create_account(provider_agent.unique_id, "provider")
                if not address:
                    return False, None
            else:
                address = self.accounts[provider_agent.unique_id]["address"]
                
            # Map mode type to enum (matching smart contract)
            mode_map = {"car": 3, "bike": 4, "bus": 2, "train": 1, "walk": 5}
            if hasattr(provider_agent, 'mode_type'):
                mode_type = mode_map.get(provider_agent.mode_type, 0)
            elif "Bike" in provider_agent.company_name:
                mode_type = 4
            elif "Uber" in provider_agent.company_name or "Car" in provider_agent.company_name:
                mode_type = 3
            else:
                mode_type = 0
            
            # Process synchronously
            registry = self.contracts["registry"]
            nonce = self._get_next_nonce(address)
            
            self.logger.info(f"Registering provider {provider_agent.unique_id} with minimal on-chain data")
            
            # Only store essential consensus-critical data on-chain
            # This matches the smart contract's Provider struct exactly
            transaction = registry.functions.addProvider(
                provider_agent.unique_id,                                    # providerId
                provider_agent.company_name,                                 # companyName  
                mode_type,                                                   # modeType (enum)
                Web3.to_wei(str(provider_agent.base_price), 'ether'),       # basePrice (in wei)
                provider_agent.capacity                                      # total capacity
            ).build_transaction({
                'chainId': self.config["chain_id"],
                'gas': self.gas_limit * 2,  # Reduced gas limit since we're storing less data
                'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                'nonce': nonce,
            })
            
            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(transaction, self.accounts[provider_agent.unique_id]["private_key"])
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Store both on-chain and off-chain data in cache for local ABM operations
            # On-chain data (what's stored in blockchain)
            on_chain_data = {
                'providerId': provider_agent.unique_id,
                'providerAddress': address,
                'companyName': provider_agent.company_name,
                'modeType': mode_type,
                'basePrice': Web3.to_wei(str(provider_agent.base_price), 'ether'),
                'capacity': provider_agent.capacity,
                'isActive': True
            }
            
            # Off-chain data (managed locally for ABM performance)
            off_chain_data = {
                'availableCapacity': provider_agent.capacity,                # Changes frequently
                'servicingArea': getattr(provider_agent, "service_area", 20),# Geographic data
                'serviceCenter': getattr(provider_agent, "service_center", [50, 50]), # Geographic data
                'responseTime': getattr(provider_agent, "response_time", 10), # Performance metric
                'reliability': getattr(provider_agent, "reliability", 70),    # Reputation score
                'qualityScore': getattr(provider_agent, "quality_score", 70), # Reputation score
                'serviceCount': 0,                                           # Analytics data
                'totalRevenue': 0,                                          # Analytics data
                'registrationTime': int(time.time()),                       # Can use block timestamp instead
                'isVerified': False,                                        # Can be managed off-chain initially
            }
            
            # Update cache with both on-chain and off-chain data for ABM operations
            self.state_cache['providers'][provider_agent.unique_id] = {
                'address': address,
                'on_chain_data': on_chain_data,      # What's actually on blockchain
                'off_chain_data': off_chain_data,    # What we manage locally
                'registered': True,
                'last_updated': time.time()
            }
            
            self.logger.info(f"Provider {provider_agent.unique_id} registered with minimal on-chain footprint")
            return True, address
            
        except Exception as e:
            self.logger.error(f"Error registering provider: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False, None
    
    def create_travel_request(self, commuter, request):
        """
        Create a travel request on the blockchain.
        
        Args:
            commuter: The commuter agent
            request: Request dictionary with details
            
        Returns:
            Success status (bool)
        """
        # Extract request details
        request_id = request.get('request_id', 0)
        
        # Convert tuples to lists if needed
        origin = request.get('origin', [0, 0])
        if isinstance(origin, tuple):
            origin = list(origin)
            
        destination = request.get('destination', [0, 0])
        if isinstance(destination, tuple):
            destination = list(destination)
            
        start_time = request.get('start_time', self.current_abm_step)  # Already in steps
        travel_purpose = request.get('travel_purpose', 0)
        flexible_time = request.get('flexible_time', 'medium')
        
        # Get requirements if available
        requirement_keys = request.get('requirement_keys', [])
        requirement_values = request.get('requirement_values', [])
        
        # Ensure requirements are properly formatted
        if not requirement_keys:
            requirement_keys = ["wheelchair", "assistance", "child_seat", "pet_friendly"]
        if not requirement_values:
            requirement_values = [False, False, False, False]
        
        # Process request using batch processing
        if self.async_mode:
            self.logger.info(f"Queuing travel request {request_id} for commuter {commuter.unique_id}")
            
            # Calculate route with midpoint for better visualization
            midpoint = [
                (origin[0] + destination[0]) / 2,
                (origin[1] + destination[1]) / 2
            ]
            
            # Prepare route info if needed
            path_info = json.dumps({
                'origin': origin,
                'destination': destination,
                'distance': self._calculate_distance(origin, destination),
                'route': [origin, midpoint, destination]  # Include a simple route with midpoint
            })
            
            # Queue transaction data with properly mapped parameters
            tx_data = {
                'requestId': request_id,
                'commuter_id': commuter.unique_id,
                'origin': origin,
                'destination': destination,
                'startTime': start_time,
                'purpose': travel_purpose,  # Make sure this is an integer
                'flexibleTime': flexible_time,
                'requirementKeys': requirement_keys,
                'requirementValues': requirement_values,
                'pathInfo': path_info
            }
            
            # Add to current batch
            self.current_batch['requests'].append(tx_data)
            
            # Process batch if limit reached
            if len(self.current_batch['requests']) >= self.batch_size_limit:
                self.process_requests_batch(self.current_batch['requests'])
                self.current_batch['requests'] = []
                
            # Queue for batch processing
            self.queue_transaction(
                TransactionData(
                    tx_type="request",
                    function_name="createTravelRequest",
                    params=tx_data,
                    sender_id=commuter.unique_id
                )
            )
            # Update cache IMMEDIATELY to mark request as active, don't wait for blockchain confirmation
            if 'requests' not in self.state_cache:
                self.state_cache['requests'] = {}
                
            self.state_cache['requests'][request_id] = {
                'commuter_id': commuter.unique_id,
                'data': {
                    'origin': origin,
                    'destination': destination,
                    'startTime': start_time,
                    'requirementKeys': requirement_keys,
                    'requirementValues': requirement_values,
                },
                'status': 'active',  # Mark as active immediately
                'blockchain_status': 'pending',
                'last_updated': time.time()
            }

            return True
        else:
            # Synchronous processing (fallback)
            self.logger.warning("Synchronous request processing not implemented")
            return False    
    
    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def submit_offer(self, provider, request_id, price, details=None):
        """
        Submit an offer for a service request on behalf of a provider.
        Uses minimal on-chain data and manages detailed analytics off-chain.
        
        Args:
            provider: Provider agent
            request_id: Request ID
            price: Offered price
            details: Additional offer details
            
        Returns:
            Success status
        """
        try:
            # Verify provider exists and has account
            if not hasattr(provider, 'unique_id') or provider.unique_id not in self.accounts:
                self.logger.error(f"Provider {getattr(provider, 'unique_id', 'unknown')} not registered")
                return False
            
            # Generate offer ID
            offer_id = int(request_id * 1000 + provider.unique_id)
            
            # Create minimal on-chain offer structure (consensus-critical data only)
            on_chain_offer = {
                'id': offer_id,
                'requestId': request_id,
                'providerId': provider.unique_id,
                'price': price,
                'mode': self._get_provider_mode_enum(provider),
                'startTime': details.get('start_time', self.current_abm_step + 3),
                'totalTime': details.get('time', 3),
                'offerTime': self.current_abm_step
            }
            
            # Create detailed off-chain data for ABM operations
            off_chain_details = {
                'totalPrice': price,
                'routeDetails': details.get('route', []),
                'capacity': getattr(provider, 'available_capacity', 1),
                'quality': getattr(provider, 'quality_score', 70),
                'reliability': getattr(provider, 'reliability', 70),
                'isVerified': getattr(provider, 'verified', False),
                'hasInsurance': False,
                'additionalServices': [],
                'isConditional': False,
                'conditions': "",
                'estimatedArrival': details.get('estimated_time', 0)
            }
            
            # Update provider's available capacity off-chain
            self.update_provider_off_chain_data(provider.unique_id, 'availableCapacity', 
                                            max(0, getattr(provider, 'available_capacity', 1) - 1))
            
            # Store offer in local cache for ABM operations
            if 'offers' not in self.state_cache:
                self.state_cache['offers'] = {}
            
            self.state_cache['offers'][offer_id] = {
                'on_chain_data': on_chain_offer,
                'off_chain_data': off_chain_details,
                'status': 'submitted',
                'last_updated': time.time()
            }
            
            # Submit minimal offer to blockchain (async)
            if self.async_mode:
                self.queue_transaction(
                    TransactionData(
                        tx_type="offer",
                        function_name="submitOffer",
                        params=on_chain_offer,
                        sender_id=provider.unique_id
                    )
                )
                
                self.logger.info(f"Provider {provider.unique_id} queued offer {offer_id} for request {request_id} at price {price}")
            else:
                # Synchronous submission (for immediate confirmation)
                success = self._submit_offer_sync(provider, on_chain_offer)
                if not success:
                    return False
                    
                self.logger.info(f"Provider {provider.unique_id} submitted offer {offer_id} for request {request_id} at price {price}")
            
            # Record service activity for loyalty points (off-chain tracking)
            self.record_service_activity(provider.unique_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error submitting offer: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _get_provider_mode_enum(self, provider):
        """Get provider mode as enum value matching smart contract"""
        if hasattr(provider, 'mode_type'):
            mode_map = {"car": 3, "bike": 4, "bus": 2, "train": 1, "walk": 5}
            return mode_map.get(provider.mode_type, 0)
        elif "Bike" in provider.company_name:
            return 4
        elif "Uber" in provider.company_name or "Car" in provider.company_name:
            return 3
        else:
            return 0

    def _submit_offer_sync(self, provider, offer_data):
        """Submit offer synchronously to blockchain"""
        try:
            address = self.accounts[provider.unique_id]["address"]
            private_key = self.accounts[provider.unique_id]["private_key"]
            
            auction = self.contracts["auction"]
            nonce = self._get_next_nonce(address)
            
            transaction = auction.functions.submitOffer(
                offer_data['requestId'],
                offer_data['price'],
                offer_data['mode'],
                offer_data['startTime'],
                offer_data['totalTime']
            ).build_transaction({
                'chainId': self.config["chain_id"],
                'gas': self.gas_limit,
                'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                'nonce': nonce,
            })
            
            signed_tx = self.w3.eth.account.sign_transaction(transaction, private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt['status'] == 1:
                # Update transaction counts only on actual blockchain confirmation
                if hasattr(self.model, 'transaction_count'):
                    self.model.transaction_count += 1
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error in synchronous offer submission: {e}")
            return False

    # ================ Enhanced with batch processing capability ================
    
    def process_requests_batch(self, request_batch):
        """Process multiple requests in a single batch with proper nonce management"""
        results = []
        
        # Group requests by commuter_id to handle nonces properly
        requests_by_commuter = {}
        for request in request_batch:
            commuter_id = request.get('commuter_id')
            if commuter_id not in requests_by_commuter:
                requests_by_commuter[commuter_id] = []
            requests_by_commuter[commuter_id].append(request)
        
        # Process each commuter's requests sequentially with correct nonces
        for commuter_id, requests in requests_by_commuter.items():
            # Find commuter agent
            commuter_agent = None
            for cache_id, cache_data in self.state_cache['commuters'].items():
                if isinstance(cache_data, dict) and 'data' in cache_data:
                    if isinstance(cache_data['data'], dict) and cache_data['data'].get('commuterId') == commuter_id:
                        commuter_agent = cache_data
                        break
                    elif hasattr(cache_data['data'], 'params') and cache_data['data'].params.get('commuterId') == commuter_id:
                        commuter_agent = cache_data
                        break
            
            if not commuter_agent:
                for request in requests:
                    results.append((False, f"Commuter {commuter_id} not found"))
                continue
            
            # Get address for this commuter
            if commuter_id not in self.accounts:
                for request in requests:
                    results.append((False, f"Account for commuter {commuter_id} not found"))
                continue
                
            address = self.accounts[commuter_id]["address"]
            
            # Get current nonce from network (not cached)
            try:
                current_nonce = self.w3.eth.get_transaction_count(address, 'pending')
                self.tx_nonce_map[address] = current_nonce  # Update nonce map
            except Exception as e:
                self.logger.error(f"Error getting nonce for {address}: {e}")
                for request in requests:
                    results.append((False, f"Failed to get nonce: {str(e)}"))
                continue
            
            # Process each request for this commuter with sequential nonces
            for request in requests:
                request_id = request.get('request_id', int(uuid.uuid4().int & (2**64 - 1)))
                
                # Prepare transaction data
                tx_data = {
                    "requestId": request_id,
                    "origin": request['origin'],
                    "destination": request['destination'],
                    "startTime": request.get('startTime', request.get('start_time')),
                    "purpose": request.get('purpose', request.get('travel_purpose', 0)),
                    "flexibleTime": request.get('flexibleTime', request.get('flexible_time', 'medium')),
                    "requirementValues": request.get('requirement_values', [False, False, False, False]),
                    "requirementKeys": request.get('requirement_keys', ["wheelchair", "assistance", "child_seat", "pet_friendly"]),
                    "pathInfo": self._calculate_route(request['origin'], request['destination'])
                }
                
                # Create transaction data with explicit nonce
                tx_obj = TransactionData(
                    tx_type="request",
                    function_name="createTravelRequest",
                    params=tx_data,
                    sender_id=commuter_id
                )
                
                # Add explicit nonce for this transaction
                tx_obj.explicit_nonce = current_nonce
                current_nonce += 1  # Increment for next transaction
                
                # Queue transaction
                self.queue_transaction(tx_obj)
                
                # Update local cache
                self.state_cache['requests'][request_id] = {
                    'commuter_id': commuter_id,
                    'data': tx_data,
                    'status': 'pending',
                    'last_updated': time.time()
                }
                
                results.append((True, request_id))
        
        # Trigger processing if queue is large enough
        if len(self.tx_queue) >= self.batch_size_limit:
            self.thread_pool.submit(self._process_transaction_batch)
        
        return results
        
    # ================ NFT marketplace operations ================
        
    def create_nft(self, service_details, provider_id, commuter_id):
        """
        Create NFT representing mobility service
        """
        try:
            # Get owner address
            if commuter_id not in self.accounts:
                self.logger.error(f"Commuter {commuter_id} not registered")
                return False, None
                    
            address = self.accounts[commuter_id]["address"]
            private_key = self.accounts[commuter_id]["private_key"]
            
            # Prepare NFT metadata
            request_id = service_details.get('request_id', int(uuid.uuid4().int & (2**64 - 1)))
            route_details = json.dumps(service_details.get('route_details', {}))
            price = Web3.to_wei(str(service_details.get('price', 0)), 'ether')
            start_time = service_details.get('start_time')
            duration = service_details.get('duration', 3)  # 30 minutes default
            token_uri = f"ipfs://QmServiceNFT{request_id}"  # Placeholder
            
            # Process synchronously (skipping async mode part for now)
            # First approve token spending if needed
            token = self.contracts["mockToken"]
            nonce = self._get_next_nonce(address)
            
            approval_tx = token.functions.approve(
                self.contracts["nft"].address,
                price
            ).build_transaction({
                'chainId': self.config["chain_id"],
                'gas': 100000,
                'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                'nonce': nonce,
            })
            
            # Sign and send approval
            signed_approval = self.w3.eth.account.sign_transaction(approval_tx, private_key)
            approval_hash = self.w3.eth.send_raw_transaction(signed_approval.rawTransaction)
            
            # Wait for approval transaction
            self.w3.eth.wait_for_transaction_receipt(approval_hash)
            
            # Now mint the NFT
            nft = self.contracts["nft"]
            nonce = self._get_next_nonce(address)
            
            # Log the parameters being sent
            self.logger.info(f"NFT mint parameters: requestId={request_id}, providerId={provider_id}, " +
                            f"startTime={start_time}, duration={duration}, tokenURI={token_uri}")
            
            transaction = nft.functions.mintServiceNFT(
                request_id,
                provider_id,
                route_details,
                price,
                start_time,
                duration,
                token_uri
            ).build_transaction({
                'chainId': self.config["chain_id"],
                'gas': self.gas_limit,
                'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                'nonce': nonce,
            })
            
            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(transaction, private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Log entire receipt for debugging
            self.logger.info(f"NFT mint transaction receipt: {receipt}")
            
            # Extract token ID from event logs
            token_id = None
            if receipt['status'] == 1:
                # Method 1: Try direct event processing
                for log in receipt['logs']:
                    try:
                        if log['address'].lower() == self.contracts["nft"].address.lower():
                            # Try to decode event
                            evt = self.contracts["nft"].events.ServiceTokenized().process_log(log)
                            if 'args' in evt and 'tokenId' in evt['args']:
                                token_id = evt['args']['tokenId']
                                self.logger.info(f"Found token ID {token_id} in logs")
                                break
                    except Exception as e:
                        self.logger.debug(f"Error processing log: {e}")
                        continue
                
                # Method 2: If first method failed, try direct log parsing (advanced approach)
                if token_id is None:
                    self.logger.info("Trying alternative event parsing method")
                    # This is a more manual approach that might work if the event structure is complex
                    for log in receipt['logs']:
                        try:
                            # Check if this log is from the NFT contract
                            if log['address'].lower() == self.contracts["nft"].address.lower():
                                # For ServiceTokenized event, try to extract token ID from data
                                # This requires knowledge of the event structure
                                # Example: if the first topic is the ServiceTokenized event hash
                                topics = log.get('topics', [])
                                if len(topics) >= 1:
                                    # Check for a transfer event which often includes the token ID
                                    transfer_event_hash = self.w3.keccak(
                                        text="Transfer(address,address,uint256)").hex()
                                    if topics[0].hex() == transfer_event_hash and len(topics) >= 4:
                                        # The token ID is typically the 3rd topic
                                        token_id = int(topics[3].hex(), 16)
                                        self.logger.info(f"Found token ID {token_id} from Transfer event")
                                        break
                        except Exception as e:
                            self.logger.debug(f"Error in alternative parsing: {e}")
                            continue
            
                # Method 3: If both methods fail, use a fallback token ID for testing
                if token_id is None:
                    self.logger.warning("Could not extract token ID from logs, using fallback for testing")
                    # Get the next token ID by querying totalSupply if available
                    try:
                        if hasattr(self.contracts["nft"].functions, 'totalSupply'):
                            token_id = self.contracts["nft"].functions.totalSupply().call()
                        else:
                            # Last resort: generate a random ID for testing
                            token_id = random.randint(1000, 9999)
                    except:
                        token_id = random.randint(1000, 9999)
            
            if token_id is not None:
                # Update cache
                self.state_cache['nfts'][token_id] = {
                    'owner': commuter_id,
                    'provider': provider_id,
                    'data': service_details,
                    'status': 'minted',
                    'request_id': request_id,
                    'last_updated': time.time()
                }
                
                self.logger.info(f"NFT created with ID {token_id} for service {request_id}")
                return True, token_id
            else:
                self.logger.error(f"Failed to extract token ID from receipt")
                return False, None
                    
        except Exception as e:
            self.logger.error(f"Error creating NFT: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False, None
    
    def list_nft_for_sale(self, nft_id, price, time_parameters=None):
        """
        List NFT on marketplace with time-sensitive pricing
        
        Args:
            nft_id: NFT token ID
            price: Initial price
            time_parameters: Dict with time-sensitive pricing parameters
                - initial_price: Starting price
                - final_price: Ending price after decay
                - decay_start: When price starts to decay
                - decay_end: When price reaches final value
                
        Returns:
            bool: Success status
        """
        try:
            # Verify NFT exists in cache or on chain
            if nft_id not in self.state_cache['nfts']:
                # Try to fetch from blockchain
                nft_info = self._fetch_nft_info(nft_id)
                if not nft_info:
                    self.logger.error(f"NFT {nft_id} not found")
                    return False
                    
                owner_id = nft_info.get('owner_id')
            else:
                owner_id = self.state_cache['nfts'][nft_id]['owner']
            
            # Check if owner account exists
            if owner_id not in self.accounts:
                self.logger.error(f"Owner {owner_id} not registered")
                return False
                
            address = self.accounts[owner_id]["address"]
            private_key = self.accounts[owner_id]["private_key"]
            
            # Process time parameters
            use_dynamic_pricing = False
            if time_parameters:
                use_dynamic_pricing = True
                initial_price = Web3.to_wei(str(time_parameters.get('initial_price', price)), 'ether')
                final_price = Web3.to_wei(str(time_parameters.get('final_price', price * 0.5)), 'ether')
                decay_duration = time_parameters.get('decay_duration', 3)  # Default 0.5 hour
            else:
                initial_price = Web3.to_wei(str(price), 'ether')
                final_price = initial_price
                decay_duration = 0
            
            # Check if we should process asynchronously
            if self.async_mode:
                # Queue for asynchronous processing
                tx_data = {
                    "nftId": nft_id,
                    "price": initial_price,
                    "useDynamicPricing": use_dynamic_pricing,
                    "initialPrice": initial_price,
                    "finalPrice": final_price,
                    "decayDuration": decay_duration
                }
                
                self.queue_transaction(
                    TransactionData(
                        tx_type="marketplace",
                        function_name="listNFTForSale" if not use_dynamic_pricing else "listNFTWithDynamicPricing",
                        params=tx_data,
                        sender_id=owner_id
                    )
                )
                
                # Update local cache
                self.state_cache['marketplace'][nft_id] = {
                    'seller': owner_id,
                    'initial_price': initial_price,
                    'current_price': initial_price,
                    'final_price': final_price,
                    'dynamic_pricing': use_dynamic_pricing,
                    'decay_duration': decay_duration,
                    'listing_time': self.current_abm_step,
                    'status': 'pending',
                    'last_updated': time.time()
                }
                
                self.logger.info(f"NFT {nft_id} queued for listing at price {Web3.from_wei(initial_price, 'ether')} ETH")
                return True
            else:
                # Process synchronously
                # First approve NFT transfer to market
                nft = self.contracts["nft"]
                nonce = self._get_next_nonce(address)
                
                approval_tx = nft.functions.approve(
                    self.contracts["market"].address,
                    nft_id
                ).build_transaction({
                    'chainId': self.config["chain_id"],
                    'gas': 100000,
                    'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                    'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                    'nonce': nonce,
                })
                
                # Sign and send approval
                signed_approval = self.w3.eth.account.sign_transaction(approval_tx, private_key)
                approval_hash = self.w3.eth.send_raw_transaction(signed_approval.rawTransaction)
                
                # Wait for approval transaction
                self.w3.eth.wait_for_transaction_receipt(approval_hash)
                
                # Now list on market
                market = self.contracts["market"]
                nonce = self._get_next_nonce(address)
                
                if use_dynamic_pricing:
                    transaction = market.functions.listNFTWithDynamicPricing(
                        nft_id,
                        initial_price,
                        final_price,
                        decay_duration
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': self.gas_limit,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': nonce,
                    })
                else:
                    transaction = market.functions.listNFTForSale(
                        nft_id,
                        initial_price
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': self.gas_limit,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': nonce,
                    })
                
                # Sign and send transaction
                signed_tx = self.w3.eth.account.sign_transaction(transaction, private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Wait for transaction receipt
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                
                # Update cache
                if receipt['status'] == 1:
                    self.state_cache['marketplace'][nft_id] = {
                        'seller': owner_id,
                        'initial_price': initial_price,
                        'current_price': initial_price,
                        'final_price': final_price,
                        'dynamic_pricing': use_dynamic_pricing,
                        'decay_duration': decay_duration,
                        'listing_time': self.current_abm_step,
                        'status': 'listed',
                        'last_updated': time.time()
                    }
                    
                    self.logger.info(f"NFT {nft_id} listed for sale at price {Web3.from_wei(initial_price, 'ether')} ETH")
                    return True
                else:
                    self.logger.error(f"Failed to list NFT {nft_id}")
                    return False
                
        except Exception as e:
            self.logger.error(f"Error listing NFT: {e}")
            return False
    
    def search_nft_market(self, search_params):
        """
        Search for suitable NFTs in marketplace
        
        Args:
            search_params: Dict with search parameters
                - min_price: Minimum price
                - max_price: Maximum price
                - min_departure: Minimum departure time
                - max_departure: Maximum departure time
                - origin_area: [x, y, radius] for origin search
                - dest_area: [x, y, radius] for destination search
                - mode: Transport mode filter
                
        Returns:
            list: Matching NFTs
        """
        try:
            # Convert prices to wei
            min_price = Web3.to_wei(str(search_params.get('min_price', 0)), 'ether')
            max_price = Web3.to_wei(str(search_params.get('max_price', 0)), 'ether')
            
            # Convert times to timestamps
            current_time = int(time.time())
            min_departure = search_params.get('min_departure', current_time)
            max_departure = search_params.get('max_departure', current_time + 86400)  # Default 24h window
            
            # Try cache first for performance
            cache_key = f"search_{min_price}_{max_price}_{min_departure}_{max_departure}"
            cached_result = self._get_cached_result('marketplace_search', cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                return cached_result
                
            self.stats['cache_misses'] += 1
            
            # Call contract search function if available
            market = self.contracts["market"]
            contract_results = market.functions.searchListings(
                min_price,
                max_price,
                min_departure,
                max_departure
            ).call()
            
            results = []
            for listing in contract_results:
                token_id = listing[0]
                price = listing[1]
                seller = listing[2]
                
                # Get NFT details from blockchain or cache
                nft_info = self._get_cached_nft_info(token_id)
                
                # Process route details
                route_details = json.loads(nft_info.get('route_details', '{}'))
                
                # Check origin/destination constraints if specified
                if 'origin_area' in search_params and len(search_params['origin_area']) == 3:
                    origin_x, origin_y, radius = search_params['origin_area']
                    nft_origin = route_details.get('origin', [0, 0])
                    
                    # Calculate distance
                    distance = ((nft_origin[0] - origin_x) ** 2 + (nft_origin[1] - origin_y) ** 2) ** 0.5
                    if distance > radius:
                        continue
                        
                if 'dest_area' in search_params and len(search_params['dest_area']) == 3:
                    dest_x, dest_y, radius = search_params['dest_area']
                    nft_dest = route_details.get('destination', [0, 0])
                    
                    # Calculate distance
                    distance = ((nft_dest[0] - dest_x) ** 2 + (nft_dest[1] - dest_y) ** 2) ** 0.5
                    if distance > radius:
                        continue
                
                # Add to results if it passes all filters
                results.append({
                    'token_id': token_id,
                    'price': Web3.from_wei(price, 'ether'),
                    'seller': seller,
                    'start_time': nft_info.get('start_time', 0),
                    'duration': nft_info.get('duration', 0),
                    'route_details': route_details
                })
            
            # Cache results
            self._cache_search_results('marketplace_search', cache_key, results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching marketplace: {e}")
            return []
    
    def purchase_nft(self, nft_id, buyer_id):
        """
        Purchase NFT from marketplace
        
        Args:
            nft_id: NFT token ID
            buyer_id: ID of the buyer
            
        Returns:
            bool: Success status
        """
        try:
            # Verify buyer exists
            if buyer_id not in self.accounts:
                self.logger.error(f"Buyer {buyer_id} not registered")
                return False
                
            address = self.accounts[buyer_id]["address"]
            private_key = self.accounts[buyer_id]["private_key"]
            
            # Check if NFT is in marketplace
            if nft_id not in self.state_cache['marketplace']:
                # Try to fetch from blockchain
                market = self.contracts["market"]
                try:
                    listing = market.functions.getMarketItem(nft_id).call()
                    if not listing or listing[2]:  # Check if sold
                        self.logger.error(f"NFT {nft_id} not available for purchase")
                        return False
                        
                    price = listing[1]
                except Exception:
                    self.logger.error(f"NFT {nft_id} not found in marketplace")
                    return False
            else:
                listing_info = self.state_cache['marketplace'][nft_id]
                if listing_info.get('status') != 'listed':
                    self.logger.error(f"NFT {nft_id} not available for purchase")
                    return False
                    
                # Calculate current price if dynamic pricing
                if listing_info.get('dynamic_pricing', False):
                    current_time = time.time()
                    listing_time = listing_info.get('listing_time', current_time)
                    decay_duration = listing_info.get('decay_duration', 3600)
                    initial_price = listing_info.get('initial_price')
                    final_price = listing_info.get('final_price')
                    
                    # Linear price decay
                    if current_time >= listing_time + decay_duration:
                        price = final_price
                    else:
                        elapsed = current_time - listing_time
                        decay_factor = elapsed / decay_duration
                        price_diff = initial_price - final_price
                        price = initial_price - (price_diff * decay_factor)
                else:
                    price = listing_info.get('initial_price')
            
            # Process asynchronously if enabled
            if self.async_mode:
                tx_data = {
                    "nftId": nft_id,
                    "price": price
                }
                
                self.queue_transaction(
                    TransactionData(
                        tx_type="marketplace",
                        function_name="purchaseNFT",
                        params=tx_data,
                        sender_id=buyer_id
                    )
                )
                
                # Update local cache immediately for simulation continuity
                if nft_id in self.state_cache['marketplace']:
                    self.state_cache['marketplace'][nft_id]['status'] = 'pending_purchase'
                    
                if nft_id in self.state_cache['nfts']:
                    self.state_cache['nfts'][nft_id]['pending_owner'] = buyer_id
                
                self.logger.info(f"NFT {nft_id} purchase queued by {buyer_id}")
                return True
            else:
                # Process synchronously
                # First approve token transfer
                token = self.contracts["mockToken"]
                nonce = self._get_next_nonce(address)
                
                approval_tx = token.functions.approve(
                    self.contracts["market"].address,
                    price
                ).build_transaction({
                    'chainId': self.config["chain_id"],
                    'gas': 100000,
                    'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                    'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                    'nonce': nonce,
                })
                
                # Sign and send approval
                signed_approval = self.w3.eth.account.sign_transaction(approval_tx, private_key)
                approval_hash = self.w3.eth.send_raw_transaction(signed_approval.rawTransaction)
                
                # Wait for approval transaction
                self.w3.eth.wait_for_transaction_receipt(approval_hash)
                
                # Now purchase NFT
                market = self.contracts["market"]
                nonce = self._get_next_nonce(address)
                
                transaction = market.functions.purchaseNFT(nft_id).build_transaction({
                    'chainId': self.config["chain_id"],
                    'gas': self.gas_limit,
                    'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                    'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                    'nonce': nonce,
                })
                
                # Sign and send transaction
                signed_tx = self.w3.eth.account.sign_transaction(transaction, private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Wait for transaction receipt
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                
                # Update cache
                if receipt['status'] == 1:
                    if nft_id in self.state_cache['marketplace']:
                        self.state_cache['marketplace'][nft_id]['status'] = 'sold'
                        
                    if nft_id in self.state_cache['nfts']:
                        self.state_cache['nfts'][nft_id]['owner'] = buyer_id
                    
                    self.logger.info(f"NFT {nft_id} purchased by {buyer_id}")
                    return True
                else:
                    self.logger.error(f"Failed to purchase NFT {nft_id}")
                    return False
                
        except Exception as e:
            self.logger.error(f"Error purchasing NFT: {e}")
            return False
    
    def update_marketplace_prices(self):
        """
        Update prices of listed NFTs based on time decay
        """
        current_step = self.current_abm_step
        
        # Process only active listings with dynamic pricing
        for nft_id, listing in list(self.state_cache.get('marketplace', {}).items()):
            if listing.get('status') != 'listed' or not listing.get('dynamic_pricing', False):
                continue
                
            # Get NFT details
            nft_info = self._get_cached_nft_info(nft_id)
            if not nft_info:
                continue
                
            # Check if service time has passed
            service_step = nft_info.get('start_time', 0)
            if current_step >= service_step:
                # Service time has passed, mark as expired
                listing['status'] = 'expired'
                self.logger.info(f"Marking NFT {nft_id} as expired")
                continue
                
            # Calculate new price based on time decay
            listing_step = listing.get('listing_step', current_step)
            decay_duration = listing.get('decay_duration', 3)
            initial_price = listing.get('initial_price', 0)
            final_price = listing.get('final_price', 0)
            
            # Calculate progress through decay period
            if decay_duration > 0:
                elapsed_steps = current_step - listing_step
                progress = min(1.0, elapsed_steps / decay_duration)
                
                # Linear interpolation between initial and final price
                new_price = initial_price - (progress * (initial_price - final_price))
            else:
                new_price = initial_price
                
            # Update price if changed by more than 1%
            if abs(new_price - listing.get('current_price', 0)) > initial_price * 0.01:
                listing['current_price'] = new_price
                listing['last_updated'] = time.time()  # Real time for cache tracking
                
                self.logger.debug(f"Updated NFT {nft_id} price to {Web3.from_wei(new_price, 'ether')} ETH")
                
                # If synchronized with blockchain, also update on-chain
                if not self.async_mode and 'seller' in listing:
                    self._update_nft_price_on_blockchain(nft_id, new_price, listing['seller'])

    def _update_nft_price_on_blockchain(self, nft_id, new_price, seller_id):
        """
        Update NFT price on the blockchain
        
        Args:
            nft_id: NFT token ID
            new_price: New price in wei
            seller_id: Seller's ID
        """
        try:
            if seller_id not in self.accounts:
                return
                
            address = self.accounts[seller_id]["address"]
            private_key = self.accounts[seller_id]["private_key"]
            
            # Get the market contract
            market = self.contracts["market"]
            
            # Build price update transaction
            nonce = self._get_next_nonce(address)
            
            tx = market.functions.updateListingPrice(
                nft_id, 
                new_price
            ).build_transaction({
                'chainId': self.config["chain_id"],
                'gas': 100000,
                'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                'nonce': nonce,
            })
            
            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            self.logger.debug(f"Submitted price update transaction {tx_hash.hex()} for NFT {nft_id}")
        except Exception as e:
            self.logger.error(f"Error updating NFT price: {e}")
    # ================ Bundle handling ================
    
    def create_bundle_components(self, route_segments):
        """
        Generate signed offers from providers for bundle components
        
        Args:
            route_segments: List of route segments with provider preferences
                Each segment: {
                    'origin': [x, y],
                    'destination': [x, y],
                    'start_time': timestamp,
                    'preferred_mode': mode_id,
                    'preferred_provider': provider_id (optional)
                }
                
        Returns:
            dict: Provider signed offers for each segment
        """
        bundle_components = {}
        
        # Process each segment
        for i, segment in enumerate(route_segments):
            segment_id = f"segment_{i}"
            
            # Find suitable providers
            suitable_providers = self._find_providers_for_segment(segment)
            
            # Sort providers by preference if specified
            if 'preferred_provider' in segment:
                suitable_providers.sort(key=lambda p: p['provider_id'] == segment['preferred_provider'], reverse=True)
            
            # Get signed offers from top providers
            max_providers = 3  # Limit to top 3 providers per segment
            segment_offers = []
            
            for provider in suitable_providers[:max_providers]:
                # Generate offer for this segment
                offer = self._generate_provider_offer(
                    provider['provider_id'],
                    segment['origin'],
                    segment['destination'],
                    segment['start_time'],
                    segment.get('duration', 1800)  # Default 30 mins
                )
                
                if offer:
                    segment_offers.append(offer)
            
            bundle_components[segment_id] = segment_offers
        
        return bundle_components
    
    def execute_bundle_purchase(self, bundle_details, commuter_id):
        """
        Execute the bundle purchase with minimal on-chain operations
        
        Args:
            bundle_details: Details of the bundle components
                {
                    'components': {
                        'segment_id': {
                            'provider_id': provider_id,
                            'price': price,
                            'start_time': timestamp,
                            'duration': seconds,
                            'offer_signature': signature
                        }
                    },
                    'total_price': total_price,
                    'name': bundle_name
                }
            commuter_id: ID of the commuter making the purchase
            
        Returns:
            tuple: (success, bundle_id)
        """
        try:
            # Verify commuter exists
            if commuter_id not in self.accounts:
                self.logger.error(f"Commuter {commuter_id} not registered")
                return False, None
                
            address = self.accounts[commuter_id]["address"]
            private_key = self.accounts[commuter_id]["private_key"]
            
            # Process bundle components
            component_nfts = []
            
            # Process each component
            for segment_id, details in bundle_details['components'].items():
                provider_id = details['provider_id']
                
                # Create NFT for this service
                service_details = {
                    'price': details['price'],
                    'start_time': details['start_time'],
                    'duration': details['duration'],
                    'route_details': {
                        'segment_id': segment_id,
                        'offer_signature': details.get('offer_signature')
                    }
                }
                
                success, nft_id = self.create_nft(service_details, provider_id, commuter_id)
                
                if success:
                    component_nfts.append(nft_id)
                else:
                    self.logger.error(f"Failed to create NFT for segment {segment_id}")
                    # Continue to try other components
            
            # If no components were created, fail
            if not component_nfts:
                self.logger.error("No components were successfully created")
                return False, None
            
            # Now create the bundle
            if self.async_mode:
                # Generate temporary bundle ID
                bundle_id = int(uuid.uuid4().int & (2**64 - 1))
                
                tx_data = {
                    "tokenIds": component_nfts,
                    "name": bundle_details.get('name', f"MaaS Bundle {bundle_id}"),
                    "price": Web3.to_wei(str(bundle_details.get('total_price', 0)), 'ether')
                }
                
                self.queue_transaction(
                    TransactionData(
                        tx_type="bundle",
                        function_name="createServiceBundle",
                        params=tx_data,
                        sender_id=commuter_id
                    )
                )
                
                # Update local cache
                self.state_cache['bundles'] = self.state_cache.get('bundles', {})
                self.state_cache['bundles'][bundle_id] = {
                    'owner': commuter_id,
                    'components': component_nfts,
                    'name': bundle_details.get('name', f"MaaS Bundle {bundle_id}"),
                    'price': bundle_details.get('total_price', 0),
                    'status': 'pending',
                    'last_updated': time.time()
                }
                
                self.logger.info(f"Bundle creation queued with {len(component_nfts)} components")
                return True, bundle_id
            else:
                # Process synchronously
                nft = self.contracts["nft"]
                nonce = self._get_next_nonce(address)
                
                transaction = nft.functions.createServiceBundle(
                    component_nfts,
                    bundle_details.get('name', f"MaaS Bundle"),
                    Web3.to_wei(str(bundle_details.get('total_price', 0)), 'ether')
                ).build_transaction({
                    'chainId': self.config["chain_id"],
                    'gas': self.gas_limit,
                    'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                    'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                    'nonce': nonce,
                })
                
                # Sign and send transaction
                signed_tx = self.w3.eth.account.sign_transaction(transaction, private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Wait for transaction receipt
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                
                # Extract bundle ID from event logs
                bundle_id = None
                if receipt['status'] == 1:
                    for log in receipt['logs']:
                        if log['address'].lower() == self.contracts["nft"].address.lower():
                            # Try to decode BundleCreated event
                            try:
                                evt = self.contracts["nft"].events.BundleCreated().process_log(log)
                                bundle_id = evt['args']['bundleId']
                                break
                            except:
                                continue
                
                if bundle_id:
                    # Update cache
                    self.state_cache['bundles'] = self.state_cache.get('bundles', {})
                    self.state_cache['bundles'][bundle_id] = {
                        'owner': commuter_id,
                        'components': component_nfts,
                        'name': bundle_details.get('name'),
                        'price': bundle_details.get('total_price'),
                        'status': 'created',
                        'last_updated': time.time()
                    }
                    
                    self.logger.info(f"Bundle {bundle_id} created with {len(component_nfts)} components")
                    return True, bundle_id
                else:
                    self.logger.error("Failed to extract bundle ID from receipt")
                    return False, None
                
        except Exception as e:
            self.logger.error(f"Error creating bundle: {e}")
            return False, None
    
    def sign_bundle_offer(self, provider, offer):
        """
        Sign a bundle offer with provider's private key
        
        Args:
            provider: Provider agent
            offer: Offer details
                
        Returns:
            str: Signature
        """
        # Get provider's account
        if provider.unique_id not in self.accounts:
            self.logger.error(f"Provider {provider.unique_id} not registered")
            return None
        
        private_key = self.accounts[provider.unique_id]["private_key"]
        
        # Create message hash from offer
        # In a real implementation, this would properly hash the offer details
        # For simulation purposes, we'll use a simplified approach
        offer_data = f"{offer['provider_id']}_{offer['segment_id']}_{offer['price']}_{offer['start_time']}"
        
        # "Sign" the message (this is a simplified simulation of signing)
        # In production, you would use proper cryptographic signatures
        signature = f"0x{offer['provider_id']}_{int(time.time())}_{hash(offer_data) % 10000:04d}"
        
        return signature
        
    # ================ Helper methods for MaaS bundles ================
    
    def _find_providers_for_segment(self, segment):
        """Find suitable providers for a route segment"""
        suitable_providers = []
        
        # Get all registered providers from cache or blockchain
        providers = self.state_cache.get('providers', {})
        if not providers:
            # Fallback to registry query
            provider_count = self.contracts["registry"].functions.numberOfProviders().call()
            for i in range(1, provider_count + 1):
                provider_address = self.contracts["registry"].functions.providerIdToAddress(i).call()
                if provider_address != '0x0000000000000000000000000000000000000000':
                    provider_data = self.contracts["registry"].functions.getProvider(provider_address).call()
                    providers[provider_data[0]] = {
                        'address': provider_address,
                        'data': provider_data
                    }
        
        # Filter by mode if specified
        preferred_mode = segment.get('preferred_mode')
        
        for provider_id, provider_info in providers.items():
            provider_data = provider_info.get('data', {})
            
            # Check if mode matches if specified
            if preferred_mode is not None and provider_data.get('modeType') != preferred_mode:
                continue
                
            # Check if provider has sufficient capacity
            if provider_data.get('availableCapacity', 0) <= 0:
                continue
                
            # Check if provider service area covers this segment
            service_center = provider_data.get('serviceCenter', [50, 50])
            service_radius = provider_data.get('servicingArea', 25)
            
            # Check if origin and destination are within service area
            origin_distance = ((segment['origin'][0] - service_center[0])**2 + 
                              (segment['origin'][1] - service_center[1])**2)**0.5
                              
            dest_distance = ((segment['destination'][0] - service_center[0])**2 + 
                            (segment['destination'][1] - service_center[1])**2)**0.5
            
            if origin_distance > service_radius or dest_distance > service_radius:
                continue
            
            # Provider is suitable
            suitable_providers.append({
                'provider_id': provider_id,
                'mode_type': provider_data.get('modeType'),
                'base_price': provider_data.get('basePrice', 0),
                'capacity': provider_data.get('availableCapacity', 0)
            })
        
        return suitable_providers
    
    def _generate_provider_offer(self, provider_id, origin, destination, start_time, duration):
        """Generate an offer from a provider for a route segment"""
        try:
            if provider_id not in self.state_cache.get('providers', {}):
                self.logger.error(f"Provider {provider_id} not found in cache")
                return None
                
            provider_info = self.state_cache['providers'][provider_id]
            provider_data = provider_info.get('data', {})
            
            # Calculate route
            route_details = self._calculate_route(origin, destination)
            
            # Calculate price based on provider's base price and distance
            route_obj = json.loads(route_details)
            distance = route_obj.get('distance', 0)
            base_price = provider_data.get('basePrice', 0)
            
            if isinstance(base_price, str) and base_price.startswith('0x'):
                # Convert hex string to int if needed
                base_price = int(base_price, 16)
                
            # Convert from wei if needed
            if base_price > 1e10:  # Assume it's in wei
                base_price = Web3.from_wei(base_price, 'ether')
                
            price = base_price * distance
            
            # Create offer object
            offer = {
                'provider_id': provider_id,
                'origin': origin,
                'destination': destination,
                'start_time': start_time,
                'duration': duration,
                'price': price,
                'route_details': route_obj
            }
            
            # Generate signature for the offer (simplified for now)
            # In production, you'd use the provider's private key to sign the offer
            offer['signature'] = f"0x{provider_id}_{start_time}_{int(price * 100)}"
            
            return offer
            
        except Exception as e:
            self.logger.error(f"Error generating provider offer: {e}")
            return None
    
    # ================ Asynchronous transaction handling ================
    
    def queue_transaction(self, transaction_data):
        """
        Add transaction to queue
        
        Args:
            transaction_data: TransactionData object
        """
        self.tx_queue.append(transaction_data)
        self.logger.debug(f"Transaction queued: {transaction_data.tx_type} - {transaction_data.function_name}")
        
        # If queue exceeds batch size limit, trigger processing
        if len(self.tx_queue) >= self.batch_size_limit and self.async_mode:
            self.thread_pool.submit(self._process_transaction_batch)
    
    def _process_transaction_batch(self):
        """Process a batch of transactions from the queue with proper nonce handling"""
        if not self.tx_queue:
            return
            
        # Take a batch of transactions
        batch_size = min(self.batch_size_limit, len(self.tx_queue))
        batch = []
        for _ in range(batch_size):
            if self.tx_queue:
                batch.append(self.tx_queue.popleft())
        
        self.logger.info(f"Processing batch of {len(batch)} transactions")
        
        # Group transactions by sender_id first, then by type
        tx_by_sender = {}
        for tx in batch:
            if tx.sender_id not in tx_by_sender:
                tx_by_sender[tx.sender_id] = {
                    'registration': [], 'request': [], 'offer': [],
                    'nft': [], 'marketplace': [], 'bundle': []
                }
            tx_by_sender[tx.sender_id][tx.tx_type].append(tx)
        
        # Process each sender's transactions to maintain proper nonce sequence
        for sender_id, grouped_tx in tx_by_sender.items():
            # Get current nonce for this sender
            if sender_id not in self.accounts:
                self.logger.error(f"Account for {sender_id} not found")
                continue
                
            address = self.accounts[sender_id]["address"]
            try:
                current_nonce = self.w3.eth.get_transaction_count(address, 'pending')
            except Exception as e:
                self.logger.error(f"Error getting nonce for {address}: {e}")
                continue
                
            # Process each transaction type, maintaining nonce sequence
            for tx_type in ['registration', 'request', 'offer', 'nft', 'marketplace', 'bundle']:
                transactions = grouped_tx[tx_type]
                if not transactions:
                    continue
                    
                self.logger.debug(f"Processing {len(transactions)} {tx_type} transactions for sender {sender_id}")
                
                # Process based on transaction type, using and incrementing nonce
                if tx_type == 'registration':
                    current_nonce = self._process_registrations_with_nonce(transactions, current_nonce)
                elif tx_type == 'request':
                    current_nonce = self._process_requests_with_nonce(transactions, current_nonce)
                elif tx_type == 'offer':
                    current_nonce = self._process_offers_with_nonce(transactions, current_nonce)
                elif tx_type == 'nft':
                    current_nonce = self._process_nft_operations_with_nonce(transactions, current_nonce)
                elif tx_type == 'marketplace':
                    current_nonce = self._process_marketplace_operations_with_nonce(transactions, current_nonce)
                elif tx_type == 'bundle':
                    current_nonce = self._process_bundle_operations_with_nonce(transactions, current_nonce)
            
            # Update stored nonce for this address
            self.tx_nonce_map[address] = current_nonce
        
        self.stats['batch_operations'] += 1
        return
    
    def _process_registrations_with_nonce(self, transactions, start_nonce):
        """Process batch of registration transactions with sequential nonces"""
        current_nonce = start_nonce
        
        for tx in transactions:
            sender_id = tx.sender_id
            
            if sender_id not in self.accounts:
                self.logger.error(f"Account for {sender_id} not found")
                continue
                
            address = self.accounts[sender_id]["address"]
            private_key = self.accounts[sender_id]["private_key"]
            
            try:
                # Build transaction with explicit nonce
                registry = self.contracts["registry"]
                
                if tx.function_name == "addCommuter":
                    params = tx.params
                    tx_obj = registry.functions.addCommuter(
                        params["commuterId"],
                        params["location"],
                        params["incomeLevel"],
                        params["preferredMode"],
                        params["age"],
                        params["hasDisability"],
                        params["techAccess"],
                        params["healthStatus"],
                        params["paymentScheme"]
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': self.gas_limit,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': current_nonce,
                    })
                elif tx.function_name == "addProvider":
                    params = tx.params
                    tx_obj = registry.functions.addProvider(
                        params["providerId"],
                        params
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': self.gas_limit,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': current_nonce,
                    })
                else:
                    self.logger.error(f"Unknown registration function: {tx.function_name}")
                    continue
                
                # Sign and send transaction
                signed_tx = self.w3.eth.account.sign_transaction(tx_obj, private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Store transaction hash for tracking
                tx.tx_hash = tx_hash.hex()
                self.pending_transactions[tx_hash.hex()] = tx
                
                self.logger.info(f"Registration transaction submitted: {tx_hash.hex()} with nonce {current_nonce}")
                self.stats['transactions_submitted'] += 1
                
                # Increment nonce for next transaction
                current_nonce += 1
                
            except Exception as e:
                self.logger.error(f"Error processing registration transaction: {e}")
        
        return current_nonce

    def _process_requests_with_nonce(self, transactions, start_nonce):
        """Process request transactions with improved nonce management"""
        # Create a dictionary to track nonces per address within this batch
        address_to_nonce = {}
        
        for tx in transactions:
            sender_id = tx.sender_id
            
            if sender_id not in self.accounts:
                self.logger.error(f"Account for {sender_id} not found")
                continue
                    
            address = self.accounts[sender_id]["address"]
            private_key = self.accounts[sender_id]["private_key"]
            
            try:
                # If this is the first transaction for this address in this batch,
                # get a fresh nonce directly from the blockchain
                if address not in address_to_nonce:
                    current_nonce = self.w3.eth.get_transaction_count(address, 'pending')
                    address_to_nonce[address] = current_nonce
                else:
                    # Use the next nonce for this address
                    current_nonce = address_to_nonce[address]
                
                # Build transaction with the current nonce
                params = tx.params
                
                # Use facade contract for travel requests
                facade = self.contracts["facade"]
                
                # Make sure we have the basic required parameters
                if not all(k in params for k in ["requestId", "origin", "destination", "startTime"]):
                    self.logger.error(f"Missing required parameters: {params}")
                    continue
                    
                # Prepare parameters for contract function
                request_id = params.get("requestId", 0)
                
                # Convert tuples to lists if needed
                origin = params.get("origin", [0, 0])
                if isinstance(origin, tuple):
                    origin = list(origin)
                    
                destination = params.get("destination", [0, 0])
                if isinstance(destination, tuple):
                    destination = list(destination)
                    
                start_time = params.get("startTime", 0)
                purpose = params.get("purpose", 0)
                flexible_time = params.get("flexibleTime", "medium")
                requirement_values = params.get("requirementValues", [False, False, False, False])
                requirement_keys = params.get("requirementKeys", ["wheelchair", "assistance", "child_seat", "pet_friendly"])
                
                # Extract path_info properly or calculate it if missing
                if "pathInfo" in params:
                    path_info = params["pathInfo"]
                else:
                    # Calculate path info from origin/destination
                    distance = ((destination[0] - origin[0])**2 + (destination[1] - origin[1])**2)**0.5
                    path_info = json.dumps({
                        'origin': origin,
                        'destination': destination,
                        'distance': distance
                    })
                
                # Log the actual parameters being sent
                self.logger.info(f"Creating travel request with params: requestId={request_id}, "
                            f"origin={origin}, destination={destination}, startTime={start_time}, "
                            f"purpose={purpose}, flexibleTime={flexible_time}")
                            
                tx_obj = facade.functions.createTravelRequest(
                    request_id,
                    origin,
                    destination,
                    start_time,
                    purpose,
                    flexible_time,
                    requirement_values,
                    requirement_keys,
                    path_info
                ).build_transaction({
                    'chainId': self.config["chain_id"],
                    'gas': self.gas_limit * 3,  # Increase gas limit for complex transactions
                    'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                    'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                    'nonce': current_nonce,
                })
                
                # Sign and send transaction
                signed_tx = self.w3.eth.account.sign_transaction(tx_obj, private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Store transaction hash for tracking
                tx.tx_hash = tx_hash.hex()
                self.pending_transactions[tx_hash.hex()] = tx
                
                # Update the nonce for next transaction from this address
                address_to_nonce[address] = current_nonce + 1
                
                # Update the global nonce map too
                self.tx_nonce_map[address] = current_nonce + 1
                
                self.logger.info(f"Request transaction submitted: {tx_hash.hex()} with nonce {current_nonce}")
                self.stats['transactions_submitted'] += 1
                
            except Exception as e:
                error_str = str(e)
                self.logger.error(f"Error processing request transaction: {e}")
                
                # Handle nonce too low errors
                if "nonce too low" in error_str.lower():
                    # Extract the expected nonce from the error message
                    import re
                    match = re.search(r"Expected nonce to be (\d+)", error_str)
                    if match:
                        expected_nonce = int(match.group(1))
                        # Update our tracking
                        address_to_nonce[address] = expected_nonce
                        self.tx_nonce_map[address] = expected_nonce
                        self.logger.info(f"Updated nonce for {address} to {expected_nonce}")
                
                # Log the problematic parameters for debugging
                if 'params' in tx.__dict__:
                    self.logger.error(f"Problematic request parameters: {tx.params}")
        
        # Return the highest nonce used for any address
        return max(address_to_nonce.values()) if address_to_nonce else start_nonce

    def _process_offers_with_nonce(self, transactions, start_nonce):
        """Process offer transactions with explicit nonce management"""
        auction = self.contracts["auction"]
        current_nonce = start_nonce
        
        # Track the last nonce used for each sender in this batch
        sender_last_nonce = {}
        
        for tx in transactions:
            sender_id = tx.sender_id
            
            if sender_id not in self.accounts:
                self.logger.error(f"Account for {sender_id} not found")
                continue
                    
            address = self.accounts[sender_id]["address"]
            private_key = self.accounts[sender_id]["private_key"]
            
            try:
                # Check if we've already processed a transaction for this sender in this batch
                if address in sender_last_nonce:
                    # Use the last nonce we recorded plus one
                    current_nonce = sender_last_nonce[address] + 1
                else:
                    # For the first transaction from this sender in this batch,
                    # get the latest nonce from the network
                    current_nonce = self.w3.eth.get_transaction_count(address, 'pending')
                
                # Build transaction with explicit nonce
                params = tx.params
                
                tx_obj = auction.functions.submitOffer(params).build_transaction({
                    'chainId': self.config["chain_id"],
                    'gas': self.gas_limit,
                    'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                    'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                    'nonce': current_nonce,
                })
                
                # Sign and send transaction
                signed_tx = self.w3.eth.account.sign_transaction(tx_obj, private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Store transaction hash for tracking
                tx.tx_hash = tx_hash.hex()
                self.pending_transactions[tx_hash.hex()] = tx
                
                # Record the nonce we used for this sender
                sender_last_nonce[address] = current_nonce
                
                # Update the global nonce map too
                self.tx_nonce_map[address] = current_nonce + 1
                
                self.logger.info(f"Offer transaction submitted: {tx_hash.hex()} with nonce {current_nonce}")
                self.stats['transactions_submitted'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing offer transaction: {e}")
                
                # If we get a "nonce too low" error, try to recover
                if "nonce too low" in str(e).lower():
                    # Try to extract the expected nonce from the error message
                    import re
                    match = re.search(r"Expected nonce to be (\d+)", str(e))
                    if match:
                        expected_nonce = int(match.group(1))
                        self.logger.info(f"Updating nonce tracking for {address} to {expected_nonce}")
                        self.tx_nonce_map[address] = expected_nonce
                        sender_last_nonce[address] = expected_nonce - 1
        
        # Return the highest nonce used, though this is less important now
        # since we're tracking by address
        return max(sender_last_nonce.values()) if sender_last_nonce else current_nonce

    def _process_nft_operations_with_nonce(self, transactions, start_nonce):
        """Process NFT operations with explicit nonce management"""
        nft = self.contracts["nft"]
        token = self.contracts["mockToken"]
        current_nonce = start_nonce
        
        # Track the last nonce used for each sender in this batch
        # This helps prevent nonce conflicts when processing multiple transactions for the same sender
        sender_last_nonce = {}
        
        for tx in transactions:
            sender_id = tx.sender_id
            
            if sender_id not in self.accounts:
                self.logger.error(f"Account for {sender_id} not found")
                continue
                    
            address = self.accounts[sender_id]["address"]
            private_key = self.accounts[sender_id]["private_key"]
            
            try:
                # Check if we've already processed a transaction for this sender in this batch
                if address in sender_last_nonce:
                    # Use the last nonce we recorded plus one
                    current_nonce = sender_last_nonce[address] + 1
                else:
                    # For the first transaction from this sender in this batch,
                    # get the latest nonce from the network to ensure we're up to date
                    current_nonce = self.w3.eth.get_transaction_count(address, 'pending')
                
                # Build transaction with explicit nonce
                params = tx.params
                
                if tx.function_name == "mintServiceNFT":
                    # First check if we need to approve token spending
                    if 'price' in params:
                        approval_tx = token.functions.approve(
                            nft.address,
                            params["price"]
                        ).build_transaction({
                            'chainId': self.config["chain_id"],
                            'gas': 100000,
                            'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                            'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                            'nonce': current_nonce,
                        })
                        
                        # Sign and send approval
                        signed_approval = self.w3.eth.account.sign_transaction(approval_tx, private_key)
                        approval_hash = self.w3.eth.send_raw_transaction(signed_approval.rawTransaction)
                        
                        # Record the nonce we used for this sender
                        sender_last_nonce[address] = current_nonce
                        
                        # Update the global nonce map too
                        self.tx_nonce_map[address] = current_nonce + 1
                        
                        # Wait for approval transaction
                        receipt = self.w3.eth.wait_for_transaction_receipt(approval_hash)
                        if receipt.status != 1:
                            self.logger.error(f"Token approval failed for NFT minting")
                            continue
                        
                        # Increment nonce for next transaction
                        current_nonce += 1
                    
                    # Now mint the NFT
                    tx_obj = nft.functions.mintServiceNFT(
                        params["requestId"],
                        params["providerId"],
                        params["routeDetails"],
                        params["price"],
                        params["startTime"],
                        params["duration"],
                        params["tokenURI"]
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': self.gas_limit,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': current_nonce,
                    })
                else:
                    self.logger.error(f"Unknown NFT function: {tx.function_name}")
                    continue
                
                # Sign and send transaction
                signed_tx = self.w3.eth.account.sign_transaction(tx_obj, private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Store transaction hash for tracking
                tx.tx_hash = tx_hash.hex()
                self.pending_transactions[tx_hash.hex()] = tx
                
                # Record the nonce we used for this sender
                sender_last_nonce[address] = current_nonce
                
                # Update the global nonce map too
                self.tx_nonce_map[address] = current_nonce + 1
                
                self.logger.info(f"NFT transaction submitted: {tx_hash.hex()} with nonce {current_nonce}")
                self.stats['transactions_submitted'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing NFT transaction: {e}")
                
                # If we get a "nonce too low" error, try to recover
                if "nonce too low" in str(e).lower():
                    # Try to extract the expected nonce from the error message
                    import re
                    match = re.search(r"Expected nonce to be (\d+)", str(e))
                    if match:
                        expected_nonce = int(match.group(1))
                        self.logger.info(f"Updating nonce tracking for {address} to {expected_nonce}")
                        self.tx_nonce_map[address] = expected_nonce
                        sender_last_nonce[address] = expected_nonce - 1
        
        # Return the highest nonce used, though this is less important now
        # since we're tracking by address
        return max(sender_last_nonce.values()) if sender_last_nonce else current_nonce

    def _process_marketplace_operations_with_nonce(self, transactions, start_nonce):
        """Process marketplace operations with explicit nonce management"""
        market = self.contracts["market"]
        nft = self.contracts["nft"]
        token = self.contracts["mockToken"]
        
        # Track the last nonce used for each sender in this batch
        sender_last_nonce = {}
        
        for tx in transactions:
            sender_id = tx.sender_id
            
            if sender_id not in self.accounts:
                self.logger.error(f"Account for {sender_id} not found")
                continue
                    
            address = self.accounts[sender_id]["address"]
            private_key = self.accounts[sender_id]["private_key"]
            
            try:
                # Check if we've already processed a transaction for this sender in this batch
                if address in sender_last_nonce:
                    # Use the last nonce we recorded plus one
                    current_nonce = sender_last_nonce[address] + 1
                else:
                    # For the first transaction from this sender in this batch,
                    # get the latest nonce from the network
                    current_nonce = self.w3.eth.get_transaction_count(address, 'pending')
                
                # Build transaction with explicit nonce
                params = tx.params
                
                if tx.function_name == "listNFTForSale":
                    # First approve NFT transfer to market
                    approval_tx = nft.functions.approve(
                        market.address,
                        params["nftId"]
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': 100000,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': current_nonce,
                    })
                    
                    # Sign and send approval
                    signed_approval = self.w3.eth.account.sign_transaction(approval_tx, private_key)
                    approval_hash = self.w3.eth.send_raw_transaction(signed_approval.rawTransaction)
                    
                    # Record the nonce we used
                    sender_last_nonce[address] = current_nonce
                    current_nonce += 1  # Increment for the next transaction
                    
                    # Wait for approval transaction
                    receipt = self.w3.eth.wait_for_transaction_receipt(approval_hash)
                    if receipt.status != 1:
                        self.logger.error(f"NFT approval failed for marketplace listing")
                        continue
                    
                    # Now list for sale
                    tx_obj = market.functions.listNFTForSale(
                        params["nftId"],
                        params["price"]
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': self.gas_limit,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': current_nonce,
                    })
                
                elif tx.function_name == "listNFTWithDynamicPricing":
                    # First approve NFT transfer to market
                    approval_tx = nft.functions.approve(
                        market.address,
                        params["nftId"]
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': 100000,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': current_nonce,
                    })
                    
                    # Sign and send approval
                    signed_approval = self.w3.eth.account.sign_transaction(approval_tx, private_key)
                    approval_hash = self.w3.eth.send_raw_transaction(signed_approval.rawTransaction)
                    
                    # Record the nonce we used
                    sender_last_nonce[address] = current_nonce
                    current_nonce += 1  # Increment for the next transaction
                    
                    # Wait for approval transaction
                    receipt = self.w3.eth.wait_for_transaction_receipt(approval_hash)
                    if receipt.status != 1:
                        self.logger.error(f"NFT approval failed for marketplace dynamic listing")
                        continue
                    
                    # Now list with dynamic pricing
                    tx_obj = market.functions.listNFTWithDynamicPricing(
                        params["nftId"],
                        params["initialPrice"],
                        params["finalPrice"],
                        params["decayDuration"]
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': self.gas_limit,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': current_nonce,
                    })
                
                elif tx.function_name == "purchaseNFT":
                    # First approve token transfer
                    approval_tx = token.functions.approve(
                        market.address,
                        params["price"]
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': 100000,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': current_nonce,
                    })
                    
                    # Sign and send approval
                    signed_approval = self.w3.eth.account.sign_transaction(approval_tx, private_key)
                    approval_hash = self.w3.eth.send_raw_transaction(signed_approval.rawTransaction)
                    
                    # Record the nonce we used
                    sender_last_nonce[address] = current_nonce
                    current_nonce += 1  # Increment for the next transaction
                    
                    # Wait for approval transaction
                    receipt = self.w3.eth.wait_for_transaction_receipt(approval_hash)
                    if receipt.status != 1:
                        self.logger.error(f"Token approval failed for NFT purchase")
                        continue
                    
                    # Now purchase NFT
                    tx_obj = market.functions.purchaseNFT(params["nftId"]).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': self.gas_limit,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': current_nonce,
                    })
                else:
                    self.logger.error(f"Unknown marketplace function: {tx.function_name}")
                    continue
                
                # Sign and send transaction
                signed_tx = self.w3.eth.account.sign_transaction(tx_obj, private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Store transaction hash for tracking
                tx.tx_hash = tx_hash.hex()
                self.pending_transactions[tx_hash.hex()] = tx
                
                # Record the nonce we used
                sender_last_nonce[address] = current_nonce
                
                # Update the global nonce map
                self.tx_nonce_map[address] = current_nonce + 1
                
                self.logger.info(f"Marketplace transaction submitted: {tx_hash.hex()} with nonce {current_nonce}")
                self.stats['transactions_submitted'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing marketplace transaction: {e}")
                
                # If we get a "nonce too low" error, try to recover
                if "nonce too low" in str(e).lower():
                    # Update our tracking with the expected nonce from the error
                    error_str = str(e)
                    import re
                    match = re.search(r"Expected nonce to be (\d+)", error_str)
                    if match:
                        expected_nonce = int(match.group(1))
                        self.logger.info(f"Updating nonce tracking for {address} to {expected_nonce}")
                        self.tx_nonce_map[address] = expected_nonce
                        sender_last_nonce[address] = expected_nonce - 1
        
        # Return the highest nonce used
        return max(sender_last_nonce.values()) if sender_last_nonce else start_nonce

    def _process_bundle_operations_with_nonce(self, transactions, start_nonce):
        """Process bundle operations with explicit nonce management"""
        nft = self.contracts["nft"]
        current_nonce = start_nonce
        
        # Track the last nonce used for each sender in this batch
        sender_last_nonce = {}
        
        for tx in transactions:
            sender_id = tx.sender_id
            
            if sender_id not in self.accounts:
                self.logger.error(f"Account for {sender_id} not found")
                continue
                    
            address = self.accounts[sender_id]["address"]
            private_key = self.accounts[sender_id]["private_key"]
            
            try:
                # Check if we've already processed a transaction for this sender in this batch
                if address in sender_last_nonce:
                    # Use the last nonce we recorded plus one
                    current_nonce = sender_last_nonce[address] + 1
                else:
                    # For the first transaction from this sender in this batch,
                    # get the latest nonce from the network
                    current_nonce = self.w3.eth.get_transaction_count(address, 'pending')
                
                # Build transaction with explicit nonce
                params = tx.params
                
                if tx.function_name == "createServiceBundle":
                    tx_obj = nft.functions.createServiceBundle(
                        params["tokenIds"],
                        params["name"],
                        params["price"]
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': self.gas_limit,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': current_nonce,
                    })
                else:
                    self.logger.error(f"Unknown bundle function: {tx.function_name}")
                    continue
                
                # Sign and send transaction
                signed_tx = self.w3.eth.account.sign_transaction(tx_obj, private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Store transaction hash for tracking
                tx.tx_hash = tx_hash.hex()
                self.pending_transactions[tx_hash.hex()] = tx
                
                # Record the nonce we used for this sender
                sender_last_nonce[address] = current_nonce
                
                # Update the global nonce map too
                self.tx_nonce_map[address] = current_nonce + 1
                
                self.logger.info(f"Bundle transaction submitted: {tx_hash.hex()} with nonce {current_nonce}")
                self.stats['transactions_submitted'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing bundle transaction: {e}")
                
                # If we get a "nonce too low" error, try to recover
                error_str = str(e)
                if "nonce too low" in error_str.lower():
                    # Try to extract the expected nonce from the error message
                    import re
                    match = re.search(r"Expected nonce to be (\d+)", error_str)
                    if match:
                        expected_nonce = int(match.group(1))
                        self.logger.info(f"Updating nonce tracking for {address} to {expected_nonce}")
                        self.tx_nonce_map[address] = expected_nonce
                        sender_last_nonce[address] = expected_nonce - 1
        
        # Return the highest nonce used, though this is less important now
        # since we're tracking by address
        return max(sender_last_nonce.values()) if sender_last_nonce else current_nonce

    def _transaction_processor(self):
        """Background thread for processing transaction queue"""
        while self.running:
            if len(self.tx_queue) > 0:
                self._process_transaction_batch()
            else:
                # Sleep to avoid busy waiting
                time.sleep(0.1)
    
    def _process_registrations(self, transactions):
        """Process batch of registration transactions"""
        for tx in transactions:
            sender_id = tx.sender_id
            
            if sender_id not in self.accounts:
                self.logger.error(f"Account for {sender_id} not found")
                continue
                
            address = self.accounts[sender_id]["address"]
            private_key = self.accounts[sender_id]["private_key"]
            
            try:
                # Build transaction
                nonce = self._get_next_nonce(address)
                registry = self.contracts["registry"]
                
                if tx.function_name == "addCommuter":
                    params = tx.params
                    tx_obj = registry.functions.addCommuter(
                        params["commuterId"],
                        params["location"],
                        params["incomeLevel"],
                        params["preferredMode"],
                        params["age"],
                        params["hasDisability"],
                        params["techAccess"],
                        params["healthStatus"],
                        params["paymentScheme"]
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': self.gas_limit,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': nonce,
                    })
                elif tx.function_name == "addProvider":
                    params = tx.params
                    tx_obj = registry.functions.addProvider(
                        params["providerId"],
                        params
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': self.gas_limit,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': nonce,
                    })
                else:
                    self.logger.error(f"Unknown registration function: {tx.function_name}")
                    continue
                
                # Sign and send transaction
                signed_tx = self.w3.eth.account.sign_transaction(tx_obj, private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Store transaction hash for tracking
                tx.tx_hash = tx_hash.hex()
                self.pending_transactions[tx_hash.hex()] = tx
                
                self.logger.info(f"Registration transaction submitted: {tx_hash.hex()}")
                self.stats['transactions_submitted'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing registration transaction: {e}")

    def _process_offers(self, transactions):
        """Process batch of offer transactions"""
        auction = self.contracts["auction"]
        
        for tx in transactions:
            sender_id = tx.sender_id
            
            if sender_id not in self.accounts:
                self.logger.error(f"Account for {sender_id} not found")
                continue
                
            address = self.accounts[sender_id]["address"]
            private_key = self.accounts[sender_id]["private_key"]
            
            try:
                # Build transaction
                nonce = self._get_next_nonce(address)
                params = tx.params
                
                tx_obj = auction.functions.submitOffer(params).build_transaction({
                    'chainId': self.config["chain_id"],
                    'gas': self.gas_limit,
                    'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                    'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                    'nonce': nonce,
                })
                
                # Sign and send transaction
                signed_tx = self.w3.eth.account.sign_transaction(tx_obj, private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Store transaction hash for tracking
                tx.tx_hash = tx_hash.hex()
                self.pending_transactions[tx_hash.hex()] = tx
                
                self.logger.info(f"Offer transaction submitted: {tx_hash.hex()}")
                self.stats['transactions_submitted'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing offer transaction: {e}")
    
    def _process_nft_operations(self, transactions):
        """Process batch of NFT operations"""
        nft = self.contracts["nft"]
        
        for tx in transactions:
            sender_id = tx.sender_id
            
            if sender_id not in self.accounts:
                self.logger.error(f"Account for {sender_id} not found")
                continue
                
            address = self.accounts[sender_id]["address"]
            private_key = self.accounts[sender_id]["private_key"]
            
            try:
                # Build transaction
                nonce = self._get_next_nonce(address)
                params = tx.params
                
                if tx.function_name == "mintServiceNFT":
                    # First check if we need to approve token spending
                    if 'price' in params:
                        token = self.contracts["mockToken"]
                        
                        approval_tx = token.functions.approve(
                            nft.address,
                            params["price"]
                        ).build_transaction({
                            'chainId': self.config["chain_id"],
                            'gas': 100000,
                            'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                            'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                            'nonce': nonce,
                        })
                        
                        # Sign and send approval
                        signed_approval = self.w3.eth.account.sign_transaction(approval_tx, private_key)
                        approval_hash = self.w3.eth.send_raw_transaction(signed_approval.rawTransaction)
                        
                        # Wait for approval transaction
                        self.w3.eth.wait_for_transaction_receipt(approval_hash)
                        
                        # Increment nonce for next transaction
                        nonce = self._get_next_nonce(address)
                    
                    # Now mint the NFT
                    tx_obj = nft.functions.mintServiceNFT(
                        params["requestId"],
                        params["providerId"],
                        params["routeDetails"],
                        params["price"],
                        params["startTime"],
                        params["duration"],
                        params["tokenURI"]
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': self.gas_limit,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': nonce,
                    })
                else:
                    self.logger.error(f"Unknown NFT function: {tx.function_name}")
                    continue
                
                # Sign and send transaction
                signed_tx = self.w3.eth.account.sign_transaction(tx_obj, private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Store transaction hash for tracking
                tx.tx_hash = tx_hash.hex()
                self.pending_transactions[tx_hash.hex()] = tx
                
                self.logger.info(f"NFT transaction submitted: {tx_hash.hex()}")
                self.stats['transactions_submitted'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing NFT transaction: {e}")
    
    def _process_marketplace_operations(self, transactions):
        """Process batch of marketplace operations"""
        market = self.contracts["market"]
        
        for tx in transactions:
            sender_id = tx.sender_id
            
            if sender_id not in self.accounts:
                self.logger.error(f"Account for {sender_id} not found")
                continue
                
            address = self.accounts[sender_id]["address"]
            private_key = self.accounts[sender_id]["private_key"]
            
            try:
                # Build transaction
                nonce = self._get_next_nonce(address)
                params = tx.params
                
                if tx.function_name == "listNFTForSale":
                    # First approve NFT transfer to market
                    nft = self.contracts["nft"]
                    
                    approval_tx = nft.functions.approve(
                        market.address,
                        params["nftId"]
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': 100000,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': nonce,
                    })
                    
                    # Sign and send approval
                    signed_approval = self.w3.eth.account.sign_transaction(approval_tx, private_key)
                    approval_hash = self.w3.eth.send_raw_transaction(signed_approval.rawTransaction)
                    
                    # Wait for approval transaction
                    self.w3.eth.wait_for_transaction_receipt(approval_hash)
                    
                    # Increment nonce for next transaction
                    nonce = self._get_next_nonce(address)
                    
                    # Now list for sale
                    tx_obj = market.functions.listNFTForSale(
                        params["nftId"],
                        params["price"]
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': self.gas_limit,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': nonce,
                    })
                elif tx.function_name == "listNFTWithDynamicPricing":
                    # First approve NFT transfer to market
                    nft = self.contracts["nft"]
                    
                    approval_tx = nft.functions.approve(
                        market.address,
                        params["nftId"]
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': 100000,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': nonce,
                    })
                    
                    # Sign and send approval
                    signed_approval = self.w3.eth.account.sign_transaction(approval_tx, private_key)
                    approval_hash = self.w3.eth.send_raw_transaction(signed_approval.rawTransaction)
                    
                    # Wait for approval transaction
                    self.w3.eth.wait_for_transaction_receipt(approval_hash)
                    
                    # Increment nonce for next transaction
                    nonce = self._get_next_nonce(address)
                    
                    # Now list with dynamic pricing
                    tx_obj = market.functions.listNFTWithDynamicPricing(
                        params["nftId"],
                        params["initialPrice"],
                        params["finalPrice"],
                        params["decayDuration"]
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': self.gas_limit,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': nonce,
                    })
                elif tx.function_name == "purchaseNFT":
                    # First approve token transfer
                    token = self.contracts["mockToken"]
                    
                    approval_tx = token.functions.approve(
                        market.address,
                        params["price"]
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': 100000,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': nonce,
                    })
                    
                    # Sign and send approval
                    signed_approval = self.w3.eth.account.sign_transaction(approval_tx, private_key)
                    approval_hash = self.w3.eth.send_raw_transaction(signed_approval.rawTransaction)
                    
                    # Wait for approval transaction
                    self.w3.eth.wait_for_transaction_receipt(approval_hash)
                    
                    # Increment nonce for next transaction
                    nonce = self._get_next_nonce(address)
                    
                    # Now purchase NFT
                    tx_obj = market.functions.purchaseNFT(params["nftId"]).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': self.gas_limit,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': nonce,
                    })
                else:
                    self.logger.error(f"Unknown marketplace function: {tx.function_name}")
                    continue
                
                # Sign and send transaction
                signed_tx = self.w3.eth.account.sign_transaction(tx_obj, private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Store transaction hash for tracking
                tx.tx_hash = tx_hash.hex()
                self.pending_transactions[tx_hash.hex()] = tx
                
                self.logger.info(f"Marketplace transaction submitted: {tx_hash.hex()}")
                self.stats['transactions_submitted'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing marketplace transaction: {e}")
    
    def _process_bundle_operations(self, transactions):
        """Process batch of bundle operations"""
        nft = self.contracts["nft"]
        
        for tx in transactions:
            sender_id = tx.sender_id
            
            if sender_id not in self.accounts:
                self.logger.error(f"Account for {sender_id} not found")
                continue
                
            address = self.accounts[sender_id]["address"]
            private_key = self.accounts[sender_id]["private_key"]
            
            try:
                # Build transaction
                nonce = self._get_next_nonce(address)
                params = tx.params
                
                if tx.function_name == "createServiceBundle":
                    tx_obj = nft.functions.createServiceBundle(
                        params["tokenIds"],
                        params["name"],
                        params["price"]
                    ).build_transaction({
                        'chainId': self.config["chain_id"],
                        'gas': self.gas_limit,
                        'maxFeePerGas': Web3.to_wei('2', 'gwei'),
                        'maxPriorityFeePerGas': Web3.to_wei('1', 'gwei'),
                        'nonce': nonce,
                    })
                else:
                    self.logger.error(f"Unknown bundle function: {tx.function_name}")
                    continue
                
                # Sign and send transaction
                signed_tx = self.w3.eth.account.sign_transaction(tx_obj, private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Store transaction hash for tracking
                tx.tx_hash = tx_hash.hex()
                self.pending_transactions[tx_hash.hex()] = tx
                
                self.logger.info(f"Bundle transaction submitted: {tx_hash.hex()}")
                self.stats['transactions_submitted'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing bundle transaction: {e}")
    
    def _check_pending_transactions(self):
        """Check status of pending transactions"""
        for tx_hash, tx_data in list(self.pending_transactions.items()):
            try:
                receipt = self.w3.eth.get_transaction_receipt(tx_hash)
                if receipt is not None:
                    # Transaction has been mined
                    if receipt['status'] == 1:
                        # Success
                        self.logger.info(f"Transaction {tx_hash} confirmed")
                        self.stats['transactions_confirmed'] += 1
                        
                        # Process on-chain results
                        self._process_transaction_receipt(tx_hash, receipt, tx_data)
                        
                        # Remove from pending
                        del self.pending_transactions[tx_hash]
                    else:
                        # Failed
                        self.logger.error(f"Transaction {tx_hash} failed")
                        
                        # Remove from pending
                        del self.pending_transactions[tx_hash]
            except Exception as e:
                self.logger.error(f"Error checking transaction {tx_hash}: {e}")
    
    def _process_transaction_receipt(self, tx_hash, receipt, tx_data):
        """Process transaction receipt to update cache and call callbacks"""
        # Update cache based on transaction type
        # Direct update to model statistics for ANY confirmed transaction
        if hasattr(self.model, 'transaction_count'):
            self.model.transaction_count += 1
            self.logger.info(f"Incremented transaction count to {self.model.transaction_count} for tx {tx_hash}")
        
        # IMPORTANT FIX: Update completed_trips_count for ANY confirmed transaction
        if hasattr(self.model, 'completed_trips_count'):
            self.model.completed_trips_count += 1
            self.logger.info(f"Incremented completed trips to {self.model.completed_trips_count}")
            
        if tx_data.tx_type == 'nft' and tx_data.function_name == 'mintServiceNFT':
            # Extract token ID from receipt
            token_id = None
            for log in receipt['logs']:
                if log['address'].lower() == self.contracts["nft"].address.lower():
                    # Try to decode event
                    try:
                        evt = self.contracts["nft"].events.ServiceTokenized().process_log(log)
                        token_id = evt['args']['tokenId']
                        break
                    except:
                        continue
            
            if token_id and 'requestId' in tx_data.params:
                # Update NFT cache
                self.state_cache['nfts'][token_id] = {
                    'owner': tx_data.sender_id,
                    'provider': tx_data.params.get('providerId'),
                    'data': {
                        'request_id': tx_data.params.get('requestId'),
                        'route_details': tx_data.params.get('routeDetails'),
                        'price': tx_data.params.get('price'),
                        'start_time': tx_data.params.get('startTime'),
                        'duration': tx_data.params.get('duration')
                    },
                    'status': 'minted',
                    'last_updated': time.time()
                }
                
                # IMPORTANT FIX: Update model statistics directly
                if hasattr(self.model, 'transaction_count'):
                    self.model.transaction_count += 1
                    self.logger.info(f"Incremented transaction count to {self.model.transaction_count}")
                    
                # Check if we should update completed trips
                if hasattr(self.model, 'completed_trips_count'):
                    self.model.completed_trips_count += 1
                    self.logger.info(f"Incremented completed trips to {self.model.completed_trips_count}")
        
        elif tx_data.tx_type == 'bundle' and tx_data.function_name == 'createServiceBundle':
            # Extract bundle ID from receipt
            bundle_id = None
            for log in receipt['logs']:
                if log['address'].lower() == self.contracts["nft"].address.lower():
                    # Try to decode event
                    try:
                        evt = self.contracts["nft"].events.BundleCreated().process_log(log)
                        bundle_id = evt['args']['bundleId']
                        break
                    except:
                        continue
            
            if bundle_id:
                # Update bundle cache
                self.state_cache['bundles'] = self.state_cache.get('bundles', {})
                if 'tokenIds' in tx_data.params:
                    self.state_cache['bundles'][bundle_id] = {
                        'owner': tx_data.sender_id,
                        'components': tx_data.params.get('tokenIds', []),
                        'name': tx_data.params.get('name'),
                        'price': Web3.from_wei(tx_data.params.get('price', 0), 'ether'),
                        'status': 'created',
                        'last_updated': time.time()
                    }
                    
                # IMPORTANT FIX: Update transaction count
                if hasattr(self.model, 'transaction_count'):
                    self.model.transaction_count += 1
                    self.logger.info(f"Incremented transaction count to {self.model.transaction_count}")
        
        elif tx_data.tx_type == 'registration' and tx_data.function_name == 'addCommuter':
            if receipt['status'] == 1:  # Transaction succeeded
                sender_id = tx_data.sender_id
                
                # Update registration status in cache
                if hasattr(self, 'pending_registrations') and sender_id in self.pending_registrations:
                    self.pending_registrations[sender_id]['status'] = 'confirmed'
                    
                if sender_id in self.state_cache['commuters']:
                    self.state_cache['commuters'][sender_id]['registered'] = True
                    
                self.logger.info(f"Commuter {sender_id} registration confirmed on blockchain")
        
        elif tx_data.tx_type == 'marketplace' and tx_data.function_name in ['listNFTForSale', 'listNFTWithDynamicPricing']:
            # IMPORTANT FIX: Update marketplace listings count for statistical tracking
            if hasattr(self.model, 'active_listings_count'):
                self.model.active_listings_count += 1
                self.logger.info(f"Incremented active listings to {self.model.active_listings_count}")
                
            # Update the token status in marketplace cache
            if 'nftId' in tx_data.params:
                token_id = tx_data.params['nftId']
                if token_id in self.state_cache['marketplace']:
                    self.state_cache['marketplace'][token_id]['status'] = 'listed'
        
        elif tx_data.tx_type == 'marketplace' and tx_data.function_name == 'purchaseNFT':
            # IMPORTANT FIX: Update transaction count
            if hasattr(self.model, 'transaction_count'):
                self.model.transaction_count += 1
                self.logger.info(f"Incremented transaction count to {self.model.transaction_count}")
                
            # Update marketplace listings count
            if 'nftId' in tx_data.params and hasattr(self.model, 'active_listings_count'):
                token_id = tx_data.params['nftId']
                if token_id in self.state_cache['marketplace']:
                    self.state_cache['marketplace'][token_id]['status'] = 'sold'
                    # Decrement active listings when sold
                    self.model.active_listings_count = max(0, self.model.active_listings_count - 1)
                    self.logger.info(f"Decremented active listings to {self.model.active_listings_count}")
                    
                # Update NFT ownership
                if token_id in self.state_cache['nfts']:
                    self.state_cache['nfts'][token_id]['owner'] = tx_data.sender_id
        
        elif tx_data.tx_type == 'request' and tx_data.function_name == 'createTravelRequest':
            # Get request ID with proper error handling
            request_id = None
            if hasattr(tx_data, 'params'):
                request_id = tx_data.params.get('requestId')
                
                # Try alternative keys if requestId is not found
                if request_id is None and 'request_id' in tx_data.params:
                    request_id = tx_data.params['request_id']
            
            # Add debugging
            self.logger.info(f"Processing confirmed request transaction, request_id: {request_id}")
            self.logger.info(f"Request IDs in cache: {list(self.state_cache['requests'].keys())}")
            
            # Update status with more robust checking
            if request_id is not None:
                # Try string conversion for more robust matching
                str_request_id = str(request_id)
                
                # Check both forms of the ID
                if request_id in self.state_cache['requests']:
                    self.state_cache['requests'][request_id]['status'] = 'active'
                    self.state_cache['requests'][request_id]['blockchain_status'] = 'confirmed'
                    self.logger.info(f"Updated request {request_id} status to active")
                elif str_request_id in self.state_cache['requests']:
                    self.state_cache['requests'][str_request_id]['status'] = 'active'
                    self.state_cache['requests'][str_request_id]['blockchain_status'] = 'confirmed'
                    self.logger.info(f"Updated request {str_request_id} status to active")
                else:
                    self.logger.warning(f"Request ID {request_id} not found in state cache")
        # Ensure marketplace transaction history is updated
        if hasattr(self.model, 'marketplace') and hasattr(self.model.marketplace, 'transaction_history'):
            # Only add transaction if not already present (avoid duplicates)
            existing_tx_hashes = [tx.get('tx_hash') for tx in self.model.marketplace.transaction_history if 'tx_hash' in tx]
            if tx_hash not in existing_tx_hashes:
                # Create simplified transaction record
                new_tx = {
                    'tx_hash': tx_hash,
                    'time': self.model.schedule.time,
                    'status': 'confirmed',
                    'price': tx_data.params.get('price', 0) if hasattr(tx_data, 'params') else 0,
                    'sender_id': tx_data.sender_id
                }
                
                # For NFT transactions, add NFT ID
                if tx_data.tx_type == 'nft' or tx_data.tx_type == 'marketplace':
                    if hasattr(tx_data, 'params') and 'nftId' in tx_data.params:
                        new_tx['nft_id'] = tx_data.params['nftId']
                
                # Add to history
                self.model.marketplace.transaction_history.append(new_tx)
                self.logger.info(f"Added transaction {tx_hash} to marketplace history")

        # Call callback if provided
        if tx_data.callback:
            try:
                tx_data.callback(receipt, tx_data)
            except Exception as e:
                self.logger.error(f"Error in transaction callback: {e}")

    # ================ Blockchain state caching ================
    
    def update_cache(self):
        """Update local cache of blockchain state"""
        # Update expired entries
        now = time.time()
        for cache_type, cache_data in self.state_cache.items():
            if cache_type == 'last_updated':
                continue
                
            for key, entry in list(cache_data.items()):
                if isinstance(entry, dict) and 'last_updated' in entry:
                    if now - entry['last_updated'] > self.cache_ttl:
                        # Entry expired, remove it
                        del cache_data[key]
        # Count confirmed transactions to update model statistics
        if hasattr(self.model, 'transaction_count') and hasattr(self, 'stats'):
            # If model shows 0 but we have confirmed transactions, update the model
            if self.model.transaction_count == 0 and self.stats['transactions_confirmed'] > 0:
                self.model.transaction_count = self.stats['transactions_confirmed']
                self.logger.info(f"Syncing transaction count to {self.model.transaction_count} from confirmed transactions")

        # Count trips to update model statistics
        if hasattr(self.model, 'completed_trips_count'):
            # Count requests that have been confirmed
            confirmed_requests = sum(1 for req in self.state_cache.get('requests', {}).values() 
                                if isinstance(req, dict) and req.get('blockchain_status') == 'confirmed')
            
            # If model shows 0 but we have confirmed requests, update the model
            if self.model.completed_trips_count == 0 and confirmed_requests > 0:
                self.model.completed_trips_count = confirmed_requests
                self.logger.info(f"Syncing completed trips to {self.model.completed_trips_count} from confirmed requests")
        
        # Update marketplace listings with time-based price decay
        active_listings = 0
        for nft_id, listing in self.state_cache.get('marketplace', {}).items():
            if listing.get('status') == 'listed':
                active_listings += 1
                
                if listing.get('dynamic_pricing', False):
                    current_time = time.time()
                    listing_time = listing.get('listing_time', current_time)
                    decay_duration = listing.get('decay_duration', 3600)
                    initial_price = listing.get('initial_price')
                    final_price = listing.get('final_price')
                    
                    # Update current price
                    if current_time >= listing_time + decay_duration:
                        listing['current_price'] = final_price
                    else:
                        elapsed = current_time - listing_time
                        decay_factor = elapsed / decay_duration
                        price_diff = initial_price - final_price
                        listing['current_price'] = initial_price - (price_diff * decay_factor)
                        
                    listing['last_updated'] = current_time
        
        # IMPORTANT FIX: Update model stats directly from cache
        if hasattr(self.model, 'active_listings_count'):
            self.model.active_listings_count = active_listings
            
        # Update transaction count from completed transactions
        if hasattr(self.model, 'transaction_count') and self.model.transaction_count == 0:
            # Count various transaction types
            nft_count = len([nft for nft_id, nft in self.state_cache.get('nfts', {}).items() 
                        if nft.get('status') in ['minted', 'used']])
            market_count = len([tx for tx in self.pending_transactions.values() 
                            if tx.tx_type == 'marketplace' and tx.function_name == 'purchaseNFT'
                            and hasattr(tx, 'tx_hash')])
            
            total_transactions = nft_count + market_count
            if total_transactions > 0:
                self.model.transaction_count = total_transactions
                self.logger.info(f"Updated transaction count to {self.model.transaction_count} from cache")
    
    def _periodic_cache_update(self):
        """Background thread for periodic cache updates"""
        while self.running:
            self.update_cache()
            self._check_pending_transactions()
            
            # Sleep between updates
            time.sleep(5)
    
    def get_cached_state(self, query_type, params):
        """
        Retrieve data from cache when possible instead of blockchain
        
        Args:
            query_type: Type of query ('commuter', 'provider', 'nft', etc.)
            params: Query parameters
            
        Returns:
            Cached data or None if not in cache
        """
        if query_type == 'commuter' and 'commuter_id' in params:
            commuter_id = params['commuter_id']
            if commuter_id in self.state_cache['commuters']:
                self.stats['cache_hits'] += 1
                return self.state_cache['commuters'][commuter_id]
        elif query_type == 'provider' and 'provider_id' in params:
            provider_id = params['provider_id']
            if provider_id in self.state_cache['providers']:
                self.stats['cache_hits'] += 1
                return self.state_cache['providers'][provider_id]
        elif query_type == 'nft' and 'token_id' in params:
            token_id = params['token_id']
            if token_id in self.state_cache['nfts']:
                self.stats['cache_hits'] += 1
                return self.state_cache['nfts'][token_id]
        elif query_type == 'marketplace' and 'token_id' in params:
            token_id = params['token_id']
            if token_id in self.state_cache['marketplace']:
                self.stats['cache_hits'] += 1
                return self.state_cache['marketplace'][token_id]
        
        self.stats['cache_misses'] += 1
        return None
    
    def _get_cached_result(self, cache_type, cache_key):
        """Get cached result if available and not expired"""
        cache_dict = self.state_cache.get('cached_results', {}).get(cache_type, {})
        result = cache_dict.get(cache_key)
        
        if result and 'expires_at' in result and result['expires_at'] > time.time():
            return result['data']
            
        return None
    
    def _cache_search_results(self, cache_type, cache_key, data):
        """Cache search results with expiration"""
        if 'cached_results' not in self.state_cache:
            self.state_cache['cached_results'] = {}
            
        if cache_type not in self.state_cache['cached_results']:
            self.state_cache['cached_results'][cache_type] = {}
            
        self.state_cache['cached_results'][cache_type][cache_key] = {
            'data': data,
            'expires_at': time.time() + 30  # Cache search results for 30 seconds
        }
    
    def _get_cached_nft_info(self, token_id):
        """Get NFT info from cache or from blockchain"""
        if token_id in self.state_cache['nfts']:
            return self.state_cache['nfts'][token_id]
            
        # Fetch from blockchain
        return self._fetch_nft_info(token_id)
    
    def _fetch_nft_info(self, token_id):
        """Fetch NFT info from blockchain and cache it"""
        try:
            nft = self.contracts["nft"]
            nft_data = nft.functions.getServiceNFT(token_id).call()
            
            # Parse and cache the results
            nft_info = {
                'token_id': nft_data[0],
                'request_id': nft_data[1],
                'provider_id': nft_data[2],
                'route_details': nft_data[3],
                'price': nft_data[4],
                'start_time': nft_data[5],
                'duration': nft_data[6],
                'original_provider': nft_data[7],
                'owner_id': nft.functions.ownerOf(token_id).call()
            }
            
            # Cache the results
            self.state_cache['nfts'][token_id] = nft_info
            
            return nft_info
        except Exception as e:
            self.logger.error(f"Error fetching NFT info: {e}")
            return None
    
    # ================ Utility methods ================
    
    def _calculate_route(self, origin, destination):
        """
        Calculate route between two points
        
        This is a simplified version for demonstration. In a real implementation,
        you would use your existing route calculation from ABM.
        """
        # Calculate direct distance
        distance = ((destination[0] - origin[0])**2 + (destination[1] - origin[1])**2)**0.5
        
        # Create a very simple route - just straight line with midpoint
        midpoint = [(origin[0] + destination[0]) / 2, (origin[1] + destination[1]) / 2]
        route = [origin, midpoint, destination]
        
         # Estimate travel time in steps (adjust based on your ABM logic)
        # Example: 1 unit of distance = 0.1 steps (1 minute)
        travel_time_steps = int(distance * 0.1)  
        
        return json.dumps({
            'origin': origin,
            'destination': destination,
            'distance': distance,
            'route': route,
            'estimated_time': travel_time_steps
        })

    def _get_next_nonce(self, address):
        """Get next transaction nonce for an address"""
        # Always get fresh nonce from network in local development
        try:
            nonce = self.w3.eth.get_transaction_count(address, 'pending')
            # Update our map with this nonce
            current_tracked = self.tx_nonce_map.get(address, 0)
            # Use whichever is higher to avoid "nonce too low" errors
            self.tx_nonce_map[address] = max(nonce, current_tracked)
            return self.tx_nonce_map[address]
        except Exception as e:
            self.logger.error(f"Error getting nonce: {e}")
            # Fallback to stored nonce if available
            return self.tx_nonce_map.get(address, 0)

    def _estimate_gas_price(self, priority='medium'):
        """
        Estimate the gas price based on network conditions and priority
        
        Args:
            priority: Transaction priority ('low', 'medium', 'high')
            
        Returns:
            int: Estimated gas price in wei
        """
        # In production, you might query the network for gas estimates
        # For simulation, we use fixed prices based on priority
        multipliers = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.5
        }
        
        base_price = Web3.to_wei('2', 'gwei')
        return int(base_price * multipliers.get(priority, 1.0))

    def _calculate_time_decay(self, initial_value, final_value, start_time, end_time, current_time):
        """
        Calculate time-decay value between two points based on current time
        Used for implementing time-sensitive pricing for mobility NFTs
        
        Args:
            initial_value: Starting value
            final_value: Final value after decay
            start_time: When decay begins
            end_time: When decay ends
            current_time: Current time to calculate value for
            
        Returns:
            float: Current value after time decay
        """
        if current_time <= start_time:
            return initial_value
        elif current_time >= end_time:
            return final_value
        else:
            # Linear decay
            progress = (current_time - start_time) / (end_time - start_time)
            return initial_value - (initial_value - final_value) * progress

    def _verify_signature(self, message, signature, expected_signer):
        """
        Verify a signature matches the expected signer
        
        Args:
            message: Original message that was signed
            signature: Signature to verify
            expected_signer: Address that should have signed the message
            
        Returns:
            bool: Whether signature is valid
        """
        try:
            # For simulation purposes, we use a simplified verification
            # In production, you should use proper cryptographic verification
            if signature.startswith("0x"):
                parts = signature[2:].split('_')
                if len(parts) >= 2:
                    return parts[0] == str(expected_signer)
            
            return False
        except Exception as e:
            self.logger.error(f"Error verifying signature: {e}")
            return False

    def _interpolate_route_points(self, route, num_points=10):
        """
        Generate additional points along a route for smoother visualization
        
        Args:
            route: List of route points [[x1,y1], [x2,y2], ...]
            num_points: Number of points to generate between each pair of points
            
        Returns:
            list: Route with interpolated points
        """
        if len(route) < 2:
            return route
            
        interpolated_route = [route[0]]
        
        for i in range(len(route) - 1):
            start = route[i]
            end = route[i+1]
            
            for j in range(1, num_points + 1):
                t = j / (num_points + 1)
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                interpolated_route.append([x, y])
                
            interpolated_route.append(end)
        
        return interpolated_route

    def get_stats(self):
        """
        Get performance statistics for analysis
        
        Returns:
            dict: Current statistics
        """
        return {
            'transactions_submitted': self.stats['transactions_submitted'],
            'transactions_confirmed': self.stats['transactions_confirmed'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'batch_operations': self.stats['batch_operations'],
            'pending_transactions': len(self.pending_transactions),
            'queue_size': len(self.tx_queue),
            'cache_size': {
                'commuters': len(self.state_cache['commuters']),
                'providers': len(self.state_cache['providers']),
                'requests': len(self.state_cache['requests']),
                'nfts': len(self.state_cache['nfts']),
                'marketplace': len(self.state_cache.get('marketplace', {}))
            }
        }

    def reset_cache(self, cache_type=None):
        """
        Reset cache data to free memory or refresh state
        
        Args:
            cache_type: Specific cache to reset, or None for all
            
        Returns:
            bool: Success status
        """
        try:
            if cache_type and cache_type in self.state_cache:
                self.state_cache[cache_type] = {}
                self.logger.info(f"Cache reset for {cache_type}")
                return True
            elif not cache_type:
                for key in self.state_cache:
                    if key != 'last_updated':
                        self.state_cache[key] = {}
                self.logger.info("All caches reset")
                return True
            else:
                self.logger.error(f"Invalid cache type: {cache_type}")
                return False
        except Exception as e:
            self.logger.error(f"Error resetting cache: {e}")
            return False

    def get_transaction_status(self, tx_hash):
        """
        Get status of a transaction for monitoring and debugging
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            dict: Transaction status and details
        """
        try:
            if tx_hash in self.pending_transactions:
                tx_data = self.pending_transactions[tx_hash]
                return {
                    'status': 'pending',
                    'sender_id': tx_data.sender_id,
                    'function': tx_data.function_name,
                    'created_at': tx_data.created_at,
                    'elapsed': time.time() - tx_data.created_at
                }
                
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            if receipt:
                status = 'confirmed' if receipt['status'] == 1 else 'failed'
                return {
                    'status': status,
                    'block_number': receipt['blockNumber'],
                    'gas_used': receipt['gasUsed'],
                    'timestamp': self.w3.eth.get_block(receipt['blockNumber'])['timestamp']
                }
            else:
                # Transaction not found in receipts, check if pending
                tx = self.w3.eth.get_transaction(tx_hash)
                if tx:
                    return {'status': 'pending', 'nonce': tx['nonce']}
                else:
                    return {'status': 'not_found'}
        except Exception as e:
            self.logger.error(f"Error getting transaction status: {e}")
            return {'status': 'error', 'message': str(e)}

    def estimate_transaction_costs(self, tx_type, params=None):
        """
        Estimate transaction costs for various operations
        
        Args:
            tx_type: Type of transaction ('registration', 'request', 'nft', etc.)
            params: Optional parameters affecting gas usage
            
        Returns:
            dict: Estimated gas costs and ETH value
        """
        gas_estimates = {
            'registration': 150000,
            'request': 250000,
            'offer': 180000,
            'nft_mint': 300000,
            'marketplace_list': 120000,
            'marketplace_buy': 220000,
            'bundle_create': 350000
        }
        
        base_gas = gas_estimates.get(tx_type, 100000)
        
        # Adjust based on params if needed
        if params:
            if 'data_size' in params:
                # More data increases gas cost
                base_gas += params['data_size'] * 100
                
            if 'complexity' in params:
                # Complex operations may use more gas
                base_gas += params['complexity'] * 50000
        
        gas_price = self._estimate_gas_price()
        eth_cost = Web3.from_wei(gas_price * base_gas, 'ether')
        
        return {
            'estimated_gas': base_gas,
            'gas_price': gas_price,
            'eth_cost': eth_cost,
            'total_wei': gas_price * base_gas
        }