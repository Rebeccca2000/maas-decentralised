#!/usr/bin/env python3
"""
Integration test script for blockchain_interface.py

This script performs basic operations against a local blockchain to validate 
that the BlockchainInterface works outside of the mock environment.

Requirements:
- Running local blockchain (Ganache or Hardhat) on default port (8545)
- Deployment of MaaS contracts (deployment-info.json should be present)
- Python dependencies: web3, json, time

Usage:
python blockchain_integration_test.py
"""

import sys
import os
import json
import time
import random
from pathlib import Path

# Add parent directory to path if needed
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
if parent_dir not in sys.path:
    sys.path.append(str(parent_dir))

# Import the BlockchainInterface
from blockchain_interface import BlockchainInterface

def main():
    print("Starting blockchain interface integration test...")
    
    # Initialize the interface with default config
    interface = BlockchainInterface(async_mode=False)  # Use synchronous mode for testing
    
    # Test connection
    if not interface.w3.is_connected():
        print("ERROR: Could not connect to local blockchain. Make sure it's running.")
        return False
    
    print(f"Connected to blockchain at {interface.w3.provider.endpoint_uri}")
    print(f"Chain ID: {interface.w3.eth.chain_id}")
    
    # Get available accounts on the blockchain
    accounts = interface.w3.eth.accounts
    if not accounts:
        print("ERROR: No accounts available on the blockchain.")
        return False
    
    print(f"Found {len(accounts)} accounts on the blockchain")
    
    # Test 1: Create account
    print("\n=== Test 1: Create Account ===")
    commuter_id = random.randint(1000, 9999)  # Random ID for testing
    address = interface.create_account(commuter_id, "commuter")
    if not address:
        print("ERROR: Failed to create account")
        return False
    
    print(f"Created commuter account with ID {commuter_id} at address: {address}")
    
    # Test 2: Register a commuter
    print("\n=== Test 2: Register Commuter ===")
    # Create a mock commuter agent
    class MockCommuterAgent:
        def __init__(self, unique_id):
            self.unique_id = unique_id
            self.location = (10, 20)
            self.income_level = "middle"
            self.age = 35
            self.has_disability = False
            self.tech_access = True
            self.health_status = "good"
            self.payment_scheme = "PAYG"
    
    commuter_agent = MockCommuterAgent(commuter_id)
    success, address = interface.register_commuter(commuter_agent)
    if not success:
        print("ERROR: Failed to register commuter")
        return False
    
    print(f"Registered commuter with ID {commuter_id}")
    
    # Test 3: Register a provider
    print("\n=== Test 3: Register Provider ===")
    # Create a mock provider agent
    class MockProviderAgent:
        def __init__(self, unique_id):
            self.unique_id = unique_id
            self.company_name = "TestProvider"
            self.base_price = 10
            self.capacity = 50
            self.service_center = [50, 50]
            self.servicing_area = 25
            self.response_time = 10
            self.reliability = 70
            self.quality_score = 70
            self.certifications = []
            self.is_verified = False
            self.is_active = True
    
    provider_id = random.randint(1000, 9999)  # Random ID for testing
    provider_agent = MockProviderAgent(provider_id)
    success, address = interface.register_provider(provider_agent)
    if not success:
        print("ERROR: Failed to register provider")
        return False
    
    print(f"Registered provider with ID {provider_id} at address: {address}")
    
    # Test 4: Create a travel request
    print("\n=== Test 4: Create Travel Request ===")
    request_batch = [{
        'commuter_id': commuter_id,
        'origin': [10, 20],
        'destination': [30, 40],
        'start_time': int(time.time()) + 3600,  # 1 hour from now
        'travel_purpose': 'work',
        'flexible_time': 'medium'
    }]
    
    results = interface.process_requests_batch(request_batch)
    if not results or not results[0][0]:
        print("ERROR: Failed to create travel request")
        return False
    
    request_id = results[0][1]
    print(f"Created travel request with ID: {request_id}")
    
    # Test 5: Create an NFT
    print("\n=== Test 5: Create NFT ===")
    service_details = {
        'request_id': request_id,
        'price': 10,
        'start_time': int(time.time()) + 3600,
        'duration': 1800,  # 30 minutes
        'route_details': {
            'route': [[10, 20], [20, 30], [30, 40]],
            'distance': 30,
            'estimated_time': 1800
        }
    }
    
    success, token_id = interface.create_nft(service_details, provider_id, commuter_id)
    if not success:
        print("ERROR: Failed to create NFT")
        return False
    
    print(f"Created NFT with token ID: {token_id}")
    
    # Test 6: List NFT for sale
    print("\n=== Test 6: List NFT for Sale ===")
    price = 15  # Higher than original price
    success = interface.list_nft_for_sale(token_id, price)
    if not success:
        print("ERROR: Failed to list NFT for sale")
        return False
    
    print(f"Listed NFT {token_id} for sale at price: {price}")
    
    # Test 7: Search NFT market
    print("\n=== Test 7: Search NFT Market ===")
    search_params = {
        'min_price': 5,
        'max_price': 20,
        'min_departure': int(time.time()),
        'max_departure': int(time.time()) + 86400,  # Within 24 hours
    }
    
    results = interface.search_nft_market(search_params)
    print(f"Found {len(results)} NFTs matching search criteria")
    for i, result in enumerate(results):
        print(f"  {i+1}. Token ID: {result.get('token_id')}, Price: {result.get('price')}")
    
    # Test 8: Get current blockchain stats
    print("\n=== Test 8: Get Blockchain Stats ===")
    stats = interface.get_stats()
    print("Blockchain interface statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ All integration tests completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Integration tests failed!")
        sys.exit(1)
    sys.exit(0)