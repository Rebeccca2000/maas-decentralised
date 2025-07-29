import unittest
from unittest.mock import MagicMock, patch
import json
import time
from blockchain_interface import BlockchainInterface, TransactionData

class TestBlockchainInterface(unittest.TestCase):
    
    def setUp(self):
        # Mock Web3 and related components
        self.mock_w3 = MagicMock()
        self.mock_contracts = {
            'registry': MagicMock(),
            'request': MagicMock(),
            'auction': MagicMock(),
            'nft': MagicMock(),
            'market': MagicMock(),
            'facade': MagicMock(),
            'mockToken': MagicMock()
        }
        
        # Create a test instance with mocked components
        with patch('blockchain_interface.Web3', return_value=self.mock_w3):
            with patch.object(BlockchainInterface, '_load_contracts', return_value=self.mock_contracts):
                self.interface = BlockchainInterface(async_mode=False)  # Test in synchronous mode
                self.interface.w3 = self.mock_w3
                self.interface.contracts = self.mock_contracts
    
    def test_create_account(self):
        # Mock account creation
        mock_account = MagicMock()
        mock_account.address = "0x123abc"
        mock_account.key.hex.return_value = "0xprivatekey"
        self.mock_w3.eth.account.create.return_value = mock_account
        
        # Test account creation
        address = self.interface.create_account(1, "commuter")
        
        # Verify results
        self.assertEqual(address, "0x123abc")
        self.assertIn(1, self.interface.accounts)
        self.assertEqual(self.interface.accounts[1]["address"], "0x123abc")
        self.assertEqual(self.interface.accounts[1]["type"], "commuter")
    
    def test_register_commuter(self):
        # Create mock commuter agent
        commuter = MagicMock()
        commuter.unique_id = 1
        commuter.location = (10, 20)
        commuter.income_level = "middle"
        commuter.age = 35
        commuter.has_disability = False
        commuter.tech_access = True
        commuter.health_status = "good"
        commuter.payment_scheme = "PAYG"
        
        # Setup mock account
        self.interface.accounts[1] = {
            "address": "0x123abc",
            "private_key": "0xprivatekey",
            "type": "commuter"
        }
        
        # Mock transaction methods
        self.mock_w3.eth.account.sign_transaction.return_value.rawTransaction = b'signed_tx'
        self.mock_w3.eth.send_raw_transaction.return_value = b'tx_hash'
        self.mock_w3.eth.wait_for_transaction_receipt.return_value = {'status': 1}
        
        # Test register commuter
        success, address = self.interface.register_commuter(commuter)
        
        # Verify results
        self.assertTrue(success)
        self.assertEqual(address, "0x123abc")
        self.mock_contracts['registry'].functions.addCommuter.assert_called_once()
    
    def test_batch_processing(self):
        # Create test batch of requests
        batch = [
            {
                'commuter_id': 1,
                'origin': [10, 20],
                'destination': [30, 40],
                'start_time': int(time.time()) + 3600,
                'flexible_time': 'medium'
            },
            {
                'commuter_id': 2,
                'origin': [15, 25],
                'destination': [35, 45],
                'start_time': int(time.time()) + 3600,
                'flexible_time': 'low'
            }
        ]
        
        # Setup mock accounts and cache
        self.interface.accounts[1] = {"address": "0x123abc", "private_key": "0xprivatekey", "type": "commuter"}
        self.interface.accounts[2] = {"address": "0x456def", "private_key": "0xprivatekey2", "type": "commuter"}
        self.interface.state_cache['commuters'][1] = {'data': {'commuterId': 1}}
        self.interface.state_cache['commuters'][2] = {'data': {'commuterId': 2}}
        
        # Test batch processing
        results = self.interface.process_requests_batch(batch)
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result[0] for result in results))
    
    def test_create_nft(self):
        # Setup mock account
        self.interface.accounts[1] = {"address": "0x123abc", "private_key": "0xprivatekey", "type": "commuter"}
        
        # Mock transaction methods
        self.mock_w3.eth.account.sign_transaction.return_value.rawTransaction = b'signed_tx'
        self.mock_w3.eth.send_raw_transaction.return_value = b'tx_hash'
        
        # Setup mock NFT contract address
        self.mock_contracts['nft'].address = "0xnftaddress"  # Add this line
        
        # Create a better structured log that matches what your code is looking for
        mock_log = {
            'address': "0xnftaddress",  # Must match the NFT contract address
            'topics': [b'0x123'],  # Some topic
            'data': '0x0',
            'logIndex': 0
        }
        receipt = {'status': 1, 'logs': [mock_log]}
        self.mock_w3.eth.wait_for_transaction_receipt.return_value = receipt
        
        # Mock the events better - the key issue is in this part
        event_instance = MagicMock()
        event_instance.process_log.return_value = {'args': {'tokenId': 123}}
        self.mock_contracts['nft'].events.ServiceTokenized = MagicMock(return_value=event_instance)
        
        # Test NFT creation
        service_details = {
            'request_id': 1,
            'price': 100,
            'start_time': int(time.time()) + 3600,
            'duration': 1800,
            'route_details': {'route': [[10, 20], [30, 40]]}
        }
        
        success, token_id = self.interface.create_nft(service_details, 2, 1)
        
        # Verify results
        self.assertTrue(success)
        self.assertEqual(token_id, 123)
    
    def test_list_nft_for_sale(self):
        # Setup mock account and NFT in cache
        self.interface.accounts[1] = {"address": "0x123abc", "private_key": "0xprivatekey", "type": "commuter"}
        self.interface.state_cache['nfts'][123] = {
            'owner': 1,
            'provider': 2,
            'data': {'price': 100},
            'status': 'minted'
        }
        
        # Mock transaction methods
        self.mock_w3.eth.account.sign_transaction.return_value.rawTransaction = b'signed_tx'
        self.mock_w3.eth.send_raw_transaction.return_value = b'tx_hash'
        self.mock_w3.eth.wait_for_transaction_receipt.return_value = {'status': 1}
        
        # Test listing NFT
        time_params = {
            'initial_price': 100,
            'final_price': 50,
            'decay_duration': 3600
        }
        
        success = self.interface.list_nft_for_sale(123, 100, time_params)
        
        # Verify results
        self.assertTrue(success)
        self.mock_contracts['nft'].functions.approve.assert_called_once()
        self.mock_contracts['market'].functions.listNFTWithDynamicPricing.assert_called_once()
    
    def test_search_nft_market(self):
        # Reset the mock to clear any previous calls
        self.mock_contracts['market'].functions.searchListings.reset_mock()
        
        # Prepare the search result more explicitly
        search_result = [
            [123, 100, "0x123abc"],
            [456, 200, "0x456def"]
        ]
        
        # Setup a more specific mock for searchListings with parameters
        search_listings_mock = MagicMock()
        search_listings_mock.call.return_value = search_result
        self.mock_contracts['market'].functions.searchListings.return_value = search_listings_mock
        
        # Setup NFT cache
        self.interface.state_cache['nfts'][123] = {
            'route_details': '{"origin": [10, 20], "destination": [30, 40]}',
            'start_time': int(time.time()) + 3600,
            'duration': 1800
        }
        self.interface.state_cache['nfts'][456] = {
            'route_details': '{"origin": [15, 25], "destination": [35, 45]}',
            'start_time': int(time.time()) + 7200,
            'duration': 1800
        }
        
        # Test search
        search_params = {
            'min_price': 50,
            'max_price': 150,
            'min_departure': int(time.time()),
            'max_departure': int(time.time()) + 86400,
            'origin_area': [5, 15, 20]  # x, y, radius
        }
        
        results = self.interface.search_nft_market(search_params)
        
        # Verify results
        self.assertEqual(len(results), 2)
        
        # Verify the searchListings was called with the right parameters
        self.mock_contracts['market'].functions.searchListings.assert_called_with(
            self.mock_w3.toWei('50', 'ether'),
            self.mock_w3.toWei('150', 'ether'),
            search_params['min_departure'],
            search_params['max_departure']
        )
    
    def test_purchase_nft(self):
        # Setup mock account
        self.interface.accounts[2] = {"address": "0x456def", "private_key": "0xprivatekey2", "type": "commuter"}
        
        # Setup NFT in marketplace cache
        self.interface.state_cache['marketplace'][123] = {
            'seller': 1,
            'initial_price': 100,
            'current_price': 80,
            'final_price': 50,
            'dynamic_pricing': True,
            'decay_duration': 3600,
            'listing_time': time.time() - 1800,  # Listed 30 minutes ago
            'status': 'listed'
        }
        
        # Mock transaction methods
        self.mock_w3.eth.account.sign_transaction.return_value.rawTransaction = b'signed_tx'
        self.mock_w3.eth.send_raw_transaction.return_value = b'tx_hash'
        self.mock_w3.eth.wait_for_transaction_receipt.return_value = {'status': 1}
        
        # Test purchase
        success = self.interface.purchase_nft(123, 2)
        
        # Verify results
        self.assertTrue(success)
        self.mock_contracts['mockToken'].functions.approve.assert_called_once()
        self.mock_contracts['market'].functions.purchaseNFT.assert_called_once()
        self.assertEqual(self.interface.state_cache['marketplace'][123]['status'], 'sold')
    
    def test_bundle_handling(self):
        # Setup mock accounts
        self.interface.accounts[1] = {"address": "0x123abc", "private_key": "0xprivatekey", "type": "commuter"}
        
        # Setup mock NFT contract address
        self.mock_contracts['nft'].address = "0xnftaddress"  # Add this line
        
        # Create a better structured log that matches what your code is looking for
        mock_log = {
            'address': "0xnftaddress",  # Must match the NFT contract address
            'topics': [b'0x123'],  # Some topic
            'data': '0x0',
            'logIndex': 0
        }
        receipt = {'status': 1, 'logs': [mock_log]}
        
        # Mock the bundle creation steps better
        with patch.object(self.interface, 'create_nft', return_value=(True, 123)):
            # Setup bundle details
            bundle_details = {
                'components': {
                    'segment_1': {
                        'provider_id': 2,
                        'price': 100,
                        'start_time': int(time.time()) + 3600,
                        'duration': 1800
                    }
                },
                'total_price': 100,
                'name': 'Test Bundle'
            }
            
            # Mock transaction
            self.mock_w3.eth.account.sign_transaction.return_value.rawTransaction = b'signed_tx'
            self.mock_w3.eth.send_raw_transaction.return_value = b'tx_hash'
            self.mock_w3.eth.wait_for_transaction_receipt.return_value = receipt
            
            # Mock the events better
            event_instance = MagicMock()
            event_instance.process_log.return_value = {'args': {'bundleId': 456}}
            self.mock_contracts['nft'].events.BundleCreated = MagicMock(return_value=event_instance)
            
            # Test bundle creation
            success, bundle_id = self.interface.execute_bundle_purchase(bundle_details, 1)
            
            # Verify results
            self.assertTrue(success)
            self.assertEqual(bundle_id, 456)

if __name__ == '__main__':
    unittest.main()