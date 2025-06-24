#!/usr/bin/env python3
"""
Debug script to test blockchain interface initialization.
"""

import os
import json
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_provenance.provenance.blockchain import BlockchainManager, EthereumInterface

def test_blockchain_initialization():
    """Test blockchain interface initialization."""
    
    # Load the blockchain config
    config_path = Path(__file__).parent.parent / "configs" / "blockchain_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("=== Blockchain Configuration ===")
    print(f"Config loaded from: {config_path}")
    print(f"Blockchain enabled: {config.get('blockchain', {}).get('enabled', False)}")
    print(f"Ethereum enabled: {config.get('blockchain', {}).get('ethereum', {}).get('enabled', False)}")
    print(f"IPFS enabled: {config.get('blockchain', {}).get('ipfs', {}).get('enabled', False)}")
    
    # Check environment variables
    print("\n=== Environment Variables ===")
    eth_private_key = os.getenv('ETH_PRIVATE_KEY')
    print(f"ETH_PRIVATE_KEY set: {'Yes' if eth_private_key else 'No'}")
    if eth_private_key:
        print(f"ETH_PRIVATE_KEY length: {len(eth_private_key)}")
        print(f"ETH_PRIVATE_KEY prefix: {eth_private_key[:10]}...")
    
    # Test Ethereum interface creation
    print("\n=== Testing Ethereum Interface ===")
    try:
        eth_config = config.get('blockchain', {}).get('ethereum', {})
        private_key = eth_config.get('private_key')
        if private_key is None:
            private_key = os.getenv('ETH_PRIVATE_KEY')
        
        print(f"Private key from config: {eth_config.get('private_key')}")
        print(f"Private key from env: {os.getenv('ETH_PRIVATE_KEY')}")
        print(f"Final private key: {'Set' if private_key else 'None'}")
        
        eth_interface = EthereumInterface(
            rpc_url=eth_config.get('rpc_url', 'http://localhost:8545'),
            private_key=private_key,
            contract_address=eth_config.get('contract_address')
        )
        print("✅ Ethereum interface created successfully")
        print(f"Web3 available: {eth_interface.web3 is not None}")
        
    except Exception as e:
        print(f"❌ Failed to create Ethereum interface: {e}")
        import traceback
        traceback.print_exc()
    
    # Test BlockchainManager initialization
    print("\n=== Testing BlockchainManager ===")
    try:
        blockchain_manager = BlockchainManager(config)
        print(f"✅ BlockchainManager created successfully")
        print(f"Number of interfaces: {len(blockchain_manager.interfaces)}")
        print(f"Available interfaces: {list(blockchain_manager.interfaces.keys())}")
        
        # Test storing a hash
        print("\n=== Testing Hash Storage ===")
        test_hash = "test_hash_1234567890abcdef"
        test_metadata = {"test": "data"}
        
        results = blockchain_manager.store_merkle_hash(test_hash, test_metadata)
        print(f"Storage results: {results}")
        
    except Exception as e:
        print(f"❌ Failed to create BlockchainManager: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_blockchain_initialization() 