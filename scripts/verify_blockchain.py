#!/usr/bin/env python3
"""
Simple script to verify blockchain functionality.
"""

import json
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_provenance.provenance.blockchain import EthereumInterface, BlockchainManager

def test_ethereum_connection():
    """Test Ethereum connection and transaction sending."""
    print("=== Testing Ethereum Connection ===")
    
    # Test basic connection
    try:
        eth_interface = EthereumInterface("http://127.0.0.1:8545")
        print(f"✅ Connected to Ethereum RPC: {eth_interface.rpc_url}")
        
        if eth_interface.web3:
            block_number = eth_interface.web3.eth.block_number
            print(f"✅ Current block number: {block_number}")
        else:
            print("❌ Web3 not available")
            return False
            
    except Exception as e:
        print(f"❌ Failed to connect to Ethereum: {e}")
        return False
    
    # Test transaction sending (simulation)
    try:
        test_data = {
            "merkle_root_hash": "test_hash_123",
            "timestamp": "2025-06-23T15:00:00",
            "type": "test"
        }
        
        print("\n=== Testing Transaction Sending ===")
        tx_id = eth_interface.store_hash("test_hash_123", test_data)
        print(f"✅ Transaction ID: {tx_id}")
        
        # Test verification
        is_valid = eth_interface.verify_hash("test_hash_123", tx_id)
        print(f"✅ Verification: {'SUCCESS' if is_valid else 'FAILED'}")
        
        # Get transaction info
        tx_info = eth_interface.get_transaction_info(tx_id)
        print(f"✅ Transaction Info: {tx_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to send transaction: {e}")
        return False

def check_latest_blockchain_report():
    """Check the latest blockchain report."""
    print("\n=== Latest Blockchain Report ===")
    
    try:
        # Find the latest blockchain report
        provenance_dirs = sorted(Path('artifacts/provenance').glob('*'), key=lambda x: x.stat().st_mtime, reverse=True)
        latest_dir = provenance_dirs[0] if provenance_dirs else None
        
        if latest_dir and (latest_dir / 'blockchain_report.json').exists():
            with open(latest_dir / 'blockchain_report.json', 'r') as f:
                report = json.load(f)
            
            print(f"📁 Report Location: {latest_dir / 'blockchain_report.json'}")
            print(f"⏰ Timestamp: {report.get('timestamp', 'N/A')}")
            
            stored_hashes = report.get('stored_hashes', {})
            print(f"\n📊 Stored Hashes:")
            for stage, data in stored_hashes.items():
                print(f"  {stage}:")
                print(f"    Merkle Root: {data.get('merkle_root_hash', 'N/A')[:16]}...")
                print(f"    Transaction IDs: {data.get('transaction_ids', {})}")
                print(f"    Timestamp: {data.get('timestamp', 'N/A')}")
            
            verification = report.get('verification_results', {})
            print(f"\n🔍 Verification Results:")
            print(f"  Chain Integrity: {'✅' if verification.get('chain_integrity') else '❌'}")
            
            # Check Ethereum config
            eth_config = report.get('blockchain_config', {}).get('ethereum', {})
            print(f"\n⚡ Ethereum Configuration:")
            print(f"  Enabled: {'✅' if eth_config.get('enabled') else '❌'}")
            print(f"  RPC URL: {eth_config.get('rpc_url', 'N/A')}")
            print(f"  Private Key Set: {'✅' if eth_config.get('private_key') is not None else '❌'}")
            
            return True
        else:
            print("❌ No blockchain report found")
            return False
            
    except Exception as e:
        print(f"❌ Error reading blockchain report: {e}")
        return False

def main():
    """Main verification function."""
    print("🔍 Blockchain Verification Tool")
    print("=" * 50)
    
    # Test Ethereum connection
    eth_success = test_ethereum_connection()
    
    # Check latest report
    report_success = check_latest_blockchain_report()
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY:")
    print(f"  Ethereum Connection: {'✅' if eth_success else '❌'}")
    print(f"  Report Available: {'✅' if report_success else '❌'}")
    
    if eth_success and report_success:
        print("\n🎉 Blockchain verification completed successfully!")
    else:
        print("\n⚠️  Some issues detected. Check the output above.")
    
    return 0 if eth_success and report_success else 1

if __name__ == "__main__":
    sys.exit(main()) 