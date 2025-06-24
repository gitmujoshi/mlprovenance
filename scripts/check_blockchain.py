#!/usr/bin/env python3
"""
Simple blockchain status checker using only built-in modules.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def test_ethereum_rpc():
    """Test Ethereum RPC connection using curl."""
    print("=== Testing Ethereum RPC Connection ===")
    
    try:
        # Use curl to test RPC
        result = subprocess.run([
            'curl', '-X', 'POST', '-H', 'Content-Type: application/json',
            '--data', '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}',
            'http://127.0.0.1:8545'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            try:
                response = json.loads(result.stdout)
                block_number = int(response.get("result", "0x0"), 16)
                print(f"✅ Connected to Ethereum RPC")
                print(f"✅ Current block number: {block_number}")
                return True
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON response: {result.stdout}")
                return False
        else:
            print(f"❌ RPC request failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to connect to Ethereum RPC: {e}")
        return False

def check_ethereum_accounts():
    """Check Ethereum accounts using curl."""
    print("\n=== Checking Ethereum Accounts ===")
    
    try:
        # Get accounts
        result = subprocess.run([
            'curl', '-X', 'POST', '-H', 'Content-Type: application/json',
            '--data', '{"jsonrpc":"2.0","method":"eth_accounts","params":[],"id":1}',
            'http://127.0.0.1:8545'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            try:
                response = json.loads(result.stdout)
                accounts = response.get("result", [])
                print(f"✅ Found {len(accounts)} account(s)")
                
                for i, account in enumerate(accounts):
                    print(f"  Account {i+1}: {account}")
                    
                    # Get balance
                    balance_result = subprocess.run([
                        'curl', '-X', 'POST', '-H', 'Content-Type: application/json',
                        '--data', f'{{"jsonrpc":"2.0","method":"eth_getBalance","params":["{account}","latest"],"id":1}}',
                        'http://127.0.0.1:8545'
                    ], capture_output=True, text=True, timeout=10)
                    
                    if balance_result.returncode == 0:
                        try:
                            balance_response = json.loads(balance_result.stdout)
                            balance_hex = balance_response.get("result", "0x0")
                            balance_wei = int(balance_hex, 16)
                            balance_eth = balance_wei / (10**18)
                            print(f"    Balance: {balance_eth:.6f} ETH")
                        except (json.JSONDecodeError, ValueError) as e:
                            print(f"    Balance: Error reading balance - {e}")
                
                return len(accounts) > 0
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON response: {result.stdout}")
                return False
        else:
            print(f"❌ Failed to get accounts: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking accounts: {e}")
        return False

def check_blockchain_reports():
    """Check all blockchain reports."""
    print("\n=== Checking Blockchain Reports ===")
    
    provenance_dir = Path("artifacts/provenance")
    if not provenance_dir.exists():
        print("❌ No provenance directory found")
        return False
    
    # Find all blockchain reports
    blockchain_reports = list(provenance_dir.glob("*/blockchain_report.json"))
    
    if not blockchain_reports:
        print("❌ No blockchain reports found")
        return False
    
    print(f"📊 Found {len(blockchain_reports)} blockchain report(s)")
    
    # Check the latest report
    latest_report = max(blockchain_reports, key=lambda x: x.stat().st_mtime)
    print(f"📁 Latest report: {latest_report}")
    
    try:
        with open(latest_report, 'r') as f:
            report = json.load(f)
        
        print(f"⏰ Timestamp: {report.get('timestamp', 'N/A')}")
        
        # Check stored hashes
        stored_hashes = report.get('stored_hashes', {})
        print(f"\n📊 Stored Hashes:")
        for stage, data in stored_hashes.items():
            print(f"  {stage}:")
            print(f"    Merkle Root: {data.get('merkle_root_hash', 'N/A')[:16]}...")
            tx_ids = data.get('transaction_ids', {})
            print(f"    Transaction IDs: {tx_ids}")
            if tx_ids:
                print(f"    ✅ Transactions found")
            else:
                print(f"    ❌ No transactions found")
            print(f"    Timestamp: {data.get('timestamp', 'N/A')}")
        
        # Check verification results
        verification = report.get('verification_results', {})
        print(f"\n🔍 Verification Results:")
        chain_integrity = verification.get('chain_integrity', False)
        print(f"  Chain Integrity: {'✅' if chain_integrity else '❌'}")
        
        # Check Ethereum config
        eth_config = report.get('blockchain_config', {}).get('ethereum', {})
        print(f"\n⚡ Ethereum Configuration:")
        print(f"  Enabled: {'✅' if eth_config.get('enabled') else '❌'}")
        print(f"  RPC URL: {eth_config.get('rpc_url', 'N/A')}")
        private_key_set = eth_config.get('private_key') is not None
        print(f"  Private Key Set: {'✅' if private_key_set else '❌'}")
        
        # Summary
        print(f"\n📋 Summary:")
        has_transactions = any(
            bool(data.get('transaction_ids', {})) 
            for data in stored_hashes.values()
        )
        print(f"  Has Transactions: {'✅' if has_transactions else '❌'}")
        print(f"  Chain Integrity: {'✅' if chain_integrity else '❌'}")
        print(f"  Ethereum Enabled: {'✅' if eth_config.get('enabled') else '❌'}")
        print(f"  Private Key Set: {'✅' if private_key_set else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading blockchain report: {e}")
        return False

def main():
    """Main verification function."""
    print("🔍 Blockchain Status Checker")
    print("=" * 60)
    
    # Test Ethereum RPC
    rpc_success = test_ethereum_rpc()
    
    # Check accounts
    accounts_success = check_ethereum_accounts()
    
    # Check blockchain reports
    reports_success = check_blockchain_reports()
    
    print("\n" + "=" * 60)
    print("📋 FINAL SUMMARY:")
    print(f"  Ethereum RPC: {'✅' if rpc_success else '❌'}")
    print(f"  Ethereum Accounts: {'✅' if accounts_success else '❌'}")
    print(f"  Blockchain Reports: {'✅' if reports_success else '❌'}")
    
    if rpc_success and accounts_success and reports_success:
        print("\n🎉 Blockchain verification completed successfully!")
        print("💡 To verify transactions, run training with ETH_PRIVATE_KEY set")
        print("💡 Example: ETH_PRIVATE_KEY=your_private_key python src/ml_provenance/training/train.py")
    else:
        print("\n⚠️  Some issues detected. Check the output above.")
    
    return 0 if rpc_success and accounts_success and reports_success else 1

if __name__ == "__main__":
    sys.exit(main()) 