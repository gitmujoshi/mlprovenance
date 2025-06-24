# Security Best Practices

This document outlines security best practices for the ML Provenance system, particularly regarding private key management and sensitive data handling.

## 🔐 Private Key Management

### ❌ Never Do This

```json
{
  "ethereum": {
    "private_key": "5d8c0114a01fb0f3efe566cfbdd6dd5963321a980938ce8df602431ff7952584"
  }
}
```

**Why this is dangerous:**
- Private keys are committed to version control
- Anyone with repository access can steal funds
- Keys persist in git history even after removal
- No access control or audit trail

### ✅ Do This Instead

#### Option 1: Environment Variables (Recommended)

```bash
# Set environment variables
export ETH_PRIVATE_KEY=your_ethereum_private_key_here
export BTC_PRIVATE_KEY=your_bitcoin_private_key_here

# Run your application
python src/ml_provenance/training/train.py
```

#### Option 2: .env File

```bash
# Copy the example file
cp env.example .env

# Edit .env with your actual private keys
nano .env
```

```bash
# .env file contents
ETH_PRIVATE_KEY=your_ethereum_private_key_here
BTC_PRIVATE_KEY=your_bitcoin_private_key_here
```

#### Option 3: Key Management Service (Production)

```python
import boto3
from ml_provenance.provenance.blockchain import BlockchainManager

# Use AWS Secrets Manager or similar
secrets_client = boto3.client('secrets-manager')
response = secrets_client.get_secret_value(SecretId='ml-provenance-keys')
private_key = response['SecretString']

config = {
    "blockchain": {
        "ethereum": {
            "private_key": private_key
        }
    }
}
```

## 🛡️ Configuration Security

### Configuration File Structure

```json
{
  "_comment": "SECURITY: Private keys should be set via environment variables (ETH_PRIVATE_KEY, BTC_PRIVATE_KEY) rather than in this file. Never commit private keys to version control.",
  "blockchain": {
    "ethereum": {
      "enabled": true,
      "rpc_url": "http://127.0.0.1:8545",
      "private_key": null,  // Will use ETH_PRIVATE_KEY environment variable
      "contract_address": null
    }
  }
}
```

### Environment Variable Support

The system automatically checks for these environment variables:

- `ETH_PRIVATE_KEY` - Ethereum private key (without 0x prefix)
- `BTC_PRIVATE_KEY` - Bitcoin private key (WIF format)
- `ETH_RPC_URL` - Override Ethereum RPC URL
- `BTC_RPC_URL` - Override Bitcoin RPC URL
- `IPFS_URL` - Override IPFS daemon URL

## 🔒 Development vs Production

### Development Environment

```bash
# Use testnet keys only
export ETH_PRIVATE_KEY=testnet_private_key_here
export BTC_PRIVATE_KEY=testnet_private_key_here

# Use local blockchain nodes
export ETH_RPC_URL=http://127.0.0.1:8545
export BTC_RPC_URL=http://localhost:8332
```

### Production Environment

```bash
# Use mainnet keys with proper security
export ETH_PRIVATE_KEY=mainnet_private_key_here
export BTC_PRIVATE_KEY=mainnet_private_key_here

# Use secure RPC endpoints
export ETH_RPC_URL=https://mainnet.infura.io/v3/YOUR_PROJECT_ID
export BTC_RPC_URL=https://your-bitcoin-node.com
```

## 🚨 Security Checklist

### Before Deployment

- [ ] Private keys are NOT in configuration files
- [ ] Environment variables are properly set
- [ ] .env file is in .gitignore
- [ ] Using testnet keys for development
- [ ] RPC endpoints are secure (HTTPS)
- [ ] Access logs are enabled
- [ ] Key rotation schedule is established

### Runtime Security

- [ ] Private keys are loaded from environment variables
- [ ] No private keys are logged or printed
- [ ] Network connections use TLS/SSL
- [ ] Error messages don't leak sensitive data
- [ ] Access controls are in place

### Monitoring

- [ ] Monitor for unauthorized access attempts
- [ ] Log all blockchain transactions
- [ ] Set up alerts for unusual activity
- [ ] Regular security audits
- [ ] Key usage monitoring

## 🔍 Security Testing

### Test Private Key Handling

```python
import os
from ml_provenance.provenance.blockchain import BlockchainManager

# Test that private keys are loaded from environment
os.environ['ETH_PRIVATE_KEY'] = 'test_key_123'
config = {"blockchain": {"ethereum": {"private_key": None}}}

manager = BlockchainManager(config)
# Verify private key is loaded from environment
assert manager.interfaces['ethereum'].private_key == 'test_key_123'
```

### Test Configuration Validation

```python
# Test that hardcoded private keys are rejected
config_with_hardcoded_key = {
    "blockchain": {
        "ethereum": {
            "private_key": "hardcoded_key_here"
        }
    }
}

# This should raise a security warning
import warnings
with warnings.catch_warnings(record=True) as w:
    manager = BlockchainManager(config_with_hardcoded_key)
    assert len(w) > 0  # Should have security warning
```

## 🆘 Incident Response

### If Private Keys Are Compromised

1. **Immediate Actions:**
   - Revoke compromised keys immediately
   - Transfer funds to new addresses
   - Rotate all related keys
   - Audit access logs

2. **Investigation:**
   - Determine how keys were exposed
   - Check git history for accidental commits
   - Review access logs
   - Identify affected systems

3. **Recovery:**
   - Generate new private keys
   - Update all configurations
   - Deploy updated systems
   - Monitor for suspicious activity

4. **Prevention:**
   - Implement additional security measures
   - Update security policies
   - Conduct security training
   - Review and improve processes

## 📚 Additional Resources

- [OWASP Security Guidelines](https://owasp.org/)
- [Ethereum Security Best Practices](https://ethereum.org/en/developers/docs/security/)
- [Bitcoin Security Guidelines](https://bitcoin.org/en/secure-your-wallet)
- [Environment Variable Security](https://12factor.net/config)

---

**Remember: Security is everyone's responsibility. When in doubt, err on the side of caution.** 