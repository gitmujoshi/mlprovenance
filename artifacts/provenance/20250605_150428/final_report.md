# Training Summary

**Epochs:** 5
**Batch Size:** 32
**Validation Split:** 0.2
**Final Accuracy:** 0.9765999913215637
**Final Loss:** 0.0755651518702507

## Training Logs
| Epoch | Accuracy | Loss | Val Accuracy | Val Loss |
|-------|----------|------|--------------|----------|
| 1 | 0.902916669845581 | 0.3349398076534271 | 0.9554166793823242 | 0.1579045057296753 |
| 2 | 0.9523749947547913 | 0.1621086150407791 | 0.9649166464805603 | 0.11569308489561081 |
| 3 | 0.964020848274231 | 0.11961820721626282 | 0.9664999842643738 | 0.10442883521318436 |
| 4 | 0.9708124995231628 | 0.09704503417015076 | 0.9725833535194397 | 0.09380321949720383 |
| 5 | 0.9744583368301392 | 0.08216430991888046 | 0.9750000238418579 | 0.08312591165304184 |

# Provenance Details

## Data Provenance
- Train Hash: `a32bf55afb4a5c9cd4224cab3fb024d518fc9e9254ca071befcf5a828eff8c73`
- Test Hash: `66a1790b97c8e5364c6b703f03902cb548dcb6c22113b17ca209b037fdaf5549`

## Model Provenance
- Architecture Hash: `530f7474f834a993a15920e4e7811f97ad1311ab0485519af458a926ec51108e`
- Weights Hash: `0884fbd09512964dd856e8df04782da6c8ec0d62c1cf5ad84a0a8e10fe493f2b`

## Training Provenance
- Training Hash: `8f4ac4fdc990854356d0e04c009b7c2b459b205dca8473c97a6112b8cd048228`

## Overall Hashes
- train_data: `a32bf55afb4a5c9cd4224cab3fb024d518fc9e9254ca071befcf5a828eff8c73`
- test_data: `66a1790b97c8e5364c6b703f03902cb548dcb6c22113b17ca209b037fdaf5549`
- model_architecture: `530f7474f834a993a15920e4e7811f97ad1311ab0485519af458a926ec51108e`
- model_weights: `0884fbd09512964dd856e8df04782da6c8ec0d62c1cf5ad84a0a8e10fe493f2b`
- training: `8f4ac4fdc990854356d0e04c009b7c2b459b205dca8473c97a6112b8cd048228`
- overall: `422b590a325e2fdab0c510e8e22b91d75fe9c6353ef9f8b36f00c21328c36a40`

# Merkle Tree & Verification

**Merkle Root Hash:** `422b590a325e2fdab0c510e8e22b91d75fe9c6353ef9f8b36f00c21328c36a40`

## Merkle Proofs
### Proof for Data Provenance:
```
[
  {
    "position": "right",
    "hash": "3234d2252a7801ab91f5843afc1adbe1175c4f9c526e526911d3048bff85f14b"
  },
  {
    "position": "right",
    "hash": "cf136ceee80b4e6d5b996a0525fec39c82473f96d3a0d5f476346c89116c803e"
  }
]
```
### Proof for Model Provenance:
```
[
  {
    "position": "left",
    "hash": "5bea1508746d2c53345d02005c4f704353275e648173e405adc5cdbcd4bc0917"
  },
  {
    "position": "right",
    "hash": "cf136ceee80b4e6d5b996a0525fec39c82473f96d3a0d5f476346c89116c803e"
  }
]
```
### Proof for Training Provenance:
```
[
  {
    "position": "left",
    "hash": "492e012b0a2bbaff8108d538b6fdc721d40fe7d45cd9c81d4fe8fbb7757df3b4"
  }
]
```

# Verification Results

## Overall Status: ✅ SUCCESS

### Data Verification
```
{
  "train": true,
  "test": true,
  "timestamp": true,
  "train_hash": true,
  "test_hash": true,
  "merkle_verification": true
}
```
### Model Verification
```
{
  "model_hash_match": true,
  "weights_changed": true,
  "model_exists": true,
  "architecture_verification": {
    "layer_count_match": true,
    "parameter_count_match": true,
    "optimizer_match": true
  },
  "merkle_verification": true
}
```
### Training Verification
```
{
  "test_accuracy_present": true,
  "test_loss_present": true,
  "privacy_metrics_present": true,
  "training_hash_present": true,
  "merkle_verification": true
}
```
### Hash Verification
```
{
  "model_architecture_hash_match": true,
  "model_weights_changed": true,
  "training_hash_match": true,
  "overall_hash_present": true
}
```