import hashlib
import json
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional, Union
import logging

class MerkleNode:
    """A node in the Merkle tree."""
    def __init__(self, data: Optional[Any] = None):
        self.left: Optional[MerkleNode] = None
        self.right: Optional[MerkleNode] = None
        self.data = data
        self.hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate the hash of the node's data."""
        if self.data is None:
            return None
        
        if isinstance(self.data, (dict, list)):
            # Convert numpy arrays and bytes in lists to base64 strings
            if isinstance(self.data, list):
                data = [self._encode_data(x) for x in self.data]
            else:
                data = {k: self._encode_data(v) for k, v in self.data.items()}
            return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        elif isinstance(self.data, (np.ndarray, tf.Tensor)):
            if isinstance(self.data, tf.Tensor):
                data = self.data.numpy().tobytes()
            else:
                data = self.data.tobytes()
            return hashlib.sha256(data).hexdigest()
        else:
            return hashlib.sha256(str(self.data).encode()).hexdigest()
    
    def _encode_data(self, data: Any) -> Any:
        """Encode data for hashing."""
        if isinstance(data, (np.ndarray, tf.Tensor)):
            if isinstance(data, tf.Tensor):
                data = data.numpy()
            return data.tobytes().hex()
        return data

class MerkleTree:
    """A Merkle tree implementation for ML provenance tracking."""
    def __init__(self):
        self.root: Optional[MerkleNode] = None
        self.logger = logging.getLogger(__name__)
    
    def build_tree(self, data_list: List[Any]) -> None:
        """Build a Merkle tree from a list of data items."""
        if not data_list:
            return
        
        # Create leaf nodes
        nodes = [MerkleNode(data) for data in data_list]
        
        # Build tree levels
        while len(nodes) > 1:
            new_level = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else None
                
                # Create parent node
                parent = MerkleNode()
                parent.left = left
                parent.right = right
                
                # Calculate parent hash
                if right:
                    parent.hash = self._combine_hashes(left.hash, right.hash)
                else:
                    parent.hash = left.hash
                
                new_level.append(parent)
            nodes = new_level
        
        self.root = nodes[0]
        self.logger.info(f"Built Merkle tree with root hash: {self.root.hash}")
    
    def _combine_hashes(self, left_hash: str, right_hash: str) -> str:
        """Combine two hashes to create a parent hash."""
        return hashlib.sha256((left_hash + right_hash).encode()).hexdigest()
    
    def get_root_hash(self) -> Optional[str]:
        """Get the root hash of the tree."""
        return self.root.hash if self.root else None
    
    def verify_data(self, data: Any, proof: List[Dict[str, str]]) -> bool:
        """Verify if data exists in the tree using a Merkle proof."""
        if not self.root:
            return False
        
        # Calculate hash of the data
        node = MerkleNode(data)
        current_hash = node.hash
        
        # Verify the proof
        for step in proof:
            if step['position'] == 'left':
                current_hash = self._combine_hashes(step['hash'], current_hash)
            else:
                current_hash = self._combine_hashes(current_hash, step['hash'])
        
        return current_hash == self.root.hash
    
    def generate_proof(self, data: Any) -> Optional[List[Dict[str, str]]]:
        """Generate a Merkle proof for the given data."""
        if not self.root:
            return None
        
        target_hash = MerkleNode(data).hash
        proof = []
        
        def find_proof(node: MerkleNode, target: str) -> bool:
            if not node:
                return False
            
            if node.hash == target:
                return True
            
            if node.left and find_proof(node.left, target):
                if node.right:
                    proof.append({
                        'position': 'right',
                        'hash': node.right.hash
                    })
                return True
            
            if node.right and find_proof(node.right, target):
                if node.left:
                    proof.append({
                        'position': 'left',
                        'hash': node.left.hash
                    })
                return True
            
            return False
        
        if find_proof(self.root, target_hash):
            return proof
        return None

class MLProvenanceMerkleTree:
    """A specialized Merkle tree for ML provenance tracking."""
    def __init__(self):
        self.tree = MerkleTree()
        self.logger = logging.getLogger(__name__)
    
    def track_training_run(self, 
                          data_provenance: Dict[str, Any],
                          model_provenance: Dict[str, Any],
                          training_provenance: Dict[str, Any]) -> str:
        """Track a complete training run using a Merkle tree."""
        # Create leaf nodes for each component
        leaves = [
            {
                'type': 'data',
                'content': data_provenance
            },
            {
                'type': 'model',
                'content': model_provenance
            },
            {
                'type': 'training',
                'content': training_provenance
            }
        ]
        
        # Build the tree
        self.tree.build_tree(leaves)
        root_hash = self.tree.get_root_hash()
        
        self.logger.info(f"Tracked training run with root hash: {root_hash}")
        return root_hash
    
    def verify_component(self, 
                        component_type: str,
                        component_data: Dict[str, Any]) -> bool:
        """Verify a specific component of the training run."""
        component = {
            'type': component_type,
            'content': component_data
        }
        
        proof = self.tree.generate_proof(component)
        if not proof:
            self.logger.warning(f"No proof found for component: {component_type}")
            return False
        
        is_valid = self.tree.verify_data(component, proof)
        self.logger.info(f"Component {component_type} verification: {'success' if is_valid else 'failed'}")
        return is_valid
    
    def get_provenance_proof(self, 
                           component_type: str,
                           component_data: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        """Get a Merkle proof for a specific component."""
        component = {
            'type': component_type,
            'content': component_data
        }
        
        proof = self.tree.generate_proof(component)
        if proof:
            self.logger.info(f"Generated proof for component: {component_type}")
        else:
            self.logger.warning(f"No proof found for component: {component_type}")
        
        return proof 