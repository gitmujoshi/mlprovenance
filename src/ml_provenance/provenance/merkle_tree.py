import hashlib
import json
import numpy as np
import torch
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
        
        def _serialize_data(data):
            if isinstance(data, (np.ndarray, torch.Tensor)):
                if isinstance(data, torch.Tensor):
                    data = data.detach().cpu().numpy()
                return data.tobytes().hex()
            elif isinstance(data, dict):
                return {k: _serialize_data(v) for k, v in sorted(data.items())}
            elif isinstance(data, list):
                return [_serialize_data(x) for x in data]
            else:
                return str(data)
        
        serialized_data = _serialize_data(self.data)
        return hashlib.sha256(json.dumps(serialized_data, sort_keys=True).encode()).hexdigest()
    
    def _encode_data(self, data: Any) -> Any:
        """Encode data for hashing."""
        if isinstance(data, (np.ndarray, torch.Tensor)):
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
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

    def get_tree_dict(self):
        """Return the tree structure as a dictionary."""
        def serialize_node(node):
            if node is None:
                return None
            return {
                'hash': node.hash,
                'data': node.data if not isinstance(node.data, (torch.Tensor, np.ndarray)) else str(type(node.data)),
                'left': serialize_node(node.left),
                'right': serialize_node(node.right)
            }
        return serialize_node(self.root)

class MLProvenanceMerkleTree(MerkleTree):
    """A specialized Merkle tree for ML provenance tracking."""
    def __init__(self):
        super().__init__()
        self.nodes = {}
        self.logger = logging.getLogger(__name__)
    
    def _hash_data(self, data):
        # Local utility to ensure all data is JSON serializable
        def convert_to_serializable(obj):
            import numpy as np
            import torch
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        if isinstance(data, (np.ndarray, torch.Tensor)):
            # Convert to bytes for hashing
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            data_bytes = data.tobytes()
        elif isinstance(data, (dict, list)):
            serializable_data = convert_to_serializable(data)
            data_bytes = json.dumps(serializable_data, sort_keys=True).encode()
        else:
            data_bytes = str(data).encode()
        return hashlib.sha256(data_bytes).hexdigest()

    def add_node(self, node_id, data):
        """Add a node to the tree and rebuild the tree structure."""
        # Create the node
        hash_value = self._hash_data(data)
        node = MerkleNode(data)
        node.hash = hash_value
        
        # Store in nodes dictionary
        self.nodes[node_id] = {
            'hash': hash_value,
            'data': data,
            'node': node
        }
        
        # Rebuild tree with all nodes
        self._rebuild_tree()
        
        return hash_value
    
    def _rebuild_tree(self):
        """Rebuild the tree structure from stored nodes."""
        if not self.nodes:
            self.root = None
            return
        
        # Get all nodes in order
        node_list = []
        for node_id in sorted(self.nodes.keys()):
            node_list.append(self.nodes[node_id]['node'])
        
        # Build tree levels
        while len(node_list) > 1:
            new_level = []
            for i in range(0, len(node_list), 2):
                left = node_list[i]
                right = node_list[i + 1] if i + 1 < len(node_list) else None
                
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
            node_list = new_level
        
        self.root = node_list[0]
        self.logger.info(f"Rebuilt Merkle tree with root hash: {self.root.hash}")

    def get_proof(self, node_id):
        """Get a proof for a specific node."""
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]['node']
        proof = []
        
        def find_proof(current_node, target_node):
            if not current_node:
                return False
            
            if current_node == target_node:
                return True
            
            if current_node.left and find_proof(current_node.left, target_node):
                if current_node.right:
                    proof.append({
                        'position': 'right',
                        'hash': current_node.right.hash
                    })
                return True
            
            if current_node.right and find_proof(current_node.right, target_node):
                if current_node.left:
                    proof.append({
                        'position': 'left',
                        'hash': current_node.left.hash
                    })
                return True
            
            return False
        
        if find_proof(self.root, node):
            return {
                'node_id': node_id,
                'hash': node.hash,
                'proof': proof
            }
        return None

    def verify_proof(self, proof):
        """Verify a proof for a node."""
        if not proof or 'node_id' not in proof or 'hash' not in proof:
            return False
        
        node_id = proof['node_id']
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]['node']
        if node.hash != proof['hash']:
            return False
        
        if 'proof' not in proof:
            return True
        
        current_hash = node.hash
        for step in proof['proof']:
            if step['position'] == 'left':
                current_hash = self._combine_hashes(step['hash'], current_hash)
            else:
                current_hash = self._combine_hashes(current_hash, step['hash'])
        
        return current_hash == self.root.hash
    
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
        
        # Create leaf nodes with proper hashing
        leaf_nodes = []
        for leaf in leaves:
            node = MerkleNode(leaf)
            leaf_nodes.append(node)
        
        # Build the tree
        while len(leaf_nodes) > 1:
            new_level = []
            for i in range(0, len(leaf_nodes), 2):
                left = leaf_nodes[i]
                right = leaf_nodes[i + 1] if i + 1 < len(leaf_nodes) else None
                
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
            leaf_nodes = new_level
        
        self.root = leaf_nodes[0]
        root_hash = self.root.hash
        
        self.logger.info(f"Tracked training run with root hash: {root_hash}")
        return root_hash
    
    def verify_component(self, component_name, component_data):
        """Verify a component using the Merkle tree."""
        if not self.root:
            return False
            
        # Create a node for the component data
        component_node = {
            'type': component_name,
            'content': component_data
        }
        node = MerkleNode(component_node)
        
        # Find the component in the tree
        def find_component(node, target_type):
            if not node:
                return False
            if isinstance(node.data, dict) and node.data.get('type') == target_type:
                return node.hash == self._hash_data(component_node)
            return find_component(node.left, target_type) or find_component(node.right, target_type)
        
        return find_component(self.root, component_name)
    
    def get_provenance_proof(self, 
                           component_type: str,
                           component_data: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        """Get a Merkle proof for a specific component."""
        component = {
            'type': component_type,
            'content': component_data
        }
        
        proof = self.generate_proof(component)
        if proof:
            self.logger.info(f"Generated proof for component: {component_type}")
        else:
            self.logger.warning(f"No proof found for component: {component_type}")
        
        return proof 