import hashlib
import json
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union
import logging
from .hash_config import HashFactory

logger = logging.getLogger(__name__)

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

    def get_hash_structure(self):
        """Return just the hash structure of the tree without leaf node data."""
        def serialize_hash_structure(node):
            if node is None:
                return None
            return {
                'hash': node.hash,
                'left': serialize_hash_structure(node.left),
                'right': serialize_hash_structure(node.right)
            }
        return serialize_hash_structure(self.root)

    def dump_hash_structure(self, filepath=None):
        """Dump the hash structure to a file or return as string."""
        hash_structure = self.get_hash_structure()
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(hash_structure, f, indent=2)
            self.logger.info(f"Hash structure dumped to {filepath}")
        return json.dumps(hash_structure, indent=2)

class MLProvenanceMerkleTree:
    """Merkle tree implementation for ML provenance verification."""
    
    def __init__(self):
        """Initialize an empty Merkle tree."""
        self.root = None
        self.nodes = {}  # Dictionary to store all nodes by their type
        self.epoch_nodes = {}  # Store epoch-specific nodes
        self.architecture_node = None  # Store architecture node separately
        logger.info("Initialized MLProvenanceMerkleTree")
    
    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """
        Compute hash of data using configured hash function.
        
        Args:
            data: Data to hash
            
        Returns:
            Hex digest of the hash
        """
        def _serialize_data(obj):
            try:
                if isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().numpy().tolist()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {str(k): _serialize_data(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [_serialize_data(x) for x in obj]
                elif hasattr(obj, '__dict__'):
                    return _serialize_data(obj.__dict__)
                elif isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                else:
                    return str(obj)
            except Exception as e:
                logger.warning(f"Failed to serialize object of type {type(obj)}: {str(e)}")
                return str(obj)

        try:
            # Get hash function from factory
            hash_func = HashFactory.get_hash_function()
            
            # Serialize data to bytes, handling tensors
            serialized_data = _serialize_data(data)
            data_bytes = json.dumps(serialized_data, sort_keys=True).encode()
            
            # Compute hash
            return hash_func(data_bytes)
        except Exception as e:
            logger.error(f"Error computing hash: {str(e)}")
            raise
    
    def add_node(self, data: Dict[str, Any], parent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a node to the Merkle tree.
        
        Args:
            data: Data to store in the node
            parent: Optional parent node
            
        Returns:
            Created node
        """
        node = {
            "data": data,
            "hash": self._compute_hash(data),
            "left": None,
            "right": None,
            "parent": parent
        }
        
        # Store node in nodes dictionary if it has a type
        if "type" in data:
            self.nodes[data["type"]] = node
        
        if parent is None:
            self.root = node
        else:
            if parent["left"] is None:
                parent["left"] = node
            else:
                parent["right"] = node
        
        logger.info(f"Added node with hash: {node['hash']}")
        return node
    
    def add_epoch_node(self, epoch: int, data: Dict[str, Any]):
        """
        Add a node for a specific training epoch.
        
        Args:
            epoch: Epoch number
            data: Epoch data
        """
        node = self.add_node(data)
        self.epoch_nodes[epoch] = node
        logger.info(f"Added epoch {epoch} node with hash: {node['hash']}")
    
    def add_architecture_node(self, data: Dict[str, Any]):
        """
        Add a node for model architecture.
        
        Args:
            data: Architecture data
        """
        self.architecture_node = self.add_node(data)
        logger.info(f"Added architecture node with hash: {self.architecture_node['hash']}")
    
    def verify_component(self, component: str, data: Dict[str, Any]) -> bool:
        """
        Verify a component's data against its stored hash.
        
        Args:
            component: Component name
            data: Component data to verify
            
        Returns:
            True if verification passes, False otherwise
        """
        if component == "architecture" and self.architecture_node:
            stored_hash = self.architecture_node["hash"]
        elif component in self.epoch_nodes:
            stored_hash = self.epoch_nodes[component]["hash"]
        else:
            logger.error(f"Component {component} not found in Merkle tree")
            return False
        
        computed_hash = self._compute_hash(data)
        matches = computed_hash == stored_hash
        
        logger.info(f"Verified {component} component: {'✓' if matches else '✗'}")
        logger.info(f"Stored hash: {stored_hash}")
        logger.info(f"Computed hash: {computed_hash}")
        
        return matches
    
    def verify_model_progression(self) -> Dict[str, Any]:
        """
        Verify that model changes follow expected progression through epochs.
        
        Returns:
            Dictionary containing verification results
        """
        results = {
            "verified": True,
            "mismatched_epochs": []
        }
        
        # Sort epochs to ensure ordered verification
        epochs = sorted(self.epoch_nodes.keys())
        
        for i in range(1, len(epochs)):
            prev_epoch = epochs[i-1]
            curr_epoch = epochs[i]
            
            prev_data = self.epoch_nodes[prev_epoch]["data"]
            curr_data = self.epoch_nodes[curr_epoch]["data"]
            
            # Verify that current epoch's data is different from previous
            if prev_data == curr_data:
                results["verified"] = False
                results["mismatched_epochs"].append(curr_epoch)
                logger.warning(f"Model state unchanged between epochs {prev_epoch} and {curr_epoch}")
        
        return results
    
    def get_epoch_proof(self, epoch: int) -> Optional[List[Dict[str, Any]]]:
        """
        Get Merkle proof for a specific epoch.
        
        Args:
            epoch: Epoch number
            
        Returns:
            List of nodes in the proof path, or None if epoch not found
        """
        if epoch not in self.epoch_nodes:
            logger.error(f"Epoch {epoch} not found in Merkle tree")
            return None
        
        proof = []
        current = self.epoch_nodes[epoch]
        
        while current["parent"]:
            parent = current["parent"]
            sibling = parent["right"] if current == parent["left"] else parent["left"]
            
            if sibling:
                proof.append({
                    "hash": sibling["hash"],
                    "is_left": current == parent["right"]
                })
            
            current = parent
        
        return proof
    
    def get_tree_dict(self) -> Dict[str, Any]:
        """
        Get dictionary representation of the tree.
        
        Returns:
            Dictionary containing tree structure
        """
        def node_to_dict(node: Dict[str, Any]) -> Dict[str, Any]:
            if node is None:
                return None
            
            return {
                "hash": node["hash"],
                "data": node["data"],
                "left": node_to_dict(node["left"]),
                "right": node_to_dict(node["right"])
            }
        
        return node_to_dict(self.root)

    def build_provenance_tree(self, data_provenance: Dict[str, Any], model_provenance: Dict[str, Any], training_provenance: Dict[str, Any]) -> None:
        """
        Build the Merkle tree from provenance data.
        
        Args:
            data_provenance: Dictionary containing data provenance
            model_provenance: Dictionary containing model provenance
            training_provenance: Dictionary containing training provenance
        """
        logger.info("Building provenance Merkle tree...")
        
        # Create data node
        data_node = self.add_node({
            "type": "data",
            "data": data_provenance
        })
        
        # Create model node
        model_node = self.add_node({
            "type": "model",
            "data": model_provenance
        })
        
        # Create training node
        training_node = self.add_node({
            "type": "training",
            "data": training_provenance
        })
        
        # Create root node combining all components
        self.root = self.add_node({
            "type": "root",
            "data": {
                "data": data_node,
                "model": model_node,
                "training": training_node
            }
        })
        
        logger.info("Provenance Merkle tree built successfully")
        logger.info(f"Root hash: {self.root['hash']}")

    def get_node(self, node_type: str) -> Optional[Dict[str, Any]]:
        """
        Get a node by its type.
        
        Args:
            node_type: Type of the node to retrieve
            
        Returns:
            Node dictionary if found, None otherwise
        """
        return self.nodes.get(node_type)

    def get_root_hash(self) -> str:
        """Get the root hash of the Merkle tree."""
        if self.root is None:
            raise ValueError("Merkle tree has not been built yet")
        return self.root["hash"]

    def get_component_proof(self, component: str) -> Optional[Dict[str, Any]]:
        """
        Get the Merkle proof for a specific component.
        
        Args:
            component: Component to get proof for ("data", "model", or "training")
            
        Returns:
            Dictionary containing the proof, or None if component not found
        """
        if self.root is None:
            return None
            
        if component not in ["data", "model", "training"]:
            raise ValueError(f"Invalid component: {component}")
            
        # Find the component node
        component_node = self.nodes.get(component)
        if component_node is None:
            return None
            
        # Build the proof
        proof = {
            "component": component,
            "hash": component_node.get("hash"),
            "siblings": []
        }
        
        # Add sibling hashes
        for sibling in ["data", "model", "training"]:
            if sibling != component:
                sibling_node = self.nodes.get(sibling)
                if sibling_node is not None:
                    proof["siblings"].append({
                        "type": sibling,
                        "hash": sibling_node.get("hash")
                    })
        
        return proof

    def _dump_merkle_tree(self) -> None:
        """Dump the Merkle tree structure for debugging."""
        if self.root is None:
            logger.warning("Merkle tree is empty")
            return
            
        def _dump_node(node, level=0):
            indent = "  " * level
            if node is None:
                return f"{indent}None"
                
            node_str = f"{indent}Node:\n"
            node_str += f"{indent}  Hash: {node.get('hash')}\n"
            node_str += f"{indent}  Type: {node.get('data', {}).get('type')}\n"
            
            if node.get("left"):
                node_str += f"{indent}  Left:\n{_dump_node(node['left'], level + 2)}\n"
            if node.get("right"):
                node_str += f"{indent}  Right:\n{_dump_node(node['right'], level + 2)}\n"
                
            return node_str
            
        tree_str = _dump_node(self.root)
        logger.debug(f"Merkle Tree Structure:\n{tree_str}") 