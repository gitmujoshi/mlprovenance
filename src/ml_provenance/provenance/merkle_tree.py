"""
Merkle Tree Implementation for ML Provenance Verification

This module implements a Merkle tree data structure specifically designed for
machine learning provenance verification. It provides cryptographic proof of
data integrity and enables efficient verification of large datasets.

Architecture:
├── MLProvenanceMerkleTree: Main Merkle tree implementation
├── Hash Generation: Configurable hash algorithms
├── Tree Construction: Bottom-up tree building
├── Proof Generation: Cryptographic proof creation
└── Verification: Integrity checking and validation

Key Features:
- Configurable hash algorithms (BLAKE3, SHA256, SHA512)
- Efficient tree construction for large datasets
- Cryptographic proof generation
- Fast verification algorithms
- JSON serialization for persistence
- Cross-platform compatibility

Cryptographic Properties:
- Collision resistance through cryptographic hashing
- Tamper detection through hash chaining
- Efficient verification through logarithmic proofs
- Immutable structure through hash dependencies

Use Cases:
- ML model integrity verification
- Dataset authenticity validation
- Training process audit trails
- Blockchain storage verification
- Cross-system data validation

Author: ML Provenance Team
License: MIT
"""

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
        # Get hash function from factory
        hash_func = HashFactory.get_hash_function()
        return hash_func(json.dumps(serialized_data, sort_keys=True).encode())
    
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
        # Get hash function from factory
        hash_func = HashFactory.get_hash_function()
        return hash_func((left_hash + right_hash).encode())
    
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
    """
    Merkle tree implementation for ML provenance verification.
    
    This class implements a Merkle tree data structure specifically designed
    for machine learning provenance verification. It provides cryptographic
    proof of data integrity and enables efficient verification of large
    datasets and model artifacts.
    
    Key Features:
        - Configurable hash algorithms (BLAKE3, SHA256, SHA512)
        - Efficient tree construction for large datasets
        - Cryptographic proof generation and verification
        - JSON serialization for persistence
        - Cross-platform compatibility
        
    Cryptographic Properties:
        - Collision resistance through cryptographic hashing
        - Tamper detection through hash chaining
        - Efficient verification through logarithmic proofs
        - Immutable structure through hash dependencies
        
    Use Cases:
        - ML model integrity verification
        - Dataset authenticity validation
        - Training process audit trails
        - Blockchain storage verification
        - Cross-system data validation
    """
    
    def __init__(self, hash_algorithm: str = 'blake3'):
        """
        Initialize Merkle tree with specified hash algorithm.
        
        Args:
            hash_algorithm: Hash algorithm to use ('blake3', 'sha256', 'sha512')
            
        Supported Algorithms:
            - blake3: Fast, secure hash function (recommended)
            - sha256: Standard SHA-256 hash function
            - sha512: SHA-512 hash function for higher security
        """
        self.hash_factory = HashFactory()
        self.hash_factory.set_algorithm(hash_algorithm)
        self.hash_algorithm = hash_algorithm
        
        # Tree structure
        self.leaves = []
        self.nodes = {}
        self.root_hash = None
        self.tree_height = 0
        
        logger.info(f"Merkle tree initialized with algorithm: {hash_algorithm}")
    
    def build_tree(self, data: Union[Dict[str, Any], List[Any], str, bytes]) -> str:
        """
        Build Merkle tree from input data.
        
        This method constructs a complete Merkle tree from the input data,
        creating leaf nodes from individual data elements and building
        the tree structure bottom-up through hash aggregation.
        
        Args:
            data: Input data to build tree from
                - Dict: Dictionary of key-value pairs
                - List: List of data elements
                - str: String data
                - bytes: Binary data
                
        Returns:
            Root hash of the constructed Merkle tree
            
        Tree Construction Process:
            1. Convert input data to leaf nodes
            2. Hash each leaf node
            3. Build parent nodes by hashing pairs of children
            4. Repeat until single root node is created
            5. Return root hash for verification
        """
        try:
            # Convert data to leaf nodes
            self.leaves = self._prepare_leaves(data)
            
            if not self.leaves:
                raise ValueError("No valid leaves generated from input data")
            
            # Build tree structure
            self._build_tree_structure()
            
            # Calculate tree height
            self.tree_height = self._calculate_tree_height()
            
            logger.info(f"Merkle tree built with {len(self.leaves)} leaves, "
                       f"height: {self.tree_height}, root: {self.root_hash[:16]}...")
            
            return self.root_hash
            
        except Exception as e:
            logger.error(f"Failed to build Merkle tree: {e}")
            raise
    
    def _prepare_leaves(self, data: Union[Dict[str, Any], List[Any], str, bytes]) -> List[str]:
        """
        Prepare leaf nodes from input data.
        
        This method converts the input data into a list of leaf nodes
        that will form the base of the Merkle tree. The method handles
        different data types and structures appropriately.
        
        Args:
            data: Input data to convert to leaves
            
        Returns:
            List of leaf node hashes
            
        Data Type Handling:
            - Dict: Each key-value pair becomes a leaf
            - List: Each element becomes a leaf
            - str: String is split or treated as single leaf
            - bytes: Binary data is chunked into leaves
        """
        leaves = []
        
        if isinstance(data, dict):
            # Convert dictionary to sorted key-value pairs
            for key, value in sorted(data.items()):
                leaf_data = f"{key}:{json.dumps(value, sort_keys=True)}"
                leaf_hash = self.hash_factory.hash(leaf_data.encode('utf-8'))
                leaves.append(leaf_hash)
                
        elif isinstance(data, list):
            # Convert list elements to leaves
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    item_str = json.dumps(item, sort_keys=True)
                else:
                    item_str = str(item)
                
                leaf_data = f"{i}:{item_str}"
                leaf_hash = self.hash_factory.hash(leaf_data.encode('utf-8'))
                leaves.append(leaf_hash)
                
        elif isinstance(data, str):
            # Handle string data
            leaf_hash = self.hash_factory.hash(data.encode('utf-8'))
            leaves.append(leaf_hash)
            
        elif isinstance(data, bytes):
            # Handle binary data
            leaf_hash = self.hash_factory.hash(data)
            leaves.append(leaf_hash)
            
        else:
            # Convert other types to string
            data_str = str(data)
            leaf_hash = self.hash_factory.hash(data_str.encode('utf-8'))
            leaves.append(leaf_hash)
        
        return leaves
    
    def _build_tree_structure(self) -> None:
        """
        Build the complete tree structure from leaves.
        
        This method constructs the complete Merkle tree structure by
        creating parent nodes from pairs of child nodes, working
        bottom-up until a single root node is created.
        
        Tree Building Algorithm:
            1. Start with leaf nodes at level 0
            2. For each level, pair adjacent nodes
            3. Hash each pair to create parent nodes
            4. Repeat until single root node remains
            5. Store all nodes in self.nodes for later access
        """
        if not self.leaves:
            raise ValueError("No leaves available for tree construction")
        
        # Initialize level 0 with leaves
        current_level = 0
        self.nodes[current_level] = self.leaves.copy()
        
        # Build tree levels bottom-up
        while len(self.nodes[current_level]) > 1:
            current_nodes = self.nodes[current_level]
            next_level = current_level + 1
            self.nodes[next_level] = []
            
            # Process nodes in pairs
            for i in range(0, len(current_nodes), 2):
                left_node = current_nodes[i]
                right_node = current_nodes[i + 1] if i + 1 < len(current_nodes) else left_node
                
                # Create parent node by hashing the pair
                parent_data = left_node + right_node
                parent_hash = self.hash_factory.hash(parent_data.encode('utf-8'))
                self.nodes[next_level].append(parent_hash)
            
            current_level = next_level
        
        # Set root hash
        if self.nodes[current_level]:
            self.root_hash = self.nodes[current_level][0]
        else:
            raise ValueError("Failed to construct tree root")
    
    def _calculate_tree_height(self) -> int:
        """
        Calculate the height of the constructed tree.
        
        Returns:
            Height of the Merkle tree (number of levels)
        """
        return len(self.nodes) - 1 if self.nodes else 0
    
    def generate_proof(self, leaf_index: int) -> Dict[str, Any]:
        """
        Generate cryptographic proof for a specific leaf.
        
        This method creates a cryptographic proof that a specific leaf
        is part of the Merkle tree. The proof consists of sibling hashes
        at each level needed to reconstruct the path to the root.
        
        Args:
            leaf_index: Index of the leaf to generate proof for
            
        Returns:
            Dictionary containing proof information
            
        Proof Structure:
            - leaf_index: Index of the proven leaf
            - leaf_hash: Hash of the leaf node
            - siblings: List of sibling hashes for each level
            - path: Binary path from leaf to root
            - root_hash: Root hash for verification
        """
        if not self.root_hash:
            raise ValueError("Merkle tree not built")
        
        if leaf_index < 0 or leaf_index >= len(self.leaves):
            raise ValueError(f"Invalid leaf index: {leaf_index}")
        
        try:
            proof = {
                'leaf_index': leaf_index,
                'leaf_hash': self.leaves[leaf_index],
                'siblings': [],
                'path': [],
                'root_hash': self.root_hash,
                'tree_height': self.tree_height
            }
            
            # Generate proof path
            current_index = leaf_index
            current_level = 0
            
            while current_level < self.tree_height:
                current_nodes = self.nodes[current_level]
                
                # Determine if current node is left or right child
                is_left = current_index % 2 == 0
                
                # Get sibling index
                if is_left:
                    sibling_index = current_index + 1
                    proof['path'].append(0)  # 0 for left
                else:
                    sibling_index = current_index - 1
                    proof['path'].append(1)  # 1 for right
                
                # Add sibling hash to proof
                if sibling_index < len(current_nodes):
                    proof['siblings'].append(current_nodes[sibling_index])
                else:
                    # Handle case where sibling doesn't exist (odd number of nodes)
                    proof['siblings'].append(current_nodes[current_index])
                
                # Move to parent level
                current_index = current_index // 2
                current_level += 1
            
            logger.info(f"Generated proof for leaf {leaf_index}")
            return proof
            
        except Exception as e:
            logger.error(f"Failed to generate proof for leaf {leaf_index}: {e}")
            raise
    
    def verify_proof(self, proof: Dict[str, Any], leaf_data: Union[str, bytes]) -> bool:
        """
        Verify a cryptographic proof for a leaf.
        
        This method verifies that a given leaf data is part of the Merkle
        tree by reconstructing the path to the root using the provided proof.
        
        Args:
            proof: Proof dictionary from generate_proof()
            leaf_data: Original leaf data to verify
            
        Returns:
            True if proof is valid, False otherwise
            
        Verification Process:
            1. Hash the leaf data
            2. Use proof siblings to reconstruct path to root
            3. Compare computed root with proof root
            4. Return True if they match, False otherwise
        """
        try:
            # Hash the leaf data
            if isinstance(leaf_data, str):
                leaf_hash = self.hash_factory.hash(leaf_data.encode('utf-8'))
            else:
                leaf_hash = self.hash_factory.hash(leaf_data)
            
            # Verify leaf hash matches proof
            if leaf_hash != proof['leaf_hash']:
                logger.warning("Leaf hash mismatch in proof verification")
                return False
            
            # Reconstruct path to root
            current_hash = leaf_hash
            siblings = proof['siblings']
            path = proof['path']
            
            for i, (sibling_hash, direction) in enumerate(zip(siblings, path)):
                if direction == 0:  # Left child
                    parent_data = current_hash + sibling_hash
                else:  # Right child
                    parent_data = sibling_hash + current_hash
                
                current_hash = self.hash_factory.hash(parent_data.encode('utf-8'))
            
            # Compare with root hash
            is_valid = current_hash == proof['root_hash']
            
            if is_valid:
                logger.info(f"Proof verification successful for leaf {proof['leaf_index']}")
            else:
                logger.warning(f"Proof verification failed for leaf {proof['leaf_index']}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Failed to verify proof: {e}")
            return False
    
    def get_tree_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the Merkle tree.
        
        Returns:
            Dictionary containing tree information
        """
        return {
            'root_hash': self.root_hash,
            'tree_height': self.tree_height,
            'leaf_count': len(self.leaves),
            'hash_algorithm': self.hash_algorithm,
            'total_nodes': sum(len(nodes) for nodes in self.nodes.values()),
            'levels': {level: len(nodes) for level, nodes in self.nodes.items()}
        }
    
    def save_tree(self, filepath: Union[str, Path]) -> None:
        """
        Save Merkle tree to file.
        
        This method serializes the complete Merkle tree structure to a JSON file
        for persistence and later reconstruction.
        
        Args:
            filepath: Path to save the tree file
        """
        try:
            tree_data = {
                'root_hash': self.root_hash,
                'tree_height': self.tree_height,
                'leaves': self.leaves,
                'nodes': self.nodes,
                'hash_algorithm': self.hash_algorithm,
                'metadata': {
                    'created_at': self._get_timestamp(),
                    'leaf_count': len(self.leaves),
                    'total_nodes': sum(len(nodes) for nodes in self.nodes.values())
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(tree_data, f, indent=2, default=str)
            
            logger.info(f"Merkle tree saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save Merkle tree: {e}")
            raise
    
    def load_tree(self, filepath: Union[str, Path]) -> None:
        """
        Load Merkle tree from file.
        
        This method deserializes a Merkle tree from a JSON file and
        reconstructs the complete tree structure.
        
        Args:
            filepath: Path to the tree file to load
        """
        try:
            with open(filepath, 'r') as f:
                tree_data = json.load(f)
            
            # Restore tree state
            self.root_hash = tree_data['root_hash']
            self.tree_height = tree_data['tree_height']
            self.leaves = tree_data['leaves']
            self.nodes = {int(k): v for k, v in tree_data['nodes'].items()}
            self.hash_algorithm = tree_data['hash_algorithm']
            
            # Reinitialize hash factory
            self.hash_factory.set_algorithm(self.hash_algorithm)
            
            logger.info(f"Merkle tree loaded from: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load Merkle tree: {e}")
            raise
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp string.
        
        Returns:
            ISO format timestamp string
        """
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_root_hash(self) -> Optional[str]:
        """
        Get the root hash of the Merkle tree.
        
        Returns:
            Root hash string, or None if tree not built
        """
        return self.root_hash
    
    def get_leaf_count(self) -> int:
        """
        Get the number of leaves in the tree.
        
        Returns:
            Number of leaf nodes
        """
        return len(self.leaves)
    
    def get_tree_height(self) -> int:
        """
        Get the height of the tree.
        
        Returns:
            Height of the Merkle tree
        """
        return self.tree_height
    
    def is_built(self) -> bool:
        """
        Check if the tree has been built.
        
        Returns:
            True if tree is built, False otherwise
        """
        return self.root_hash is not None

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