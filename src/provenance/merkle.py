import json
import hashlib
from typing import Dict, Any

class MLProvenanceMerkleTree:
    """Merkle tree implementation for ML provenance."""
    
    def __init__(self):
        self.nodes = {}
        self.root = None
        
    def add_node(self, component: str, data: Dict[str, Any]) -> None:
        """Add a node to the Merkle tree."""
        # Generate hash for the component
        component_hash = self._hash_component(component, data)
        
        # Create node
        node = {
            "component": component,
            "hash": component_hash,
            "timestamp": data.get("timestamp", ""),
            "children": []
        }
        
        # Add to nodes
        self.nodes[component] = node
        
        # Update root if this is the first node
        if not self.root:
            self.root = component
            
    def _hash_component(self, component: str, data: Dict[str, Any]) -> str:
        """Generate hash for a component."""
        # Create a deterministic string representation
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
        
    def verify_component(self, component: str, data: Dict[str, Any]) -> bool:
        """Verify a component's hash."""
        if component not in self.nodes:
            return False
            
        # Recompute hash
        computed_hash = self._hash_component(component, data)
        
        # Compare with stored hash
        return computed_hash == self.nodes[component]["hash"]
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary representation."""
        if not self.root:
            return {}
            
        def build_tree(component: str) -> Dict[str, Any]:
            node = self.nodes[component]
            return {
                "component": node["component"],
                "hash": node["hash"],
                "timestamp": node["timestamp"],
                "children": [build_tree(child) for child in node["children"]]
            }
            
        return build_tree(self.root)
        
    def from_dict(self, tree_dict: Dict[str, Any]) -> None:
        """Build tree from dictionary representation."""
        def add_node(node_dict: Dict[str, Any]) -> None:
            component = node_dict["component"]
            self.nodes[component] = {
                "component": component,
                "hash": node_dict["hash"],
                "timestamp": node_dict["timestamp"],
                "children": []
            }
            
            for child in node_dict.get("children", []):
                self.nodes[component]["children"].append(child["component"])
                add_node(child)
                
        if tree_dict:
            self.root = tree_dict["component"]
            add_node(tree_dict) 