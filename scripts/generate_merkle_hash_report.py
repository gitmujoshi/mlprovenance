#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path
import json

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from ml_provenance.provenance.merkle_tree import MLProvenanceMerkleTree


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_latest_provenance_dir():
    provenance_base = Path("artifacts/provenance")
    if not provenance_base.exists():
        return None
    dirs = [d for d in provenance_base.glob("*") if d.is_dir() and "_" in d.name]
    if not dirs:
        return None
    return str(sorted(dirs, reverse=True)[0])

def load_provenance_data(provenance_dir):
    provenance_path = Path(provenance_dir) / "provenance.json"
    if not provenance_path.exists():
        raise FileNotFoundError(f"provenance.json not found in {provenance_dir}")
    with open(provenance_path, "r") as f:
        return json.load(f)

def build_merkle_tree_from_provenance(data):
    tree = MLProvenanceMerkleTree()
    tree.build_provenance_tree(
        data["data_provenance"],
        data["model_provenance"],
        data["training_provenance"]
    )
    # Save only the nodes, excluding leaf nodes
    nodes_only = tree.get_nodes_only()
    return nodes_only

def save_full_merkle_tree(tree, provenance_dir):
    full_tree = tree.get_full_tree()
    with open(Path(provenance_dir) / "full_merkle_tree.json", "w") as f:
        json.dump(full_tree, f, indent=2)

def format_hash_structure_md(node, level=0, node_type="Root"):
    if node is None:
        return ""
    indent = "  " * level
    md = f"{indent}- {node_type} Node: `{node['hash']}`\n"
    if node.get("left") or node.get("right"):
        if node.get("left"):
            left_type = "Data" if level == 0 else "Training" if level == 1 else "Training Data"
            md += format_hash_structure_md(node["left"], level + 1, left_type)
        if node.get("right"):
            right_type = "Model" if level == 0 else "Test Data" if level == 1 else "Final State"
            md += format_hash_structure_md(node["right"], level + 1, right_type)
    return md

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    if len(sys.argv) == 2:
        provenance_dir = sys.argv[1]
    else:
        provenance_dir = get_latest_provenance_dir()
        if provenance_dir is None:
            logger.error("No provenance directory found in artifacts/provenance")
            sys.exit(1)
        logger.info(f"Using latest provenance directory: {provenance_dir}")
    if not os.path.exists(provenance_dir):
        logger.error(f"Provenance directory not found: {provenance_dir}")
        sys.exit(1)
    try:
        logger.info(f"Loading provenance data from: {provenance_dir}")
        data = load_provenance_data(provenance_dir)
        nodes_only = build_merkle_tree_from_provenance(data)
        save_full_merkle_tree(MLProvenanceMerkleTree(), provenance_dir)
        md_report = [
            f"# Merkle Tree Hash Structure Report",
            f"**Provenance Directory:** `{provenance_dir}`",
            f"**Root Hash:** `{nodes_only['hash']}`\n",
            "## Tree Structure:",
            "The Merkle tree structure represents the hierarchical organization of the ML training run components:",
            "```",
            "Root Node",
            "├── Data Node",
            "│   ├── Training Data Node",
            "│   └── Test Data Node",
            "└── Model Node",
            "    └── Training Node",
            "        ├── Epoch Nodes",
            "        │   ├── Model State",
            "        │   ├── Metrics",
            "        │   └── Privacy Metrics",
            "        └── Final State",
            "```\n",
            "## Detailed Hash Structure:",
            format_hash_structure_md(nodes_only)
        ]
        report_path = os.path.join(provenance_dir, "merkle_hash_report.md")
        with open(report_path, "w") as f:
            f.write("\n".join(md_report))
        logger.info(f"Merkle hash structure Markdown report saved to: {report_path}")
    except Exception as e:
        logger.error(f"Error generating Merkle hash structure report: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 