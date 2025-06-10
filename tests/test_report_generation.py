"""Tests for report generation functionality."""

import os
import json
import pytest
import torch
from pathlib import Path
from datetime import datetime
from ml_provenance.models.mnist_model import MNISTModel

from ml_provenance.provenance.generate_final_report import generate_final_report

@pytest.fixture
def test_model_path(tmp_path):
    """Create a test model file."""
    model_path = tmp_path / "test_model.pt"
    model = MNISTModel()
    torch.save(model.state_dict(), model_path)
    return str(model_path)

@pytest.fixture
def test_provenance_dir(tmp_path):
    """Create a test provenance directory with a dummy provenance.json file."""
    provenance_dir = tmp_path / "provenance_data"
    provenance_dir.mkdir()
    # Minimal valid provenance.json
    provenance_json = {
        "version": "2024-01-01T00:00:00",
        "data_provenance": {},
        "model_provenance": {},
        "training_provenance": {}
    }
    with open(provenance_dir / "provenance.json", "w") as f:
        json.dump(provenance_json, f)
    return str(provenance_dir)

@pytest.fixture
def test_training_config():
    """Create a test training configuration."""
    return {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "privacy_epsilon": 1.0,
        "privacy_delta": 1e-5
    }

def test_generate_final_report_creates_directory(test_model_path, test_provenance_dir, test_training_config, tmp_path):
    """Test that the report generation creates the output directory."""
    output_dir = str(tmp_path / "reports")
    generate_final_report(test_model_path, test_provenance_dir, test_training_config, output_dir)
    assert os.path.exists(output_dir)

def test_generate_final_report_creates_file(test_model_path, test_provenance_dir, test_training_config, tmp_path):
    """Test that the report generation creates a report file."""
    output_dir = str(tmp_path / "reports")
    report_path = generate_final_report(test_model_path, test_provenance_dir, test_training_config, output_dir)
    assert os.path.exists(report_path)
    assert report_path.endswith(".json")

def test_report_content_structure(test_model_path, test_provenance_dir, test_training_config, tmp_path):
    """Test that the generated report has the correct structure."""
    output_dir = str(tmp_path / "reports")
    report_path = generate_final_report(test_model_path, test_provenance_dir, test_training_config, output_dir)
    
    with open(report_path, "r") as f:
        report = json.load(f)
    
    assert "timestamp" in report
    assert "verification_report" in report
    assert "training_config" in report
    assert "verification_status" in report
    
    # Verify training config matches input
    assert report["training_config"] == test_training_config

def test_report_timestamp_format(test_model_path, test_provenance_dir, test_training_config, tmp_path):
    """Test that the report timestamp is in ISO format."""
    output_dir = str(tmp_path / "reports")
    report_path = generate_final_report(test_model_path, test_provenance_dir, test_training_config, output_dir)
    
    with open(report_path, "r") as f:
        report = json.load(f)
    
    # Try to parse the timestamp
    datetime.fromisoformat(report["timestamp"]) 