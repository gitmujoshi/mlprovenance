"""Tests for report generation functionality."""

import os
import json
import pytest
from pathlib import Path
from datetime import datetime

from ml_provenance.provenance.generate_final_report import generate_final_report
from ml_provenance.provenance.verifier import ProvenanceVerifier

@pytest.fixture
def test_model_path(tmp_path):
    """Create a test model file."""
    model_path = tmp_path / "test_model.pt"
    model_path.touch()
    return str(model_path)

@pytest.fixture
def test_data_path(tmp_path):
    """Create a test data file."""
    data_path = tmp_path / "test_data.pt"
    data_path.touch()
    return str(data_path)

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

def test_generate_final_report_creates_directory(test_model_path, test_data_path, test_training_config, tmp_path):
    """Test that the report generation creates the output directory."""
    output_dir = str(tmp_path / "reports")
    generate_final_report(test_model_path, test_data_path, test_training_config, output_dir)
    assert os.path.exists(output_dir)

def test_generate_final_report_creates_file(test_model_path, test_data_path, test_training_config, tmp_path):
    """Test that the report generation creates a report file."""
    output_dir = str(tmp_path / "reports")
    report_path = generate_final_report(test_model_path, test_data_path, test_training_config, output_dir)
    assert os.path.exists(report_path)
    assert report_path.endswith(".json")

def test_report_content_structure(test_model_path, test_data_path, test_training_config, tmp_path):
    """Test that the generated report has the correct structure."""
    output_dir = str(tmp_path / "reports")
    report_path = generate_final_report(test_model_path, test_data_path, test_training_config, output_dir)
    
    with open(report_path, "r") as f:
        report = json.load(f)
    
    assert "timestamp" in report
    assert "model_provenance" in report
    assert "data_provenance" in report
    assert "training_config" in report
    assert "verification_status" in report
    
    # Verify training config matches input
    assert report["training_config"] == test_training_config

def test_report_timestamp_format(test_model_path, test_data_path, test_training_config, tmp_path):
    """Test that the report timestamp is in ISO format."""
    output_dir = str(tmp_path / "reports")
    report_path = generate_final_report(test_model_path, test_data_path, test_training_config, output_dir)
    
    with open(report_path, "r") as f:
        report = json.load(f)
    
    # Try to parse the timestamp
    datetime.fromisoformat(report["timestamp"]) 