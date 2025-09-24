"""
Tests for the FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock
import json

from src.api.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint returns basic information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "0.1.0"

def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data
    assert "workflows_available" in data

def test_list_workflows_endpoint():
    """Test the workflows listing endpoint."""
    response = client.get("/workflows")
    assert response.status_code == 200
    data = response.json()
    assert "workflows" in data
    assert "total_count" in data
    assert "timestamp" in data
    assert isinstance(data["workflows"], list)

def test_workflow_schema_endpoint():
    """Test getting workflow schema."""
    response = client.get("/workflows/medical_timeline/schema")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "description" in data
    assert "input_schema" in data
    assert "output_schema" in data

def test_workflow_schema_not_found():
    """Test getting schema for non-existent workflow."""
    response = client.get("/workflows/nonexistent/schema")
    assert response.status_code == 404

def test_execute_workflow_invalid_request():
    """Test workflow execution with invalid request."""
    # Missing required documents field
    invalid_request = {
        "inputs": {},
        "config": {}
    }
    
    response = client.post("/workflows/medical_timeline/execute", json=invalid_request)
    assert response.status_code == 400

def test_execute_workflow_not_found():
    """Test executing non-existent workflow."""
    request_data = {
        "inputs": {
            "documents": [{"content": "test"}]
        }
    }
    
    response = client.post("/workflows/nonexistent/execute", json=request_data)
    assert response.status_code == 404

def test_execute_workflow_valid_request():
    """Test successful workflow execution."""
    request_data = {
        "inputs": {
            "documents": [
                {
                    "content": "Patient John Doe presented with chest pain on 2023-01-15",
                    "type": "emergency_record",
                    "date": "2023-01-15",
                    "source": "General Hospital"
                }
            ],
            "patient_info": {
                "name": "John Doe",
                "dob": "1980-05-12"
            },
            "case_type": "malpractice"
        },
        "config": {
            "detail_level": "high",
            "include_analysis": True
        }
    }
    
    response = client.post("/workflows/medical_timeline/execute", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "request_id" in data
    assert "workflow_name" in data
    assert data["workflow_name"] == "medical_timeline"
    assert "status" in data
    assert "outputs" in data
    assert isinstance(data["outputs"], list)

@pytest.mark.asyncio
async def test_api_startup():
    """Test that the API starts up correctly."""
    # This test ensures the startup event handler works
    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200