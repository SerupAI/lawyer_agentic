"""
Tests for workflow classes and workflow manager.
"""

import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime

from src.workflows.base import WorkflowManager, BaseWorkflow
from src.workflows.medical_timeline import MedicalTimelineGenerator
from src.models.workflow import WorkflowStatus, WorkflowOutput

class TestWorkflowManager:
    """Test cases for WorkflowManager."""
    
    def test_register_workflow(self):
        """Test workflow registration."""
        manager = WorkflowManager()
        workflow = MedicalTimelineGenerator()
        
        manager.register_workflow("test_workflow", workflow)
        
        assert "test_workflow" in manager.get_available_workflows()
        assert manager._workflows["test_workflow"] == workflow
        assert "test_workflow" in manager._metrics
    
    def test_get_workflow_schema(self):
        """Test getting workflow schema."""
        manager = WorkflowManager()
        workflow = MedicalTimelineGenerator()
        manager.register_workflow("medical_timeline", workflow)
        
        schema = manager.get_workflow_schema("medical_timeline")
        assert schema.name == "medical_timeline"
        assert schema.description == "Generate comprehensive medical timelines from case documents"
        assert "type" in schema.input_schema
        assert "properties" in schema.input_schema
    
    def test_get_workflow_schema_not_found(self):
        """Test getting schema for non-existent workflow."""
        manager = WorkflowManager()
        
        with pytest.raises(ValueError, match="Workflow 'nonexistent' not found"):
            manager.get_workflow_schema("nonexistent")
    
    @pytest.mark.asyncio
    async def test_execute_workflow_success(self):
        """Test successful workflow execution."""
        manager = WorkflowManager()
        workflow = MedicalTimelineGenerator()
        manager.register_workflow("medical_timeline", workflow)
        
        inputs = {
            "documents": [
                {
                    "content": "Patient presented with chest pain",
                    "type": "emergency_record",
                    "date": "2023-01-15"
                }
            ],
            "patient_info": {"name": "John Doe"},
            "case_type": "malpractice"
        }
        
        response = await manager.execute_workflow("medical_timeline", inputs)
        
        assert response.workflow_name == "medical_timeline"
        assert response.status == WorkflowStatus.COMPLETED
        assert len(response.outputs) > 0
        assert response.execution_time_seconds is not None
        assert response.started_at is not None
        assert response.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_execute_workflow_not_found(self):
        """Test executing non-existent workflow."""
        manager = WorkflowManager()
        
        with pytest.raises(ValueError, match="Workflow 'nonexistent' not found"):
            await manager.execute_workflow("nonexistent", {})
    
    @pytest.mark.asyncio
    async def test_execute_workflow_validation_error(self):
        """Test workflow execution with invalid inputs."""
        manager = WorkflowManager()
        workflow = MedicalTimelineGenerator()
        manager.register_workflow("medical_timeline", workflow)
        
        invalid_inputs = {}  # Missing required documents
        
        with pytest.raises(RuntimeError):
            await manager.execute_workflow("medical_timeline", invalid_inputs)

class TestMedicalTimelineGenerator:
    """Test cases for MedicalTimelineGenerator workflow."""
    
    def test_initialization(self):
        """Test workflow initialization."""
        workflow = MedicalTimelineGenerator()
        
        assert workflow.name == "medical_timeline"
        assert workflow.version == "1.0.0"
        assert workflow.description == "Generate comprehensive medical timelines from case documents"
        assert workflow.estimated_execution_time_seconds == 30
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid data."""
        workflow = MedicalTimelineGenerator()
        
        valid_inputs = {
            "documents": [
                {
                    "content": "Medical record content",
                    "type": "medical_record",
                    "date": "2023-01-15"
                }
            ],
            "patient_info": {
                "name": "John Doe",
                "dob": "1980-05-12"
            },
            "case_type": "malpractice"
        }
        
        assert workflow.validate_inputs(valid_inputs) is True
    
    def test_validate_inputs_missing_documents(self):
        """Test input validation with missing documents."""
        workflow = MedicalTimelineGenerator()
        
        invalid_inputs = {
            "patient_info": {"name": "John Doe"}
        }
        
        with pytest.raises(ValueError, match="Missing required field: 'documents'"):
            workflow.validate_inputs(invalid_inputs)
    
    def test_validate_inputs_empty_documents(self):
        """Test input validation with empty documents list."""
        workflow = MedicalTimelineGenerator()
        
        invalid_inputs = {
            "documents": []
        }
        
        with pytest.raises(ValueError, match="'documents' must be a non-empty list"):
            workflow.validate_inputs(invalid_inputs)
    
    def test_validate_inputs_invalid_document_format(self):
        """Test input validation with invalid document format."""
        workflow = MedicalTimelineGenerator()
        
        invalid_inputs = {
            "documents": [
                {"type": "medical_record"}  # Missing required 'content' field
            ]
        }
        
        with pytest.raises(ValueError, match="Document 0 missing required 'content' field"):
            workflow.validate_inputs(invalid_inputs)
    
    def test_validate_inputs_invalid_case_type(self):
        """Test input validation with invalid case type."""
        workflow = MedicalTimelineGenerator()
        
        invalid_inputs = {
            "documents": [{"content": "test content"}],
            "case_type": "invalid_type"
        }
        
        with pytest.raises(ValueError, match="'case_type' must be one of"):
            workflow.validate_inputs(invalid_inputs)
    
    @pytest.mark.asyncio
    async def test_execute_basic_workflow(self):
        """Test basic workflow execution."""
        workflow = MedicalTimelineGenerator()
        
        inputs = {
            "documents": [
                {
                    "content": "Patient John Doe admitted with chest pain on January 15th. Diagnosed with myocardial infarction.",
                    "type": "emergency_record",
                    "date": "2023-01-15",
                    "source": "General Hospital"
                },
                {
                    "content": "Follow-up visit on January 20th. Patient discharged in stable condition.",
                    "type": "discharge_summary",
                    "date": "2023-01-20",
                    "source": "General Hospital"
                }
            ],
            "patient_info": {
                "name": "John Doe",
                "dob": "1980-05-12",
                "id": "P123456"
            },
            "case_type": "malpractice"
        }
        
        config = {
            "detail_level": "high",
            "include_analysis": True
        }
        
        outputs = await workflow.execute(inputs, config)
        
        assert len(outputs) >= 2  # Should have timeline and summary at minimum
        
        # Check timeline output
        timeline_output = next((o for o in outputs if o.key == "timeline"), None)
        assert timeline_output is not None
        assert timeline_output.type == "medical_timeline"
        assert isinstance(timeline_output.value, list)
        
        # Check summary output
        summary_output = next((o for o in outputs if o.key == "summary"), None)
        assert summary_output is not None
        assert summary_output.type == "timeline_summary"
        assert "patient" in summary_output.value
        
        # Check analysis output (if included)
        analysis_output = next((o for o in outputs if o.key == "analysis"), None)
        if config.get("include_analysis"):
            assert analysis_output is not None
            assert analysis_output.type == "medical_analysis"
            assert "case_strength" in analysis_output.value
    
    def test_get_input_schema(self):
        """Test getting input schema."""
        workflow = MedicalTimelineGenerator()
        schema = workflow.get_input_schema()
        
        assert schema["type"] == "object"
        assert "documents" in schema["required"]
        assert "documents" in schema["properties"]
        assert "patient_info" in schema["properties"]
        assert "case_type" in schema["properties"]
    
    def test_get_output_schema(self):
        """Test getting output schema."""
        workflow = MedicalTimelineGenerator()
        schema = workflow.get_output_schema()
        
        assert schema["type"] == "object"
        assert "timeline" in schema["properties"]
        assert "summary" in schema["properties"]
        assert "analysis" in schema["properties"]
    
    def test_get_configuration_schema(self):
        """Test getting configuration schema."""
        workflow = MedicalTimelineGenerator()
        schema = workflow.get_configuration_schema()
        
        assert schema["type"] == "object"
        assert "ai_model" in schema["properties"]
        assert "detail_level" in schema["properties"]
        assert "include_analysis" in schema["properties"]
    
    def test_get_examples(self):
        """Test getting example inputs and outputs."""
        workflow = MedicalTimelineGenerator()
        examples = workflow.get_examples()
        
        assert isinstance(examples, list)
        assert len(examples) > 0
        
        example = examples[0]
        assert "input" in example
        assert "output" in example
        assert "documents" in example["input"]
        assert "patient_info" in example["input"]