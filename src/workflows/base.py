"""
Base workflow classes and interfaces for the Lawyer Agentic Platform.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
import asyncio
import structlog
import time
import uuid

from ..models.workflow import (
    WorkflowRequest, 
    WorkflowResponse, 
    WorkflowStatus, 
    WorkflowOutput,
    WorkflowSchema,
    WorkflowMetrics
)

logger = structlog.get_logger(__name__)

class BaseWorkflow(ABC):
    """
    Abstract base class for all workflows in the platform.
    
    Provides common functionality for workflow execution, validation,
    and result processing.
    """
    
    def __init__(self):
        self.name: str = self.__class__.__name__
        self.version: str = "1.0.0"
        self.description: str = ""
        self.estimated_execution_time_seconds: int = 60
        
    @abstractmethod
    async def execute(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> List[WorkflowOutput]:
        """
        Execute the workflow with given inputs.
        
        Args:
            inputs: Input data for the workflow
            config: Optional configuration parameters
            
        Returns:
            List of workflow outputs
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If execution fails
        """
        pass
    
    @abstractmethod
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate input data for the workflow.
        
        Args:
            inputs: Input data to validate
            
        Returns:
            True if inputs are valid
            
        Raises:
            ValueError: If inputs are invalid with descriptive message
        """
        pass
    
    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for workflow inputs.
        
        Returns:
            JSON schema dict for input validation
        """
        pass
    
    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for workflow outputs.
        
        Returns:
            JSON schema dict for output validation
        """
        pass
    
    def get_configuration_schema(self) -> Optional[Dict[str, Any]]:
        """
        Get the JSON schema for workflow configuration.
        
        Returns:
            JSON schema dict for configuration validation, or None if no config needed
        """
        return None
    
    def get_examples(self) -> List[Dict[str, Any]]:
        """
        Get example inputs and outputs for the workflow.
        
        Returns:
            List of example dicts with 'input' and 'output' keys
        """
        return []
    
    def get_schema(self) -> WorkflowSchema:
        """
        Get complete schema information for this workflow.
        """
        return WorkflowSchema(
            name=self.name,
            description=self.description,
            version=self.version,
            input_schema=self.get_input_schema(),
            output_schema=self.get_output_schema(),
            configuration_schema=self.get_configuration_schema(),
            examples=self.get_examples(),
            estimated_execution_time_seconds=self.estimated_execution_time_seconds
        )

class WorkflowManager:
    """
    Manages workflow registration, execution, and monitoring.
    """
    
    def __init__(self):
        self._workflows: Dict[str, BaseWorkflow] = {}
        self._metrics: Dict[str, WorkflowMetrics] = {}
        self._active_executions: Dict[str, WorkflowResponse] = {}
        
    def register_workflow(self, name: str, workflow: BaseWorkflow) -> None:
        """
        Register a workflow with the manager.
        
        Args:
            name: Unique name for the workflow
            workflow: Workflow instance to register
        """
        if name in self._workflows:
            logger.warning("Overwriting existing workflow", workflow_name=name)
            
        self._workflows[name] = workflow
        self._metrics[name] = WorkflowMetrics(workflow_name=name)
        
        logger.info("Registered workflow", 
                   workflow_name=name, 
                   workflow_class=workflow.__class__.__name__)
    
    def get_available_workflows(self) -> List[str]:
        """Get list of available workflow names."""
        return list(self._workflows.keys())
    
    def get_workflow_schema(self, workflow_name: str) -> WorkflowSchema:
        """
        Get schema for a specific workflow.
        
        Args:
            workflow_name: Name of the workflow
            
        Returns:
            Workflow schema
            
        Raises:
            ValueError: If workflow doesn't exist
        """
        if workflow_name not in self._workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")
            
        return self._workflows[workflow_name].get_schema()
    
    async def execute_workflow(
        self, 
        workflow_name: str, 
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> WorkflowResponse:
        """
        Execute a workflow by name.
        
        Args:
            workflow_name: Name of workflow to execute
            inputs: Input data for the workflow
            config: Optional configuration
            request_id: Optional request identifier
            
        Returns:
            Workflow execution response
            
        Raises:
            ValueError: If workflow doesn't exist or inputs are invalid
            RuntimeError: If execution fails
        """
        if workflow_name not in self._workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        
        if request_id is None:
            request_id = str(uuid.uuid4())
            
        workflow = self._workflows[workflow_name]
        metrics = self._metrics[workflow_name]
        
        # Create response object for tracking
        response = WorkflowResponse(
            request_id=request_id,
            workflow_name=workflow_name,
            status=WorkflowStatus.PENDING,
            started_at=datetime.utcnow()
        )
        
        self._active_executions[request_id] = response
        
        try:
            logger.info("Starting workflow execution", 
                       workflow_name=workflow_name, 
                       request_id=request_id)
            
            # Update status to running
            response.status = WorkflowStatus.RUNNING
            
            # Validate inputs
            workflow.validate_inputs(inputs)
            
            # Execute workflow
            start_time = time.time()
            outputs = await workflow.execute(inputs, config)
            execution_time = time.time() - start_time
            
            # Update response with results
            response.status = WorkflowStatus.COMPLETED
            response.outputs = outputs
            response.execution_time_seconds = execution_time
            response.completed_at = datetime.utcnow()
            
            # Update metrics
            metrics.total_executions += 1
            metrics.successful_executions += 1
            metrics.last_execution_time = response.completed_at
            
            # Update average execution time
            if metrics.average_execution_time_seconds is None:
                metrics.average_execution_time_seconds = execution_time
            else:
                metrics.average_execution_time_seconds = (
                    (metrics.average_execution_time_seconds * (metrics.successful_executions - 1) + execution_time)
                    / metrics.successful_executions
                )
            
            logger.info("Workflow execution completed successfully", 
                       workflow_name=workflow_name,
                       request_id=request_id,
                       execution_time_seconds=execution_time)
                       
            return response
            
        except Exception as e:
            # Update response with error
            response.status = WorkflowStatus.FAILED
            response.error_message = str(e)
            response.completed_at = datetime.utcnow()
            response.execution_time_seconds = time.time() - start_time if 'start_time' in locals() else None
            
            # Update metrics
            metrics.total_executions += 1
            metrics.failed_executions += 1
            metrics.last_execution_time = response.completed_at
            
            logger.error("Workflow execution failed", 
                        workflow_name=workflow_name,
                        request_id=request_id,
                        error=str(e))
            
            raise RuntimeError(f"Workflow execution failed: {str(e)}")
            
        finally:
            # Clean up active executions
            if request_id in self._active_executions:
                del self._active_executions[request_id]
    
    def get_workflow_metrics(self, workflow_name: str) -> WorkflowMetrics:
        """
        Get execution metrics for a specific workflow.
        
        Args:
            workflow_name: Name of the workflow
            
        Returns:
            Workflow metrics
            
        Raises:
            ValueError: If workflow doesn't exist
        """
        if workflow_name not in self._metrics:
            raise ValueError(f"Workflow '{workflow_name}' not found")
            
        return self._metrics[workflow_name]
    
    def get_active_executions(self) -> Dict[str, WorkflowResponse]:
        """Get currently active workflow executions."""
        return self._active_executions.copy()