"""
Pydantic models for workflow requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import uuid

class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class WorkflowRequest(BaseModel):
    """Request model for workflow execution."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    inputs: Dict[str, Any] = Field(..., description="Input data for the workflow")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Optional workflow configuration")
    priority: int = Field(default=1, ge=1, le=10, description="Execution priority (1-10)")
    timeout_seconds: Optional[int] = Field(default=300, description="Maximum execution time in seconds")

class WorkflowOutput(BaseModel):
    """Individual workflow output with metadata."""
    key: str = Field(..., description="Output identifier")
    value: Any = Field(..., description="Output value")
    type: str = Field(..., description="Output data type")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional output metadata")

class WorkflowResponse(BaseModel):
    """Response model for workflow execution."""
    request_id: str
    workflow_name: str
    status: WorkflowStatus
    outputs: List[WorkflowOutput] = Field(default_factory=list)
    execution_time_seconds: Optional[float] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional execution metadata")

class WorkflowSchema(BaseModel):
    """Schema definition for a workflow."""
    name: str
    description: str
    version: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    configuration_schema: Optional[Dict[str, Any]] = None
    examples: Optional[List[Dict[str, Any]]] = None
    estimated_execution_time_seconds: Optional[int] = None

class WorkflowMetrics(BaseModel):
    """Metrics for workflow execution tracking."""
    workflow_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time_seconds: Optional[float] = None
    last_execution_time: Optional[datetime] = None