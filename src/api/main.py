"""
FastAPI main application for Lawyer Agentic Platform.

AI-powered workflow orchestration system for legal professionals.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import structlog
from datetime import datetime

from ..workflows.base import WorkflowManager
from ..workflows.medical_timeline import MedicalTimelineGenerator
from ..models.workflow import WorkflowRequest, WorkflowResponse

# Configure structured logging
logger = structlog.get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Lawyer Agentic Platform",
    description="AI-powered workflow orchestration system for legal professionals",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration for web client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize workflow manager
workflow_manager = WorkflowManager()

# Health check models
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    workflows_available: int

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic platform information."""
    return {
        "message": "Lawyer Agentic Platform API",
        "version": "0.1.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring and deployment verification.
    """
    try:
        available_workflows = len(workflow_manager.get_available_workflows())
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            version="0.1.0",
            workflows_available=available_workflows
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/workflows", response_model=Dict[str, Any])
async def list_workflows():
    """
    List all available workflows with their metadata.
    """
    try:
        workflows = workflow_manager.get_available_workflows()
        return {
            "workflows": workflows,
            "total_count": len(workflows),
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error("Failed to list workflows", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve workflows")

@app.post("/workflows/{workflow_name}/execute", response_model=WorkflowResponse)
async def execute_workflow(
    workflow_name: str,
    request: WorkflowRequest
):
    """
    Execute a specific workflow with provided inputs.
    
    Args:
        workflow_name: Name of the workflow to execute
        request: Workflow execution request with inputs and configuration
    """
    try:
        logger.info("Executing workflow", workflow=workflow_name, request_id=request.request_id)
        
        # Execute workflow through manager
        result = await workflow_manager.execute_workflow(
            workflow_name=workflow_name,
            inputs=request.inputs,
            config=request.config,
            request_id=request.request_id
        )
        
        logger.info("Workflow completed successfully", 
                   workflow=workflow_name, 
                   request_id=request.request_id,
                   execution_time=result.execution_time_seconds)
        
        return result
        
    except ValueError as e:
        logger.error("Invalid workflow request", error=str(e), workflow=workflow_name)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Workflow execution failed", 
                    error=str(e), 
                    workflow=workflow_name,
                    request_id=request.request_id if request else None)
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

@app.get("/workflows/{workflow_name}/schema")
async def get_workflow_schema(workflow_name: str):
    """
    Get the input schema for a specific workflow.
    """
    try:
        schema = workflow_manager.get_workflow_schema(workflow_name)
        return schema
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to get workflow schema", error=str(e), workflow=workflow_name)
        raise HTTPException(status_code=500, detail="Failed to retrieve workflow schema")

# Register workflows on startup
@app.on_event("startup")
async def startup_event():
    """Initialize workflows and services on application startup."""
    logger.info("Starting Lawyer Agentic Platform API")
    
    # Register available workflows
    medical_timeline = MedicalTimelineGenerator()
    workflow_manager.register_workflow("medical_timeline", medical_timeline)
    
    logger.info("Platform startup complete", 
               workflows_registered=len(workflow_manager.get_available_workflows()))

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Lawyer Agentic Platform API")

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )