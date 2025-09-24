-- Initial database setup for Lawyer Agentic Platform
-- This script runs when PostgreSQL container is first created

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create basic schema for workflow tracking
CREATE SCHEMA IF NOT EXISTS workflows;

-- Workflow execution tracking table
CREATE TABLE IF NOT EXISTS workflows.executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_name VARCHAR(100) NOT NULL,
    request_id VARCHAR(255) NOT NULL UNIQUE,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    inputs JSONB NOT NULL,
    outputs JSONB,
    config JSONB,
    error_message TEXT,
    execution_time_seconds FLOAT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Workflow metrics table
CREATE TABLE IF NOT EXISTS workflows.metrics (
    workflow_name VARCHAR(100) PRIMARY KEY,
    total_executions INTEGER DEFAULT 0,
    successful_executions INTEGER DEFAULT 0,
    failed_executions INTEGER DEFAULT 0,
    average_execution_time_seconds FLOAT,
    last_execution_time TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_executions_workflow_name ON workflows.executions(workflow_name);
CREATE INDEX IF NOT EXISTS idx_executions_status ON workflows.executions(status);
CREATE INDEX IF NOT EXISTS idx_executions_started_at ON workflows.executions(started_at);
CREATE INDEX IF NOT EXISTS idx_executions_request_id ON workflows.executions(request_id);

-- Create update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update timestamp trigger to tables
CREATE TRIGGER update_executions_updated_at
    BEFORE UPDATE ON workflows.executions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_metrics_updated_at
    BEFORE UPDATE ON workflows.metrics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert initial workflow metrics records
INSERT INTO workflows.metrics (workflow_name) VALUES ('medical_timeline') ON CONFLICT DO NOTHING;

-- Grant necessary permissions
GRANT USAGE ON SCHEMA workflows TO lawyeragent;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA workflows TO lawyeragent;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA workflows TO lawyeragent;