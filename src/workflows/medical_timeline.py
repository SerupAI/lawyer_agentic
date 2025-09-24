"""
Medical Timeline Generator Workflow.

Transforms hours of paralegal work into 30-second AI-powered timeline generation
for medical malpractice and personal injury cases.
"""

from typing import Dict, Any, List, Optional
import structlog
from datetime import datetime
import re
import json

from .base import BaseWorkflow
from ..models.workflow import WorkflowOutput

logger = structlog.get_logger(__name__)

class MedicalTimelineGenerator(BaseWorkflow):
    """
    AI-powered medical timeline generator for legal cases.
    
    Processes medical records, reports, and documentation to create
    comprehensive chronological timelines for legal proceedings.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "medical_timeline"
        self.description = "Generate comprehensive medical timelines from case documents"
        self.version = "1.0.0"
        self.estimated_execution_time_seconds = 30
    
    async def execute(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> List[WorkflowOutput]:
        """
        Execute medical timeline generation.
        
        Args:
            inputs: Dict containing:
                - documents: List of medical documents/records
                - patient_info: Patient identification and basic info
                - case_type: Type of case (malpractice, personal_injury, etc.)
                - date_range: Optional date range to focus on
            config: Optional configuration:
                - ai_model: AI model to use (default: gpt-4)
                - detail_level: high/medium/low (default: high)
                - include_analysis: Include AI analysis (default: true)
        
        Returns:
            List of WorkflowOutput objects containing timeline and analysis
        """
        logger.info("Starting medical timeline generation", 
                   case_type=inputs.get('case_type'),
                   document_count=len(inputs.get('documents', [])))
        
        # Extract and validate inputs
        documents = inputs.get('documents', [])
        patient_info = inputs.get('patient_info', {})
        case_type = inputs.get('case_type', 'general')
        date_range = inputs.get('date_range')
        
        # Extract configuration
        if config is None:
            config = {}
        ai_model = config.get('ai_model', 'gpt-4')
        detail_level = config.get('detail_level', 'high')
        include_analysis = config.get('include_analysis', True)
        
        # Process documents to extract medical events
        medical_events = await self._extract_medical_events(documents, patient_info)
        
        # Create chronological timeline
        timeline = await self._create_timeline(medical_events, case_type, detail_level)
        
        # Generate AI analysis if requested
        analysis = None
        if include_analysis:
            analysis = await self._generate_analysis(timeline, case_type, medical_events)
        
        # Format outputs
        outputs = [
            WorkflowOutput(
                key="timeline",
                value=timeline,
                type="medical_timeline",
                metadata={
                    "event_count": len(timeline),
                    "date_range": self._get_timeline_date_range(timeline),
                    "case_type": case_type,
                    "detail_level": detail_level
                }
            ),
            WorkflowOutput(
                key="summary",
                value=self._create_summary(timeline, patient_info),
                type="timeline_summary",
                metadata={
                    "patient": patient_info.get('name', 'Unknown'),
                    "generated_at": datetime.utcnow().isoformat()
                }
            )
        ]
        
        if analysis:
            outputs.append(
                WorkflowOutput(
                    key="analysis",
                    value=analysis,
                    type="medical_analysis",
                    metadata={
                        "ai_model": ai_model,
                        "analysis_type": case_type,
                        "confidence_score": analysis.get('confidence', 0.85)
                    }
                )
            )
        
        logger.info("Medical timeline generation completed", 
                   timeline_events=len(timeline),
                   analysis_included=include_analysis)
        
        return outputs
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate medical timeline inputs."""
        if not isinstance(inputs, dict):
            raise ValueError("Inputs must be a dictionary")
        
        # Check required fields
        if 'documents' not in inputs:
            raise ValueError("Missing required field: 'documents'")
        
        documents = inputs['documents']
        if not isinstance(documents, list) or len(documents) == 0:
            raise ValueError("'documents' must be a non-empty list")
        
        # Validate document format
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                raise ValueError(f"Document {i} must be a dictionary")
            if 'content' not in doc:
                raise ValueError(f"Document {i} missing required 'content' field")
        
        # Validate patient info if provided
        if 'patient_info' in inputs:
            patient_info = inputs['patient_info']
            if not isinstance(patient_info, dict):
                raise ValueError("'patient_info' must be a dictionary")
        
        # Validate case type if provided
        if 'case_type' in inputs:
            valid_types = ['malpractice', 'personal_injury', 'workers_comp', 'general']
            if inputs['case_type'] not in valid_types:
                raise ValueError(f"'case_type' must be one of: {valid_types}")
        
        return True
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get JSON schema for medical timeline inputs."""
        return {
            "type": "object",
            "required": ["documents"],
            "properties": {
                "documents": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["content"],
                        "properties": {
                            "content": {"type": "string", "description": "Document text content"},
                            "type": {"type": "string", "description": "Document type (medical_record, report, etc.)"},
                            "date": {"type": "string", "format": "date", "description": "Document date"},
                            "source": {"type": "string", "description": "Document source/provider"},
                            "metadata": {"type": "object", "description": "Additional document metadata"}
                        }
                    }
                },
                "patient_info": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "dob": {"type": "string", "format": "date"},
                        "id": {"type": "string"},
                        "gender": {"type": "string", "enum": ["male", "female", "other"]},
                        "contact": {"type": "object"}
                    }
                },
                "case_type": {
                    "type": "string",
                    "enum": ["malpractice", "personal_injury", "workers_comp", "general"],
                    "default": "general"
                },
                "date_range": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "format": "date"},
                        "end_date": {"type": "string", "format": "date"}
                    }
                }
            }
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get JSON schema for medical timeline outputs."""
        return {
            "type": "object",
            "properties": {
                "timeline": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "format": "date-time"},
                            "event": {"type": "string"},
                            "category": {"type": "string"},
                            "details": {"type": "string"},
                            "source": {"type": "string"},
                            "significance": {"type": "string", "enum": ["high", "medium", "low"]},
                            "metadata": {"type": "object"}
                        }
                    }
                },
                "summary": {
                    "type": "object",
                    "properties": {
                        "patient": {"type": "string"},
                        "timeline_span": {"type": "string"},
                        "key_events": {"type": "array", "items": {"type": "string"}},
                        "case_overview": {"type": "string"}
                    }
                },
                "analysis": {
                    "type": "object",
                    "properties": {
                        "case_strength": {"type": "string", "enum": ["strong", "moderate", "weak"]},
                        "key_issues": {"type": "array", "items": {"type": "string"}},
                        "timeline_gaps": {"type": "array", "items": {"type": "string"}},
                        "recommendations": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                }
            }
        }
    
    def get_configuration_schema(self) -> Dict[str, Any]:
        """Get JSON schema for workflow configuration."""
        return {
            "type": "object",
            "properties": {
                "ai_model": {
                    "type": "string",
                    "enum": ["gpt-4", "gpt-3.5-turbo", "claude-3"],
                    "default": "gpt-4"
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "default": "high"
                },
                "include_analysis": {
                    "type": "boolean",
                    "default": True
                }
            }
        }
    
    def get_examples(self) -> List[Dict[str, Any]]:
        """Get example inputs and outputs."""
        return [
            {
                "input": {
                    "documents": [
                        {
                            "content": "Patient presented with chest pain on 2023-01-15...",
                            "type": "emergency_room_report",
                            "date": "2023-01-15",
                            "source": "City General Hospital"
                        }
                    ],
                    "patient_info": {
                        "name": "John Doe",
                        "dob": "1980-05-12",
                        "id": "P123456"
                    },
                    "case_type": "malpractice"
                },
                "output": {
                    "timeline": [
                        {
                            "date": "2023-01-15T14:30:00",
                            "event": "Emergency Room Admission",
                            "category": "hospital_visit",
                            "details": "Patient presented with acute chest pain",
                            "source": "City General Hospital",
                            "significance": "high"
                        }
                    ]
                }
            }
        ]
    
    # Private helper methods
    
    async def _extract_medical_events(self, documents: List[Dict], patient_info: Dict) -> List[Dict]:
        """Extract medical events from documents."""
        # This would integrate with AI service for document processing
        # For now, return mock data structure
        events = []
        
        for doc in documents:
            # Extract dates, medical terms, and events from document content
            # This is a simplified version - real implementation would use AI/NLP
            content = doc.get('content', '')
            doc_date = doc.get('date')
            
            # Simple regex patterns for demonstration
            date_pattern = r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b'
            medical_terms = ['diagnosed', 'treated', 'prescribed', 'surgery', 'admitted', 'discharged']
            
            dates_found = re.findall(date_pattern, content)
            
            for term in medical_terms:
                if term in content.lower():
                    events.append({
                        'date': doc_date or datetime.now().isoformat(),
                        'event': f"Medical event: {term}",
                        'category': 'medical_procedure',
                        'details': f"Event extracted from document: {doc.get('type', 'unknown')}",
                        'source': doc.get('source', 'Unknown'),
                        'significance': 'medium',
                        'raw_content': content[:200] + '...' if len(content) > 200 else content
                    })
        
        return events
    
    async def _create_timeline(self, events: List[Dict], case_type: str, detail_level: str) -> List[Dict]:
        """Create chronological timeline from events."""
        # Sort events by date
        sorted_events = sorted(events, key=lambda x: x.get('date', ''))
        
        # Filter and enhance based on detail level
        if detail_level == 'low':
            # Only high significance events
            sorted_events = [e for e in sorted_events if e.get('significance') == 'high']
        
        return sorted_events
    
    async def _generate_analysis(self, timeline: List[Dict], case_type: str, events: List[Dict]) -> Dict[str, Any]:
        """Generate AI-powered analysis of the timeline."""
        # This would integrate with AI service for analysis
        # For now, return mock analysis
        return {
            'case_strength': 'moderate',
            'key_issues': [
                'Delayed diagnosis identified',
                'Treatment protocol deviations noted',
                'Documentation gaps present'
            ],
            'timeline_gaps': [
                'Missing records from January 20-25',
                'Incomplete discharge documentation'
            ],
            'recommendations': [
                'Request additional records from consulting physicians',
                'Investigate standard of care for similar cases',
                'Consider expert medical witness consultation'
            ],
            'confidence': 0.78
        }
    
    def _create_summary(self, timeline: List[Dict], patient_info: Dict) -> Dict[str, Any]:
        """Create executive summary of the timeline."""
        if not timeline:
            return {
                'patient': patient_info.get('name', 'Unknown'),
                'timeline_span': 'No events found',
                'key_events': [],
                'case_overview': 'No medical events identified in provided documents.'
            }
        
        date_range = self._get_timeline_date_range(timeline)
        key_events = [event['event'] for event in timeline[:5]]  # Top 5 events
        
        return {
            'patient': patient_info.get('name', 'Unknown'),
            'timeline_span': date_range,
            'key_events': key_events,
            'case_overview': f'Medical timeline spanning {date_range} with {len(timeline)} documented events.'
        }
    
    def _get_timeline_date_range(self, timeline: List[Dict]) -> str:
        """Get date range string for timeline."""
        if not timeline:
            return 'No dates available'
        
        dates = [event.get('date') for event in timeline if event.get('date')]
        if not dates:
            return 'No dates available'
        
        dates.sort()
        start_date = dates[0][:10] if dates[0] else 'Unknown'
        end_date = dates[-1][:10] if dates[-1] else 'Unknown'
        
        if start_date == end_date:
            return start_date
        
        return f'{start_date} to {end_date}'