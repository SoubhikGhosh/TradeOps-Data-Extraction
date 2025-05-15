# models.py
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, RootModel, validator
from pathlib import Path
from vertexai.generative_models import HarmCategory, HarmBlockThreshold # For AppSettings

# --- Pydantic Models for Vertex AI Responses ---

class BaseVertexResponse(BaseModel):
    """Base model for responses that might contain errors."""
    error: Optional[str] = None
    raw_response: Optional[str] = None # For storing raw text if JSON parsing fails or for debugging

class ExtractedFieldData(BaseModel):
    """
    Represents the extracted value, confidence, and reasoning for a single field.
    """
    value: Optional[Any] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str

class VertexExtractionResult(BaseVertexResponse):
    """
    Represents the structured result of a document extraction call.
    Contains a dictionary of extracted fields.
    """
    extracted_data: Optional[Dict[str, ExtractedFieldData]] = None

class VertexClassificationResponse(BaseVertexResponse):
    """
    Represents the structured result of a document classification call.
    """
    classified_type: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    safety_ratings: Optional[Dict[str, str]] = None # Storing safety ratings as string representations

    @validator('safety_ratings', pre=True)
    def _convert_safety_ratings(cls, v):
        if v is None:
            return None
        if isinstance(v, str): # If already a string (e.g. from error path)
            try:
                # Attempt to parse if it's a stringified dict, otherwise keep as is
                import json
                return json.loads(v)
            except json.JSONDecodeError:
                return {"raw_string": v} # Keep as a dict with raw string
        if isinstance(v, dict): # If it's already a dict
             # Ensure HarmCategory keys are strings for JSON compatibility if they aren't already
            return {str(key): str(value) for key, value in v.items()}
        return {"unknown_format": str(v)}


# --- Pydantic Model for Document Field Definitions (used in AppSettings) ---

class DocumentFieldDefinition(BaseModel):
    """
    Defines the name and description of a field to be extracted for a document type.
    """
    name: str
    description: str

# --- Pydantic Model for Application Settings ---
# This will be defined in config.py using pydantic_settings.BaseSettings
# but related models like DocumentFieldDefinition are here for clarity.

