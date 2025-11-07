"""
Pydantic schemas for API requests and responses
"""

from typing import Optional
from pydantic import BaseModel, Field


class OCRResponse(BaseModel):
    """Response schema for OCR operations"""
    success: bool
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time: float
    model: str
    language_detected: str = "hindi"
    word_count: int = 0


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str
    model_loaded: bool
    model_name: str
    version: str


class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = False
    error: str
    detail: Optional[str] = None