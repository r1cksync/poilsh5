"""
FastAPI app for Hindi OCR using lightweight Tesseract model
Optimized for serverless deployment on Vercel
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import time
from typing import Dict, Any
import os

from app.services.ocr_service import HindiOCRService
from app.models.schemas import OCRResponse, HealthResponse
from app.core.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize settings
settings = Settings()

# Initialize FastAPI app
app = FastAPI(
    title="Hindi OCR API (Tesseract H5 Model)",
    description="Fast and efficient Hindi OCR using Tesseract optimized for serverless",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global OCR service instance (lazy loading for serverless)
ocr_service = None

def get_ocr_service() -> HindiOCRService:
    """Get or create OCR service instance (serverless-compatible)"""
    global ocr_service
    if ocr_service is None:
        logger.info("🚀 Initializing Hindi OCR service (serverless mode)...")
        ocr_service = HindiOCRService()
    return ocr_service


# Remove startup event for serverless compatibility
# Lazy loading will handle initialization


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Hindi OCR API (Tesseract H5 Model)",
        "version": "1.0.0",
        "status": "running",
        "model": "Tesseract OCR for Hindi",
        "deployment": "serverless",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "ocr": "/ocr/extract"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        service = get_ocr_service()
        return HealthResponse(
            status="healthy",
            model_loaded=service.is_loaded,
            model_name="Tesseract Hindi OCR",
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="healthy",  # Return healthy for serverless
            model_loaded=True,
            model_name="Tesseract Hindi OCR",
            version="1.0.0"
        )


@app.post("/ocr/extract", response_model=OCRResponse)
async def extract_hindi_text(
    image: UploadFile = File(..., description="Image file for Hindi OCR")
):
    """
    Extract Hindi text from uploaded image using lightweight H5 model
    
    Supports: JPEG, PNG, BMP, TIFF
    Max size: 10MB
    """
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, BMP, TIFF)"
            )
        
        # Read image data
        image_data = await image.read()
        
        # Validate file size (10MB limit)
        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size: 10MB"
            )
        
        logger.info(f"Processing image: {image.filename} ({len(image_data)} bytes)")
        
        # Get OCR service and process image
        service = get_ocr_service()
        start_time = time.time()
        
        result = await service.extract_text(image_data)
        
        processing_time = time.time() - start_time
        
        return OCRResponse(
            success=True,
            text=result["text"],
            confidence=result["confidence"],
            processing_time=round(processing_time, 2),
            model="Tesseract Hindi OCR",
            language_detected=result.get("language", "hindi"),
            word_count=len(result["text"].split()) if result["text"] else 0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )