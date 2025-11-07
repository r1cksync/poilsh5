"""
Hindi OCR Service using EasyOCR (lightweight alternative)
Optimized for Vercel serverless deployment - no TensorFlow dependencies
"""

import cv2
import numpy as np
import easyocr
import logging
import os
import io
from PIL import Image
from typing import Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from app.core.config import Settings

logger = logging.getLogger(__name__)


class HindiOCRService:
    """
    Lightweight Hindi OCR service using EasyOCR
    Designed for serverless deployment without heavy dependencies
    """
    
    def __init__(self):
        self.settings = Settings()
        self.reader = None
        self.is_loaded = False
        self.executor = ThreadPoolExecutor(max_workers=1)  # Reduced for serverless
        self._initialize_reader()
    
    def _initialize_reader(self):
        """Initialize EasyOCR reader with Hindi language support"""
        try:
            logger.info("🚀 Initializing EasyOCR for Hindi text recognition...")
            
            # Initialize with Hindi and English languages
            # gpu=False for serverless compatibility
            self.reader = easyocr.Reader(
                ['hi', 'en'],  # Hindi and English
                gpu=False,  # Always False for serverless
                verbose=False,
                download_enabled=True
            )
            
            self.is_loaded = True
            logger.info("✅ Hindi OCR reader initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OCR reader: {e}")
            self.is_loaded = False
    
    def _preprocess_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        Lightweight image preprocessing for better OCR accuracy
        """
        try:
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Apply adaptive thresholding for better text contrast
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Denoise
            denoised = cv2.medianBlur(processed, 3)
            
            return denoised
            
        except Exception as e:
            logger.warning(f"Preprocessing failed, using original image: {e}")
            return image_array
    
    def _extract_text_sync(self, image_data: bytes) -> Dict[str, Any]:
        """
        Synchronous text extraction using EasyOCR
        """
        try:
            if not self.is_loaded or not self.reader:
                raise ValueError("OCR reader not properly initialized")
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image_array is None:
                raise ValueError("Invalid image data")
            
            # Preprocess image
            processed_image = self._preprocess_image(image_array)
            
            # Extract text using EasyOCR
            results = self.reader.readtext(
                processed_image,
                detail=1,  # Include bounding boxes and confidence
                paragraph=True,  # Group text into paragraphs
                width_ths=0.7,  # Threshold for combining text horizontally
                height_ths=0.7  # Threshold for combining text vertically
            )
            
            # Process results
            extracted_text = ""
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > self.settings.CONFIDENCE_THRESHOLD:
                    extracted_text += text + " "
                    confidences.append(confidence)
            
            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Clean up text
            extracted_text = extracted_text.strip()
            
            # Detect language (simple heuristic)
            hindi_chars = sum(1 for char in extracted_text if '\u0900' <= char <= '\u097F')
            total_chars = len([c for c in extracted_text if c.isalpha()])
            language = "hindi" if hindi_chars > total_chars * 0.3 else "english"
            
            return {
                "text": extracted_text,
                "confidence": float(avg_confidence),
                "language": language,
                "num_detections": len(results)
            }
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "language": "unknown",
                "num_detections": 0,
                "error": str(e)
            }
    
    async def extract_text(self, image_data: bytes) -> Dict[str, Any]:
        """
        Extract Hindi text from image data asynchronously
        """
        try:
            logger.info(f"Processing image ({len(image_data)} bytes)")
            
            # Run OCR in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._extract_text_sync,
                image_data
            )
            
            logger.info(f"Extracted text: '{result['text'][:50]}...' with confidence: {result['confidence']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Async text extraction failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "language": "unknown",
                "num_detections": 0,
                "error": str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the OCR service"""
        return {
            "model_name": "EasyOCR Hindi (Lightweight)",
            "model_type": "lightweight-ocr",
            "languages": ["Hindi", "English"],
            "status": "loaded" if self.is_loaded else "not_loaded",
            "gpu_enabled": False,  # Always False for serverless
            "confidence_threshold": self.settings.CONFIDENCE_THRESHOLD,
            "features": [
                "Hindi script recognition",
                "English text support",
                "Lightweight processing",
                "Serverless compatible",
                "No TensorFlow dependencies"
            ]
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            
            # Clear reader
            self.reader = None
            self.is_loaded = False
            
            logger.info("🧹 OCR service cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()