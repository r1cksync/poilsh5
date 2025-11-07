"""
Configuration settings for Hindi OCR API
"""

import os
from typing import List
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Model configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./models/hindi_ocr_model.h5")
    MODEL_WEIGHTS_PATH: str = os.getenv("MODEL_WEIGHTS_PATH", "./models/hindi_ocr_weights.h5")
    
    # Image processing
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    IMAGE_SIZE: tuple = (224, 224)  # Model input size
    
    # OCR settings
    CONFIDENCE_THRESHOLD: float = 0.5
    SUPPORTED_FORMATS: List[str] = [
        "image/jpeg", "image/png", "image/bmp", "image/tiff"
    ]
    
    # Performance settings
    BATCH_SIZE: int = 1
    USE_GPU: bool = False  # Set to True if you have GPU
    
    class Config:
        env_file = ".env"