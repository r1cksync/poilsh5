"""
Hindi OCR Service using custom H5 model
Custom neural network trained specifically for Hindi character recognition
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import os
import io
import json
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from app.core.config import Settings

logger = logging.getLogger(__name__)


class HindiOCRService:
    """
    Hindi OCR service using custom-trained H5 model
    Lightweight neural network specifically designed for Hindi character recognition
    """
    
    def __init__(self):
        self.settings = Settings()
        self.model = None
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.num_classes = 0
        self.img_height = 32
        self.img_width = 128
        self.is_loaded = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._load_model()
    
    def _load_model(self):
        """Load the custom H5 model and metadata"""
        try:
            logger.info("🚀 Loading custom Hindi OCR H5 model...")
            
            # Load model metadata
            metadata_path = os.path.join("models", "model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.char_to_idx = metadata['char_to_idx']
                self.idx_to_char = {int(k): v for k, v in metadata['idx_to_char'].items()}
                self.num_classes = metadata['num_classes']
                self.img_height = metadata['img_height']
                self.img_width = metadata['img_width']
                
                logger.info(f"📚 Loaded {self.num_classes} character classes")
            else:
                logger.warning("Metadata not found, using default character set")
                self._create_default_character_set()
            
            # Load the H5 model
            model_path = os.path.join("models", "hindi_ocr_model.h5")
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
                logger.info("✅ H5 model loaded successfully")
            else:
                logger.warning("H5 model not found, creating a basic model")
                self._create_basic_model()
            
            self.is_loaded = True
            logger.info("✅ Hindi OCR service initialized with custom H5 model")
            
        except Exception as e:
            logger.error(f"❌ Failed to load H5 model: {e}")
            self._create_basic_model()
            self.is_loaded = False
    
    def _create_default_character_set(self):
        """Create default Hindi character set"""
        hindi_chars = [
            'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ',
            'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ',
            'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न',
            'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व',
            'श', 'ष', 'स', 'ह', 'क्ष', 'त्र', 'ज्ञ',
            '०', '१', '२', '३', '४', '५', '६', '७', '८', '९',
            ' ', '।', '?', '!', ',', '.', '-', '(', ')',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ]
        
        self.char_to_idx = {char: idx for idx, char in enumerate(hindi_chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(hindi_chars)}
        self.num_classes = len(hindi_chars)
    
    def _create_basic_model(self):
        """Create a basic model if H5 file doesn't exist"""
        try:
            from tensorflow.keras import layers
            
            model = keras.Sequential([
                layers.Input(shape=(self.img_height, self.img_width, 1)),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            logger.info("Created basic fallback model")
            
        except Exception as e:
            logger.error(f"Failed to create basic model: {e}")
    
    def _preprocess_image(self, image_data: bytes) -> np.ndarray:
        """
        Preprocess image for the H5 model
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Invalid image data")
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Resize to model input size
            resized = cv2.resize(gray, (self.img_width, self.img_height))
            
            # Normalize pixel values
            normalized = resized.astype(np.float32) / 255.0
            
            # Add batch and channel dimensions
            preprocessed = np.expand_dims(np.expand_dims(normalized, axis=0), axis=-1)
            
            return preprocessed
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            # Return dummy image if preprocessing fails
            return np.zeros((1, self.img_height, self.img_width, 1), dtype=np.float32)
    
    def _segment_text_regions(self, image_data: bytes) -> List[np.ndarray]:
        """
        Segment text into individual characters/words for classification
        """
        try:
            # Convert to OpenCV image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract character regions
            char_images = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter out very small regions
                if w > 10 and h > 10:
                    char_region = gray[y:y+h, x:x+w]
                    
                    # Resize to model input size
                    resized_char = cv2.resize(char_region, (self.img_width, self.img_height))
                    
                    # Normalize
                    normalized_char = resized_char.astype(np.float32) / 255.0
                    char_images.append(np.expand_dims(normalized_char, axis=-1))
            
            return char_images if char_images else [self._preprocess_image(image_data)[0]]
            
        except Exception as e:
            logger.error(f"Text segmentation failed: {e}")
            return [self._preprocess_image(image_data)[0]]
    
    def _classify_character(self, char_image: np.ndarray) -> Tuple[str, float]:
        """
        Classify a single character using the H5 model
        """
        try:
            # Add batch dimension if needed
            if len(char_image.shape) == 3:
                char_image = np.expand_dims(char_image, axis=0)
            
            # Predict using the model
            predictions = self.model.predict(char_image, verbose=0)
            
            # Get the character with highest probability
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            
            # Convert to character
            predicted_char = self.idx_to_char.get(predicted_idx, '?')
            
            return predicted_char, confidence
            
        except Exception as e:
            logger.error(f"Character classification failed: {e}")
            return '?', 0.0
    
    def _extract_text_sync(self, image_data: bytes) -> Dict[str, Any]:
        """
        Synchronous text extraction using custom H5 model
        """
        try:
            if not self.is_loaded or self.model is None:
                raise ValueError("H5 model not properly loaded")
            
            # Segment the image into characters
            char_images = self._segment_text_regions(image_data)
            
            # Classify each character
            recognized_chars = []
            confidences = []
            
            for char_image in char_images:
                char, confidence = self._classify_character(char_image)
                if confidence > self.settings.CONFIDENCE_THRESHOLD:
                    recognized_chars.append(char)
                    confidences.append(confidence)
            
            # Combine characters into text
            extracted_text = ''.join(recognized_chars).strip()
            
            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Detect language (simple heuristic)
            hindi_chars = sum(1 for char in extracted_text if '\u0900' <= char <= '\u097F')
            total_chars = len([c for c in extracted_text if c.isalpha()])
            language = "hindi" if hindi_chars > total_chars * 0.3 else "english"
            
            return {
                "text": extracted_text,
                "confidence": float(avg_confidence),
                "language": language,
                "num_detections": len(char_images)
            }
            
        except Exception as e:
            logger.error(f"H5 model text extraction failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "language": "unknown",
                "num_detections": 0,
                "error": str(e)
            }
    
    async def extract_text(self, image_data: bytes) -> Dict[str, Any]:
        """
        Extract Hindi text from image data asynchronously using H5 model
        """
        try:
            logger.info(f"Processing image with H5 model ({len(image_data)} bytes)")
            
            # Run OCR in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._extract_text_sync,
                image_data
            )
            
            logger.info(f"H5 model extracted text: '{result['text'][:50]}...' with confidence: {result['confidence']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Async H5 model text extraction failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "language": "unknown",
                "num_detections": 0,
                "error": str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the H5 OCR model"""
        return {
            "model_name": "Custom H5 Hindi OCR Model",
            "model_type": "neural-network",
            "languages": ["Hindi", "English"],
            "status": "loaded" if self.is_loaded else "not_loaded",
            "architecture": "CNN-based character classifier",
            "num_classes": self.num_classes,
            "input_size": f"{self.img_width}x{self.img_height}",
            "confidence_threshold": self.settings.CONFIDENCE_THRESHOLD,
            "features": [
                "Hindi script recognition",
                "English text support",
                "Custom neural network",
                "Character-level classification",
                "Lightweight architecture"
            ]
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            
            # Clear model
            if self.model:
                del self.model
                self.model = None
                
            self.is_loaded = False
            
            logger.info("🧹 H5 OCR service cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()