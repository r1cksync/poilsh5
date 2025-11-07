"""
Script to create and train a custom Hindi OCR H5 model
This creates a lightweight neural network specifically for Hindi character recognition
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
import logging
from typing import Tuple, List
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HindiOCRModelBuilder:
    """
    Build and train a custom Hindi OCR model
    """
    
    def __init__(self):
        # Hindi Devanagari character set (basic characters)
        self.hindi_chars = [
            'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ',
            'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ',
            'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न',
            'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व',
            'श', 'ष', 'स', 'ह', 'क्ष', 'त्र', 'ज्ञ',
            '०', '१', '२', '३', '४', '५', '६', '७', '८', '९',
            ' ', '।', '?', '!', ',', '.', '-', '(', ')',
            # English characters for mixed text
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ]
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.hindi_chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.hindi_chars)}
        self.num_classes = len(self.hindi_chars)
        
        # Model parameters
        self.img_width = 128
        self.img_height = 32
        self.max_text_length = 50
        
    def create_cnn_rnn_model(self) -> keras.Model:
        """
        Create a CNN-RNN model for text recognition
        Lightweight architecture suitable for Hindi OCR
        """
        
        # Input layer
        input_img = layers.Input(shape=(self.img_height, self.img_width, 1), name='image')
        
        # CNN layers for feature extraction
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 1))(x)  # Keep more width for text
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 1))(x)
        
        # Reshape for RNN
        new_shape = ((self.img_width // 4), (self.img_height // 4) * 128)
        x = layers.Reshape(target_shape=new_shape)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        # RNN layers for sequence modeling
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
        
        # Output layer
        x = layers.Dense(self.num_classes + 1, activation='softmax')(x)  # +1 for CTC blank
        
        return keras.Model(inputs=input_img, outputs=x, name='hindi_ocr_model')
    
    def create_simple_cnn_model(self) -> keras.Model:
        """
        Create a simpler CNN model for character classification
        More suitable for lightweight deployment
        """
        
        model = keras.Sequential([
            layers.Input(shape=(self.img_height, self.img_width, 1)),
            
            # First CNN block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second CNN block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third CNN block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global pooling and dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def generate_synthetic_data(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data for Hindi characters
        """
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        images = []
        labels = []
        
        for i in range(num_samples):
            # Create random character
            char_idx = np.random.randint(0, self.num_classes)
            char = self.hindi_chars[char_idx]
            
            # Create blank image
            img = np.ones((self.img_height, self.img_width), dtype=np.uint8) * 255
            
            # Add some synthetic character-like patterns
            # This is a simplified approach - in real scenario you'd use actual character images
            center_x, center_y = self.img_width // 2, self.img_height // 2
            
            # Add some geometric patterns to simulate characters
            if char_idx < 30:  # Hindi vowels and consonants
                cv2.rectangle(img, (center_x-10, center_y-8), (center_x+10, center_y+8), 0, -1)
                cv2.circle(img, (center_x, center_y), 5, 255, -1)
            elif char_idx < 50:  # More consonants
                cv2.ellipse(img, (center_x, center_y), (12, 8), 0, 0, 180, 0, -1)
                cv2.line(img, (center_x-8, center_y), (center_x+8, center_y), 0, 2)
            else:  # Numbers and punctuation
                cv2.circle(img, (center_x, center_y), 8, 0, 2)
                cv2.line(img, (center_x, center_y-6), (center_x, center_y+6), 0, 2)
            
            # Add noise for robustness
            noise = np.random.randint(0, 50, (self.img_height, self.img_width))
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
            
            images.append(img)
            labels.append(char_idx)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} samples")
        
        return np.array(images), np.array(labels)
    
    def train_model(self, model: keras.Model, X: np.ndarray, y: np.ndarray) -> keras.Model:
        """
        Train the model with synthetic data
        """
        logger.info("Starting model training...")
        
        # Convert labels to categorical
        y_categorical = keras.utils.to_categorical(y, self.num_classes)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y_categorical[:split_idx], y_categorical[split_idx:]
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Training callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
            keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed!")
        return model
    
    def save_model_and_metadata(self, model: keras.Model, save_path: str):
        """
        Save the trained model and associated metadata
        """
        # Save model
        model_path = os.path.join(save_path, 'hindi_ocr_model.h5')
        model.save(model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # Save character mappings
        metadata = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'num_classes': self.num_classes,
            'img_height': self.img_height,
            'img_width': self.img_width,
            'model_type': 'simple_cnn'
        }
        
        metadata_path = os.path.join(save_path, 'model_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Metadata saved to: {metadata_path}")
        
        # Save weights separately for flexibility
        weights_path = os.path.join(save_path, 'hindi_ocr.weights.h5')
        model.save_weights(weights_path)
        logger.info(f"Weights saved to: {weights_path}")


def main():
    """
    Main function to create and train the Hindi OCR model
    """
    logger.info("🚀 Starting Hindi OCR H5 Model Creation")
    
    # Create output directory
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize model builder
    builder = HindiOCRModelBuilder()
    
    # Create model
    logger.info("Creating model architecture...")
    model = builder.create_simple_cnn_model()
    
    # Print model summary
    model.summary()
    
    # Generate training data
    X, y = builder.generate_synthetic_data(num_samples=2000)
    
    # Train model
    trained_model = builder.train_model(model, X, y)
    
    # Save model and metadata
    builder.save_model_and_metadata(trained_model, models_dir)
    
    logger.info("✅ Hindi OCR H5 model creation completed!")
    logger.info(f"📁 Model files saved in: {models_dir}/")
    logger.info("Files created:")
    logger.info("  - hindi_ocr_model.h5 (full model)")
    logger.info("  - hindi_ocr_weights.h5 (weights only)")
    logger.info("  - model_metadata.json (character mappings)")


if __name__ == "__main__":
    main()