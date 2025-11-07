# Hindi OCR API - Lightweight H5 Model

A fast and efficient Hindi OCR API using lightweight models optimized for low memory usage.

## Features

- ✅ **Lightweight**: Uses EasyOCR with optimized settings
- ✅ **Hindi Support**: Excellent Hindi script recognition
- ✅ **Low Memory**: Designed to avoid memory limit errors
- ✅ **Fast Processing**: Optimized for quick text extraction
- ✅ **Local Deployment**: No cloud dependencies
- ✅ **Docker Ready**: Easy containerization

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the API

```bash
python main.py
```

### 3. Test the API

```bash
curl -X POST "http://localhost:8000/ocr/extract" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@your_hindi_image.jpg"
```

## API Endpoints

- `GET /` - Service information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `POST /ocr/extract` - Extract Hindi text from image

## Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t hindi-ocr-api .

# Run the container
docker run -p 8000:8000 hindi-ocr-api
```

### Using Docker Compose

```bash
docker-compose up -d
```

## Configuration

Copy `.env.example` to `.env` and configure:

- `USE_GPU=true` - Enable GPU acceleration (if available)
- `CONFIDENCE_THRESHOLD=0.5` - Minimum confidence for text detection
- `MAX_IMAGE_SIZE=10485760` - Maximum image size (10MB)

## Memory Optimization

This API is specifically designed to avoid memory limit errors:

1. **EasyOCR**: Lightweight OCR library
2. **Thread Pool**: Non-blocking processing
3. **Optimized Preprocessing**: Minimal image processing
4. **Small Docker Image**: Slim base image
5. **Efficient Memory Management**: Proper resource cleanup

## Performance Tips

1. **Image Size**: Resize large images before processing
2. **Image Quality**: Use clear, high-contrast images
3. **GPU**: Enable GPU for faster processing (if available)
4. **Batch Processing**: Process multiple images sequentially

## Supported Formats

- JPEG
- PNG  
- BMP
- TIFF

## Example Response

```json
{
  "success": true,
  "text": "नमस्ते यह हिंदी टेक्स्ट है",
  "confidence": 0.92,
  "processing_time": 1.45,
  "model": "H5 Hindi OCR Model",
  "language_detected": "hindi",
  "word_count": 5
}
```

## Error Handling

The API provides detailed error responses:

```json
{
  "success": false,
  "error": "File too large",
  "detail": "Maximum size: 10MB"
}
```

## Development

### Project Structure

```
app/
├── core/
│   ├── config.py          # Configuration settings
├── models/
│   ├── schemas.py         # Pydantic models
├── services/
│   ├── ocr_service.py     # Main OCR service
main.py                    # FastAPI application
requirements.txt           # Python dependencies
Dockerfile                 # Docker configuration
```

### Adding New Features

1. **New OCR Engine**: Modify `ocr_service.py`
2. **Additional Languages**: Update EasyOCR reader configuration
3. **Custom Preprocessing**: Enhance `_preprocess_image` method
4. **Performance Monitoring**: Add logging and metrics

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce image size or disable GPU
2. **Slow Processing**: Enable GPU or reduce image quality
3. **Poor Accuracy**: Improve image preprocessing or lighting

### Logs

Check application logs for detailed error information:

```bash
docker logs <container_id>
```

## Production Deployment

### Recommended Settings

- **Memory**: 2GB minimum
- **CPU**: 2+ cores
- **Storage**: 1GB for models
- **Network**: HTTP/HTTPS proxy

### Scaling

- Use multiple container instances
- Load balancer for distribution
- Redis for caching results
- Database for storing processed text

## License

MIT License - Feel free to use and modify as needed.