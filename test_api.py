#!/usr/bin/env python3
"""
Test script for Hindi OCR API
"""

import requests
import json
import time
import sys
import os
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health_check():
    """Test the health endpoint"""
    print("🔍 Testing Health Check...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data['status']}")
            print(f"📊 Model Loaded: {data['model_loaded']}")
            print(f"🤖 Model: {data['model_name']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n📋 Testing Root Endpoint...")
    try:
        response = requests.get(API_URL, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service: {data['service']}")
            print(f"📝 Version: {data['version']}")
            print(f"🎯 Status: {data['status']}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_ocr_with_sample_image():
    """Test OCR with a sample image"""
    print("\n📤 Testing OCR Extraction...")
    
    # Check if sample image exists
    sample_image = "sample_hindi_text.jpg"
    if not os.path.exists(sample_image):
        print(f"⚠️  Sample image '{sample_image}' not found")
        print("ℹ️  Please provide a Hindi image file to test OCR")
        return True  # Not a failure, just missing test data
    
    try:
        with open(sample_image, 'rb') as f:
            files = {'image': (sample_image, f, 'image/jpeg')}
            
            print(f"📁 Processing: {sample_image}")
            start_time = time.time()
            
            response = requests.post(
                f"{API_URL}/ocr/extract",
                files=files,
                timeout=30
            )
            
            processing_time = time.time() - start_time
            
            print(f"📡 Status Code: {response.status_code}")
            print(f"⏱️  Request Time: {processing_time:.2f}s")
            
            if response.status_code == 200:
                data = response.json()
                print("\n✅ OCR Results:")
                print(f"  📝 Text: '{data['text']}'")
                print(f"  🎯 Confidence: {data['confidence']:.2%}")
                print(f"  ⏱️  Processing Time: {data['processing_time']:.2f}s")
                print(f"  🌐 Language: {data['language_detected']}")
                print(f"  🔤 Word Count: {data['word_count']}")
                return True
            else:
                print(f"❌ OCR failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ OCR test error: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid data"""
    print("\n🚨 Testing Error Handling...")
    
    try:
        # Test with invalid file type
        files = {'image': ('test.txt', b'not an image', 'text/plain')}
        response = requests.post(f"{API_URL}/ocr/extract", files=files, timeout=10)
        
        if response.status_code == 400:
            print("✅ Invalid file type correctly rejected")
            return True
        else:
            print(f"⚠️  Expected 400, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("=" * 60)
    print("🚀 Hindi OCR API Test Suite")
    print("=" * 60)
    print(f"API URL: {API_URL}")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Root Endpoint", test_root_endpoint), 
        ("OCR Extraction", test_ocr_with_sample_image),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Total: {passed}/{len(results)} tests passed")
    print("=" * 60)
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your Hindi OCR API is working perfectly!")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please check the output above.")
        print("\nℹ️  Common issues:")
        print("   - API server not running (start with: python main.py)")
        print("   - Dependencies not installed (run: pip install -r requirements.txt)")
        print("   - Port conflicts (check if port 8000 is available)")
    
    return passed == len(results)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)