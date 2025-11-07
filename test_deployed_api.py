#!/usr/bin/env python3
"""
Test script for deployed Hindi OCR H5 Model API
"""

import requests
import json
import time
import sys
import os

# Will be updated with actual API URL after deployment
API_URL = "https://your-deployed-api.vercel.app"
TEST_IMAGE = r"C:\Users\user\Downloads\maxresdefault.jpg"

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

def test_ocr_with_image():
    """Test OCR with the H5 model"""
    print(f"\n📤 Testing H5 Model OCR with: {TEST_IMAGE}")
    
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ Test image not found: {TEST_IMAGE}")
        return False
    
    try:
        with open(TEST_IMAGE, 'rb') as f:
            files = {'image': (os.path.basename(TEST_IMAGE), f, 'image/jpeg')}
            
            file_size = os.path.getsize(TEST_IMAGE)
            print(f"📁 Image: {os.path.basename(TEST_IMAGE)}")
            print(f"📊 Size: {file_size / 1024:.2f} KB")
            
            print("⏳ Processing with custom H5 model...")
            start_time = time.time()
            
            response = requests.post(
                f"{API_URL}/ocr/extract",
                files=files,
                timeout=60
            )
            
            processing_time = time.time() - start_time
            
            print(f"📡 Status Code: {response.status_code}")
            print(f"⏱️  Request Time: {processing_time:.2f}s")
            
            if response.status_code == 200:
                data = response.json()
                print("\n✅ H5 Model OCR Results:")
                print(f"  📝 Text: '{data['text']}'")
                print(f"  🎯 Confidence: {data['confidence']:.2%}")
                print(f"  ⏱️  Processing Time: {data['processing_time']:.2f}s")
                print(f"  🤖 Model: {data['model']}")
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

def run_tests(api_url):
    """Run tests with the given API URL"""
    global API_URL
    API_URL = api_url
    
    print("=" * 70)
    print("🚀 Hindi OCR H5 Model API Test Suite")
    print("=" * 70)
    print(f"API URL: {API_URL}")
    print(f"Test Image: {TEST_IMAGE}")
    print("=" * 70)
    
    tests = [
        ("Health Check", test_health_check),
        ("H5 Model OCR", test_ocr_with_image)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 70}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print(f"\n{'=' * 70}")
    print("📊 Test Results Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\n🎯 Total: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 H5 Model API is working perfectly!")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above.")
    
    return passed == len(results)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    else:
        api_url = input("Enter your deployed API URL: ")
    
    success = run_tests(api_url)
    sys.exit(0 if success else 1)