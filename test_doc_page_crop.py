#!/usr/bin/env python3
"""
Test script for document page detection and cropping feature.
This script tests the basic functionality without requiring U2-Net.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kukalib.doc_page_crop import detectAndCropDocumentPage, getVersionInfo
import cv2
import numpy as np

def test_basic_detection():
    """Test basic document detection with sample images"""
    print("=" * 60)
    print("Testing Document Page Detection & Cropping")
    print("=" * 60)
    
    version = getVersionInfo()
    print(f"\nVersion: {version['version']}")
    print(f"Date: {version['date']}")
    
    # Get sample images
    sample_dir = "docs/samples/input"
    if not os.path.exists(sample_dir):
        print(f"\nError: Sample directory not found: {sample_dir}")
        return False
    
    samples = [f for f in os.listdir(sample_dir) if f.endswith('.jpg') and not ('cropped' in f or 'debug' in f)]
    
    if not samples:
        print(f"\nError: No sample images found in {sample_dir}")
        return False
    
    print(f"\nFound {len(samples)} sample images")
    print("-" * 60)
    
    success_count = 0
    fail_count = 0
    
    for sample in sorted(samples):
        sample_path = os.path.join(sample_dir, sample)
        print(f"\nTesting: {sample}")
        
        # Load image
        img = cv2.imread(sample_path)
        if img is None:
            print(f"  ❌ Failed to load image")
            fail_count += 1
            continue
        
        print(f"  Input size: {img.shape[1]}x{img.shape[0]}")
        
        # Test with OpenCV method (doesn't require extra dependencies)
        cropped, debug, corners, method = detectAndCropDocumentPage(img, method='opencv', debug=False)
        
        if method == 'opencv' and cropped is not None and cropped.size > 0:
            print(f"  ✅ Detection successful with {method}")
            print(f"  Output size: {cropped.shape[1]}x{cropped.shape[0]}")
            success_count += 1
        else:
            print(f"  ⚠️  Detection failed (method: {method})")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {success_count} successful, {fail_count} failed")
    print("=" * 60)
    
    return success_count > 0

def test_pdf_support():
    """Test if PDF support is available"""
    print("\n" + "=" * 60)
    print("Testing PDF Support")
    print("=" * 60)
    
    try:
        import fitz
        print("✅ PyMuPDF (fitz) is installed - PDF support available")
        return True
    except ImportError:
        print("⚠️  PyMuPDF not installed - PDF support not available")
        print("   Install with: pip install PyMuPDF")
        return False

def test_ai_support():
    """Test if AI detection (U2-Net) is available"""
    print("\n" + "=" * 60)
    print("Testing AI Detection Support (U2-Net)")
    print("=" * 60)
    
    try:
        import rembg
        print("✅ rembg is installed - U2-Net AI detection available")
        return True
    except ImportError:
        print("ℹ️  rembg not installed - U2-Net detection not available")
        print("   This is optional. Install with: pip install rembg")
        print("   OpenCV method will be used as fallback")
        return False

if __name__ == '__main__':
    print("\n")
    print("*" * 60)
    print("KUKA-LIB Document Page Detection Test Suite")
    print("*" * 60)
    
    # Run tests
    detection_ok = test_basic_detection()
    pdf_ok = test_pdf_support()
    ai_ok = test_ai_support()
    
    print("\n" + "*" * 60)
    print("Test Summary")
    print("*" * 60)
    print(f"Basic Detection:  {'✅ PASS' if detection_ok else '❌ FAIL'}")
    print(f"PDF Support:      {'✅ Available' if pdf_ok else '⚠️  Not available'}")
    print(f"AI Detection:     {'✅ Available' if ai_ok else 'ℹ️  Not available (optional)'}")
    print("*" * 60)
    
    if detection_ok:
        print("\n🎉 Core functionality is working!")
        print("You can now use the document page detection feature.")
        print("\nNext steps:")
        print("1. Try the demo app: streamlit run demo-app.py")
        print("2. Navigate to: http://localhost:8501/Doc-Page-Crop")
        print("3. Optional: Install rembg for AI detection: pip install rembg")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("\n")
    
    sys.exit(0 if detection_ok else 1)
