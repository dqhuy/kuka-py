#!/usr/bin/env python3
"""
Comprehensive test script for Vais dataset
Tests U2-Net and YOLO on all 118 extracted images
Generates comparative report

Author: KUKA-LIB Team
Version: 1.0.0
"""

import cv2
import numpy as np
import os
import sys
import time
from pathlib import Path
import json

# Add kukalib to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kukalib.doc_page_crop import detectAndCropDocumentPage

def test_single_image(image_path, method='u2net', model_name='u2netp'):
    """Test single image with specified method"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        start_time = time.time()
        cropped, debug_img, corners, method_used, time_ms, confidence = detectAndCropDocumentPage(
            img, method=method, model_name=model_name, debug=False
        )
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate crop ratio
        orig_area = img.shape[0] * img.shape[1]
        crop_area = cropped.shape[0] * cropped.shape[1]
        crop_ratio = crop_area / orig_area
        
        was_cropped = not np.array_equal(img.shape[:2], cropped.shape[:2])
        
        return {
            'success': True,
            'method_used': method_used,
            'confidence': confidence,
            'time_ms': time_ms,
            'processing_time_ms': processing_time,
            'was_cropped': was_cropped,
            'crop_ratio': crop_ratio,
            'orig_size': f"{img.shape[1]}x{img.shape[0]}",
            'crop_size': f"{cropped.shape[1]}x{cropped.shape[0]}"
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def run_comprehensive_test():
    """Run tests on all extracted Vais images"""
    
    dataset_dir = "docs/samples/extracted_dataset"
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    # Get all images
    images = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.jpg')])
    
    if not images:
        print(f"❌ No images found in {dataset_dir}")
        return
    
    print(f"=" * 80)
    print(f"VAIS DATASET COMPREHENSIVE TEST")
    print(f"=" * 80)
    print(f"Dataset: {dataset_dir}")
    print(f"Total images: {len(images)}")
    print(f"=" * 80)
    print()
    
    # Separate by DPI
    images_50dpi = [img for img in images if '_50dpi' in img]
    images_100dpi = [img for img in images if '_100dpi' in img]
    
    print(f"📊 Images by resolution:")
    print(f"   50 DPI: {len(images_50dpi)} images")
    print(f"   100 DPI: {len(images_100dpi)} images")
    print()
    
    # Test configurations
    test_configs = [
        {'name': 'U2-Net (u2netp)', 'method': 'u2net', 'model': 'u2netp'},
        {'name': 'U2-Net (u2net full)', 'method': 'u2net', 'model': 'u2net'},
        {'name': 'YOLO (yolov11n)', 'method': 'yolo', 'model': 'yolov11n-seg.pt'},
    ]
    
    all_results = {}
    
    for config in test_configs:
        print(f"\n{'='*80}")
        print(f"Testing: {config['name']}")
        print(f"{'='*80}")
        
        results_50dpi = []
        results_100dpi = []
        
        # Test 50 DPI images
        print(f"\n📸 Testing 50 DPI images ({len(images_50dpi)} total)...")
        for i, img_name in enumerate(images_50dpi, 1):
            img_path = os.path.join(dataset_dir, img_name)
            print(f"   [{i}/{len(images_50dpi)}] {img_name}...", end=" ")
            
            result = test_single_image(img_path, method=config['method'], model_name=config['model'])
            if result and result['success']:
                results_50dpi.append(result)
                status = "✅ CROPPED" if result['was_cropped'] else "⏭️  SKIPPED"
                print(f"{status} (conf: {result['confidence']:.2f}, {result['time_ms']:.0f}ms)")
            else:
                error = result['error'] if result else "Unknown error"
                print(f"❌ FAILED: {error}")
        
        # Test 100 DPI images
        print(f"\n📸 Testing 100 DPI images ({len(images_100dpi)} total)...")
        for i, img_name in enumerate(images_100dpi, 1):
            img_path = os.path.join(dataset_dir, img_name)
            print(f"   [{i}/{len(images_100dpi)}] {img_name}...", end=" ")
            
            result = test_single_image(img_path, method=config['method'], model_name=config['model'])
            if result and result['success']:
                results_100dpi.append(result)
                status = "✅ CROPPED" if result['was_cropped'] else "⏭️  SKIPPED"
                print(f"{status} (conf: {result['confidence']:.2f}, {result['time_ms']:.0f}ms)")
            else:
                error = result['error'] if result else "Unknown error"
                print(f"❌ FAILED: {error}")
        
        # Calculate statistics
        all_results[config['name']] = {
            '50dpi': results_50dpi,
            '100dpi': results_100dpi
        }
        
        # Print summary for this config
        print(f"\n📊 Summary for {config['name']}:")
        print(f"   50 DPI:")
        if results_50dpi:
            cropped_count = sum(1 for r in results_50dpi if r['was_cropped'])
            avg_conf = np.mean([r['confidence'] for r in results_50dpi])
            avg_time = np.mean([r['time_ms'] for r in results_50dpi])
            print(f"      Processed: {len(results_50dpi)}/{len(images_50dpi)}")
            print(f"      Cropped: {cropped_count} ({cropped_count/len(results_50dpi)*100:.1f}%)")
            print(f"      Skipped: {len(results_50dpi)-cropped_count} ({(len(results_50dpi)-cropped_count)/len(results_50dpi)*100:.1f}%)")
            print(f"      Avg Confidence: {avg_conf:.3f}")
            print(f"      Avg Time: {avg_time:.1f}ms")
        
        print(f"   100 DPI:")
        if results_100dpi:
            cropped_count = sum(1 for r in results_100dpi if r['was_cropped'])
            avg_conf = np.mean([r['confidence'] for r in results_100dpi])
            avg_time = np.mean([r['time_ms'] for r in results_100dpi])
            print(f"      Processed: {len(results_100dpi)}/{len(images_100dpi)}")
            print(f"      Cropped: {cropped_count} ({cropped_count/len(results_100dpi)*100:.1f}%)")
            print(f"      Skipped: {len(results_100dpi)-cropped_count} ({(len(results_100dpi)-cropped_count)/len(results_100dpi)*100:.1f}%)")
            print(f"      Avg Confidence: {avg_conf:.3f}")
            print(f"      Avg Time: {avg_time:.1f}ms")
    
    # Comparative analysis
    print(f"\n{'='*80}")
    print(f"COMPARATIVE ANALYSIS")
    print(f"{'='*80}")
    
    for config_name, results in all_results.items():
        print(f"\n{config_name}:")
        
        results_50 = results['50dpi']
        results_100 = results['100dpi']
        
        if results_50 and results_100:
            # Check consistency between resolutions
            cropped_50 = sum(1 for r in results_50 if r['was_cropped'])
            cropped_100 = sum(1 for r in results_100 if r['was_cropped'])
            
            consistency = abs(cropped_50 - cropped_100)
            print(f"   Resolution consistency: {consistency} differences in crop decisions")
            
            # Check confidence correlation
            conf_diff = abs(np.mean([r['confidence'] for r in results_50]) - 
                           np.mean([r['confidence'] for r in results_100]))
            print(f"   Confidence difference: {conf_diff:.3f}")
            
            # Performance comparison
            time_50 = np.mean([r['time_ms'] for r in results_50])
            time_100 = np.mean([r['time_ms'] for r in results_100])
            print(f"   Speed: 50 DPI={time_50:.1f}ms, 100 DPI={time_100:.1f}ms")
    
    # Save detailed results to JSON
    output_file = "test_results_vais_dataset.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Detailed results saved to: {output_file}")
    print(f"\n{'='*80}")
    print(f"TEST COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_comprehensive_test()
