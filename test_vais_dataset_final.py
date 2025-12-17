#!/usr/bin/env python3
"""
V4 Final Comprehensive Test Script for Vais Dataset
Tests both U2-Net and YOLO with unified logic (threshold 0.85)
Saves results with proper tagging for manual verification
Generates comparison report

Usage:
    python test_vais_dataset_final.py

Output:
    - docs/samples/extracted_dataset_output/{name}_{model}_{cropped|skipped}.jpg
    - docs/samples/extracted_dataset_output/test_report.txt
    - docs/samples/extracted_dataset_output/test_results.json

Author: KUKA-LIB Team
Version: V4 Final
"""

import cv2
import numpy as np
import os
import sys
import time
from pathlib import Path
import json
from datetime import datetime

# Add kukalib to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kukalib.doc_page_crop import detectAndCropDocumentPage

# Configuration
CONFIDENCE_THRESHOLD = 0.85  # Unified threshold for both models
USE_CONTENT_VERIFY = True    # Test with content verification enabled
DATASET_DIR = "docs/samples/extracted_dataset"
OUTPUT_DIR = "docs/samples/extracted_dataset_output"

def create_output_dir():
    """Create output directory if it doesn't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")

def test_single_image(image_path, method='u2net', model_name='u2netp'):
    """Test single image with specified method and threshold 0.85"""
    try:
        print(f"  Processing: {os.path.basename(image_path)} with {method}...", end=" ")
        
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Failed to load")
            return None
        
        start_time = time.time()
        cropped, debug_img, corners, method_used, time_ms, confidence = detectAndCropDocumentPage(
            img, 
            method=method, 
            model_name=model_name, 
            debug=False,
            use_content_verify=USE_CONTENT_VERIFY,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        processing_time = (time.time() - start_time) * 1000
        
        # Determine if it was cropped (confidence >= threshold)
        was_cropped = confidence >= CONFIDENCE_THRESHOLD and not np.array_equal(img.shape[:2], cropped.shape[:2])
        status = "cropped" if was_cropped else "skipped"
        
        # Save the result with proper tagging (save debug image with mask overlay when available)
        base_name = Path(image_path).stem  # e.g., Vais_Test_Crop_Lo3.3_page001_100dpi
        output_name = f"{base_name}_{method}_{status}_mask.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        # Prefer saving debug image (contains mask overlay). Fallback to cropped or original image.
        to_save = debug_img if debug_img is not None else (cropped if was_cropped else img)
        cv2.imwrite(output_path, to_save)
        
        print(f"✅ {status} (conf: {confidence:.3f}, time: {time_ms:.1f}ms)")
        
        return {
            'image': os.path.basename(image_path),
            'method': method,
            'model': model_name,
            'confidence': confidence,
            'time_ms': time_ms,
            'processing_time_ms': processing_time,
            'status': status,
            'was_cropped': was_cropped,
            'output_file': output_name,
            'original_size': f"{img.shape[1]}x{img.shape[0]}",
            'result_size': f"{cropped.shape[1]}x{cropped.shape[0]}"
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def test_all_images():
    """Test all images in dataset with both U2-Net and YOLO"""
    print("\n" + "="*70)
    print("V4 FINAL COMPREHENSIVE TEST")
    print("="*70)
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Content Verification: {'Enabled' if USE_CONTENT_VERIFY else 'Disabled'}")
    print(f"Dataset Directory: {DATASET_DIR}")
    print("="*70 + "\n")
    
    create_output_dir()
    
    # Find all images in dataset
    image_extensions = ['.jpg', '.jpeg', '.png']
    all_images = []
    for ext in image_extensions:
        all_images.extend(Path(DATASET_DIR).glob(f"*{ext}"))
    all_images = sorted(all_images)
    
    if not all_images:
        print(f"❌ No images found in {DATASET_DIR}")
        return None, None
    
    print(f"📊 Found {len(all_images)} images\n")
    
    # Test configurations
    configs = [
        ('u2net', 'u2netp'),  # U2-Net with u2netp model
        ('yolo', 'yolov11s-seg'),  # YOLO with yolov11s-seg model
    ]
    
    all_results = []
    model_stats = {}
    
    for method, model_name in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {method.upper()} ({model_name})")
        print(f"{'='*70}\n")
        
        results = []
        for img_path in all_images:
            result = test_single_image(str(img_path), method, model_name)
            if result:
                results.append(result)
                all_results.append(result)
        
        # Calculate statistics for this model
        if results:
            cropped_count = sum(1 for r in results if r['was_cropped'])
            skipped_count = len(results) - cropped_count
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            avg_time = sum(r['time_ms'] for r in results) / len(results)
            
            model_stats[f"{method}_{model_name}"] = {
                'total': len(results),
                'cropped': cropped_count,
                'skipped': skipped_count,
                'cropped_pct': (cropped_count / len(results)) * 100,
                'avg_confidence': avg_confidence,
                'avg_time': avg_time
            }
            
            print(f"\n📊 {method.upper()} Summary:")
            print(f"   Total: {len(results)}")
            print(f"   Cropped: {cropped_count} ({model_stats[f'{method}_{model_name}']['cropped_pct']:.1f}%)")
            print(f"   Skipped: {skipped_count}")
            print(f"   Avg Confidence: {avg_confidence:.3f}")
            print(f"   Avg Time: {avg_time:.1f}ms")
    
    return all_results, model_stats

def generate_report(all_results, model_stats):
    """Generate comprehensive comparison report"""
    report_path = os.path.join(OUTPUT_DIR, "test_report.txt")
    json_path = os.path.join(OUTPUT_DIR, "test_results.json")
    
    # Save detailed JSON results
    with open(json_path, 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'configuration': {
                'confidence_threshold': CONFIDENCE_THRESHOLD,
                'content_verification': USE_CONTENT_VERIFY
            },
            'results': all_results,
            'statistics': model_stats
        }, f, indent=2)
    
    # Generate text report
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("V4 FINAL - DOCUMENT PAGE DETECTION TEST REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}\n")
        f.write(f"Content Verification: {'Enabled' if USE_CONTENT_VERIFY else 'Disabled'}\n")
        f.write(f"Total Images Tested: {len(all_results) // 2}\n\n")  # Divided by 2 since we test with 2 models
        
        # Model comparison
        for model_key, stats in model_stats.items():
            method_name = model_key.split('_')[0].upper()
            model_name = '_'.join(model_key.split('_')[1:])
            
            f.write("="*70 + "\n")
            f.write(f"{method_name} ({model_name}) RESULTS\n")
            f.write("="*70 + "\n")
            f.write(f"Total Images:      {stats['total']}\n")
            f.write(f"Cropped:           {stats['cropped']} ({stats['cropped_pct']:.1f}%)\n")
            f.write(f"Skipped:           {stats['skipped']} ({100-stats['cropped_pct']:.1f}%)\n")
            f.write(f"Avg Confidence:    {stats['avg_confidence']:.3f}\n")
            f.write(f"Avg Time:          {stats['avg_time']:.1f}ms\n\n")
        
        # Comparison analysis
        f.write("="*70 + "\n")
        f.write("COMPARISON ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        if len(model_stats) == 2:
            models = list(model_stats.keys())
            u2net_stats = model_stats[models[0]]
            yolo_stats = model_stats[models[1]]
            
            # Speed comparison
            if yolo_stats['avg_time'] < u2net_stats['avg_time']:
                speed_winner = "YOLO"
                speed_diff = ((u2net_stats['avg_time'] - yolo_stats['avg_time']) / u2net_stats['avg_time']) * 100
            else:
                speed_winner = "U2-Net"
                speed_diff = ((yolo_stats['avg_time'] - u2net_stats['avg_time']) / yolo_stats['avg_time']) * 100
            
            f.write(f"Speed Winner:      {speed_winner} ({speed_diff:.1f}% faster)\n")
            
            # Confidence comparison
            if yolo_stats['avg_confidence'] > u2net_stats['avg_confidence']:
                conf_winner = "YOLO"
                conf_diff = yolo_stats['avg_confidence'] - u2net_stats['avg_confidence']
            else:
                conf_winner = "U2-Net"
                conf_diff = u2net_stats['avg_confidence'] - yolo_stats['avg_confidence']
            
            f.write(f"Higher Confidence: {conf_winner} (+{conf_diff:.3f})\n")
            
            # Crop rate comparison
            if yolo_stats['cropped_pct'] > u2net_stats['cropped_pct']:
                crop_winner = "YOLO"
                crop_diff = yolo_stats['cropped_pct'] - u2net_stats['cropped_pct']
            else:
                crop_winner = "U2-Net"
                crop_diff = u2net_stats['cropped_pct'] - yolo_stats['cropped_pct']
            
            f.write(f"More Crops:        {crop_winner} (+{crop_diff:.1f}%)\n\n")
            
            # Overall recommendation
            f.write("="*70 + "\n")
            f.write("RECOMMENDATION\n")
            f.write("="*70 + "\n\n")
            
            # Simple scoring: speed (30%), confidence (40%), crop rate (30%)
            u2net_score = (30 * (1 - u2net_stats['avg_time'] / (u2net_stats['avg_time'] + yolo_stats['avg_time'])) +
                          40 * (u2net_stats['avg_confidence'] / (u2net_stats['avg_confidence'] + yolo_stats['avg_confidence'])) +
                          30 * (u2net_stats['cropped_pct'] / (u2net_stats['cropped_pct'] + yolo_stats['cropped_pct'])))
            
            yolo_score = (30 * (1 - yolo_stats['avg_time'] / (u2net_stats['avg_time'] + yolo_stats['avg_time'])) +
                         40 * (yolo_stats['avg_confidence'] / (u2net_stats['avg_confidence'] + yolo_stats['avg_confidence'])) +
                         30 * (yolo_stats['cropped_pct'] / (u2net_stats['cropped_pct'] + yolo_stats['cropped_pct'])))
            
            if yolo_score > u2net_score:
                f.write("Winner: YOLO (yolov11s-seg)\n\n")
                f.write("YOLO performs better overall with:\n")
                f.write(f"- Faster processing ({yolo_stats['avg_time']:.1f}ms vs {u2net_stats['avg_time']:.1f}ms)\n")
                f.write(f"- Confidence: {yolo_stats['avg_confidence']:.3f}\n")
                f.write(f"- Crop rate: {yolo_stats['cropped_pct']:.1f}%\n\n")
                f.write("Recommended for: Production use, batch processing, CPU-only environments\n")
            else:
                f.write("Winner: U2-Net (u2netp)\n\n")
                f.write("U2-Net performs better overall with:\n")
                f.write(f"- Processing time: {u2net_stats['avg_time']:.1f}ms\n")
                f.write(f"- Confidence: {u2net_stats['avg_confidence']:.3f}\n")
                f.write(f"- Crop rate: {u2net_stats['cropped_pct']:.1f}%\n\n")
                f.write("Recommended for: General documents, when accuracy is priority\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("MANUAL VERIFICATION NOTES\n")
        f.write("="*70 + "\n\n")
        f.write("Please verify the following:\n")
        f.write("1. Check cropped images for false positives (cropped into text)\n")
        f.write("2. Check skipped images to see if they should have been cropped\n")
        f.write("3. Compare same image from u2net vs yolo\n")
        f.write("4. Adjust threshold if needed based on results\n\n")
        f.write(f"All results saved to: {OUTPUT_DIR}/\n")
        f.write(f"File naming: {{name}}_{{model}}_{{cropped|skipped}}.jpg\n")
    
    print(f"\n📄 Report saved to: {report_path}")
    print(f"📊 Detailed results saved to: {json_path}")
    
    return report_path

def main():
    """Main test execution"""
    print("\n" + "🚀 "*15)
    print("V4 FINAL - COMPREHENSIVE TEST SUITE")
    print("🚀 "*15 + "\n")
    
    start_time = time.time()
    
    # Run tests
    all_results, model_stats = test_all_images()
    
    if all_results and model_stats:
        # Generate report
        report_path = generate_report(all_results, model_stats)
        
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Total Time: {total_time:.1f}s")
        print(f"Results: {OUTPUT_DIR}/")
        print(f"Report: {report_path}")
        print("\n✅ All done! Please review the report and verify results manually.")
        print("="*70 + "\n")
    else:
        print("\n❌ Test failed - no results generated")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
