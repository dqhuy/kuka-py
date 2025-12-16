#!/usr/bin/env python3
"""
Test script to verify U2-Net detection accuracy against expected outputs.
Calculates IoU (Intersection over Union) and pixel-level similarity.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from kukalib.doc_page_crop import detectAndCropDocumentPage

def calculate_iou(img1, img2):
    """
    Calculate Intersection over Union between two images.
    Images are first resized to same dimensions.
    """
    # Resize to same size
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))
    
    # Convert to grayscale
    if len(img1_resized.shape) == 3:
        img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1_resized
    
    if len(img2_resized.shape) == 3:
        img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2_resized
    
    # Threshold to binary
    _, img1_binary = cv2.threshold(img1_gray, 127, 255, cv2.THRESH_BINARY)
    _, img2_binary = cv2.threshold(img2_gray, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate intersection and union
    intersection = np.logical_and(img1_binary, img2_binary).sum()
    union = np.logical_or(img1_binary, img2_binary).sum()
    
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return iou


def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index"""
    try:
        from skimage.metrics import structural_similarity as ssim
        
        # Resize to same size
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        
        img1_resized = cv2.resize(img1, (w, h))
        img2_resized = cv2.resize(img2, (w, h))
        
        # Convert to grayscale
        if len(img1_resized.shape) == 3:
            img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1_resized
        
        if len(img2_resized.shape) == 3:
            img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        else:
            img2_gray = img2_resized
        
        # Calculate SSIM
        score = ssim(img1_gray, img2_gray)
        return score
    except ImportError:
        return None


def calculate_size_similarity(img1, img2):
    """Calculate size similarity as percentage"""
    w1, h1 = img1.shape[1], img1.shape[0]
    w2, h2 = img2.shape[1], img2.shape[0]
    
    area1 = w1 * h1
    area2 = w2 * h2
    
    if area1 == 0 or area2 == 0:
        return 0.0
    
    # Calculate similarity as ratio of smaller to larger
    similarity = min(area1, area2) / max(area1, area2)
    return similarity


def compare_dimensions(result, expected):
    """Compare dimensions and calculate differences"""
    r_h, r_w = result.shape[:2]
    e_h, e_w = expected.shape[:2]
    
    width_diff = abs(r_w - e_w)
    height_diff = abs(r_h - e_h)
    
    width_ratio = r_w / e_w if e_w > 0 else 0
    height_ratio = r_h / e_h if e_h > 0 else 0
    
    return {
        'result_size': (r_w, r_h),
        'expected_size': (e_w, e_h),
        'width_diff': width_diff,
        'height_diff': height_diff,
        'width_ratio': width_ratio,
        'height_ratio': height_ratio
    }


def test_sample(input_path, expected_path, method='u2net', debug=True):
    """Test a single sample and return accuracy metrics"""
    print("\n" + "="*80)
    print(f"Testing: {os.path.basename(input_path)}")
    print("="*80)
    
    # Load images
    input_img = cv2.imread(input_path)
    expected_img = cv2.imread(expected_path)
    
    if input_img is None:
        print(f"❌ Failed to load input: {input_path}")
        return None
    
    if expected_img is None:
        print(f"❌ Failed to load expected: {expected_path}")
        return None
    
    print(f"Input size: {input_img.shape[1]}x{input_img.shape[0]}")
    print(f"Expected output size: {expected_img.shape[1]}x{expected_img.shape[0]}")
    
    # Run detection
    print(f"\nRunning detection with method: {method}")
    print("-"*80)
    
    result, debug_img, corners, method_used = detectAndCropDocumentPage(
        input_img, 
        method=method,
        debug=debug
    )
    
    print("-"*80)
    print(f"\nMethod used: {method_used}")
    
    if method_used == "failed":
        print("❌ Detection FAILED")
        return {
            'success': False,
            'method': method_used,
            'input_size': input_img.shape[:2],
            'expected_size': expected_img.shape[:2],
            'result_size': result.shape[:2]
        }
    
    print(f"Result size: {result.shape[1]}x{result.shape[0]}")
    
    # Calculate metrics
    print("\n" + "-"*80)
    print("ACCURACY METRICS")
    print("-"*80)
    
    # Dimension comparison
    dim_comp = compare_dimensions(result, expected_img)
    print(f"\nDimension Comparison:")
    print(f"  Result: {dim_comp['result_size'][0]}x{dim_comp['result_size'][1]}")
    print(f"  Expected: {dim_comp['expected_size'][0]}x{dim_comp['expected_size'][1]}")
    print(f"  Width diff: {dim_comp['width_diff']}px ({(1-dim_comp['width_ratio'])*100:.1f}% difference)")
    print(f"  Height diff: {dim_comp['height_diff']}px ({(1-dim_comp['height_ratio'])*100:.1f}% difference)")
    
    # Size similarity
    size_sim = calculate_size_similarity(result, expected_img)
    print(f"\nSize Similarity: {size_sim*100:.2f}%")
    
    # IoU
    iou = calculate_iou(result, expected_img)
    print(f"IoU (Intersection over Union): {iou*100:.2f}%")
    
    # SSIM
    ssim_score = calculate_ssim(result, expected_img)
    if ssim_score is not None:
        print(f"SSIM (Structural Similarity): {ssim_score*100:.2f}%")
    else:
        print(f"SSIM: Not available (install scikit-image)")
    
    # Overall accuracy (weighted average)
    accuracy = (size_sim * 0.3 + iou * 0.4 + (ssim_score or 0) * 0.3) * 100
    print(f"\n{'='*80}")
    print(f"OVERALL ACCURACY: {accuracy:.2f}%")
    print(f"{'='*80}")
    
    # Determine if passes 95% threshold
    passes = accuracy >= 95.0
    if passes:
        print(f"✅ PASS - Accuracy >= 95%")
    else:
        print(f"⚠️  NEEDS IMPROVEMENT - Accuracy < 95% (got {accuracy:.2f}%)")
    
    return {
        'success': True,
        'method': method_used,
        'input_size': input_img.shape[:2],
        'expected_size': expected_img.shape[:2],
        'result_size': result.shape[:2],
        'size_similarity': size_sim * 100,
        'iou': iou * 100,
        'ssim': ssim_score * 100 if ssim_score else None,
        'accuracy': accuracy,
        'passes_95': passes,
        'dim_comparison': dim_comp
    }


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("U2-NET ACCURACY TEST SUITE")
    print("="*80)
    
    samples_dir = "docs/samples"
    input_dir = os.path.join(samples_dir, "input")
    output_dir = os.path.join(samples_dir, "output")
    
    # Test cases
    test_cases = [
        ("customer_sample_1.jpg", "customer_sample_1_cropped.jpg"),
        ("customer_sample_2.jpg", "customer_sample_2_cropped.jpg"),
    ]
    
    results = []
    
    for input_file, expected_file in test_cases:
        input_path = os.path.join(input_dir, input_file)
        expected_path = os.path.join(output_dir, expected_file)
        
        if not os.path.exists(input_path):
            print(f"\n⚠️  Skipping {input_file} - file not found")
            continue
        
        if not os.path.exists(expected_path):
            print(f"\n⚠️  Skipping {input_file} - expected output not found")
            continue
        
        result = test_sample(input_path, expected_path, method='u2net', debug=True)
        if result:
            results.append((input_file, result))
    
    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if not results:
        print("No tests were run!")
        return
    
    for filename, result in results:
        status = "✅ PASS" if result.get('passes_95', False) else "⚠️  NEEDS IMPROVEMENT"
        accuracy = result.get('accuracy', 0)
        print(f"{filename}: {status} - {accuracy:.2f}% accuracy")
    
    # Overall pass rate
    passed = sum(1 for _, r in results if r.get('passes_95', False))
    total = len(results)
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\nPass Rate: {passed}/{total} ({pass_rate:.1f}%)")
    
    if pass_rate >= 100:
        print("\n🎉 All tests passed with >= 95% accuracy!")
    elif pass_rate >= 50:
        print("\n⚠️  Some tests need improvement")
    else:
        print("\n❌ Most tests failing - significant improvements needed")


if __name__ == "__main__":
    main()
