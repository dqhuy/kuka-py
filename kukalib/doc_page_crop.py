"""
KUKA LIB - Document Page Detection and Cropping Module

This module provides functionality to detect and crop document pages from images,
removing background and applying perspective correction.

Supports multiple detection methods:
1. U2-Net based AI detection (recommended for complex backgrounds)
2. Traditional OpenCV methods (fast, good for simple cases)
3. Hybrid auto-selection (tries best method automatically)

Author: KUKA-LIB Team
Version: 1.0.0
Date: 2024-12-16
"""

import cv2
import numpy as np
import os
import sys
import datetime
import traceback
from typing import Tuple, Optional, List

# Import existing cardcrop utilities
from kukalib.cardcrop import (
    order_points, 
    find_dest, 
    biggestContour,
    sobel,
    getLineAngle,
    getLineLength
)

def getVersionInfo():
    """Get version information for the module"""
    versionInfo = {
        "version": "1.0.0",
        "date": datetime.date(2024, 12, 16)
    }
    return versionInfo


def detectDocumentPage_U2Net(src: np.ndarray, model_name: str = 'u2netp', debug: bool = False, 
                            use_content_verify: bool = False, confidence_threshold: float = 0.85,
                            source_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, tuple, bool, float, float]:
    """
    Detect document page using U2-Net AI model with intelligent tight cropping.
    
    V4 Final Unified Logic:
    - Default confidence threshold: 0.85 (unified with YOLO)
    - Content-based verification for medium confidence (60-85%)
    - Three-tier decision system (high/medium/low confidence)
    - Default to u2netp (lightweight model, only model option)
    - Intelligent tight cropping (no fixed margins)
    - Multiple contour detection (handles 2-page documents)
    - Smart content verification to avoid cutting text
    
    This method uses deep learning for robust detection even with complex backgrounds.
    Uses standalone U2-Net implementation with simple bounding box cropping.
    NO perspective transform/dewarp - prioritizes preserving content over perfect crop.
    
    Best for: Complex backgrounds, poor lighting, cluttered scenes, multi-page documents
    
    Parameters:
    -----------
    src : np.ndarray
        Input image (BGR format)
    model_name : str
        'u2netp' for lightweight (4.4MB, default and only option in V4 Final)
    debug : bool
        If True, print extensive debug information
    use_content_verify : bool
        If True, use content-based verification for medium confidence (60-85%)
    confidence_threshold : float
        Threshold for cropping decision (default 0.85, unified with YOLO)
        
    Returns:
    --------
    tuple: (cropped_img, debug_img, corners, success, time_ms, confidence)
        cropped_img: Cropped image (bounding box only, no warp)
        debug_img: Visualization of detection process
        corners: Tuple of (tl, tr, br, bl) corner points
        success: Boolean indicating if detection succeeded
        time_ms: Detection time in milliseconds
        confidence: Confidence score (0-1) for the detection quality
    """
    import time
    start_time = time.time()
    
    if debug:
        print("\n" + "="*70, file=sys.stderr)
        print(f"U2-Net Method: Starting document detection (model: {model_name})", file=sys.stderr)
        if source_path:
            print(f"Processing file: {source_path}", file=sys.stderr)
        print("="*70, file=sys.stderr)
        print(f"Input image shape: {src.shape}", file=sys.stderr)
    
    try:
        # Import our custom U2-Net implementation
        from kukalib.u2net_model import run_u2net_inference
        
        if debug:
            print("✅ U2-Net module imported successfully", file=sys.stderr)
        
        # Run U2-Net inference to get mask
        mask = run_u2net_inference(src, model_name=model_name, debug=debug, source_path=source_path)
        
        if mask is None:
            if debug:
                print("❌ U2-Net inference failed", file=sys.stderr)
            time_ms = (time.time() - start_time) * 1000
            return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False, time_ms, 0.0
        
        if debug:
            print(f"✅ Got mask from U2-Net: {mask.shape}", file=sys.stderr)
            print(f"   Mask value range: [{mask.min()}, {mask.max()}]", file=sys.stderr)
            print(f"   Non-zero pixels: {np.count_nonzero(mask)} ({np.count_nonzero(mask)/mask.size*100:.1f}%)", file=sys.stderr)
        
        # Use Otsu's thresholding for better adaptive threshold
        _, mask_binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if debug:
            print(f"✅ Binary mask created (Otsu's method)", file=sys.stderr)
            print(f"   White pixels: {np.count_nonzero(mask_binary)} ({np.count_nonzero(mask_binary)/mask_binary.size*100:.1f}%)", file=sys.stderr)
        
        # Morphological operations to clean mask - more aggressive
        kernel_large = np.ones((25, 25), np.uint8)
        kernel_small = np.ones((15, 15), np.uint8)
        
        # Close small gaps
        mask_cleaned = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_large, iterations=3)
        # Remove small noise
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel_small, iterations=2)
        # Dilate slightly to include edges
        mask_cleaned = cv2.dilate(mask_cleaned, kernel_small, iterations=1)
        
        if debug:
            print(f"✅ Morphological operations applied", file=sys.stderr)
            print(f"   Cleaned mask white pixels: {np.count_nonzero(mask_cleaned)} ({np.count_nonzero(mask_cleaned)/mask_cleaned.size*100:.1f}%)", file=sys.stderr)
        
        # Find contours in mask
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if debug:
            print(f"✅ Found {len(contours)} contours", file=sys.stderr)
        
        if len(contours) == 0:
            if debug:
                print("❌ No contours found in mask", file=sys.stderr)
            time_ms = (time.time() - start_time) * 1000
            return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False, time_ms, 0.0
        
        # Sort contours by area to analyze multiple potential document regions
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        if debug:
            print(f"✅ Analyzing top contours:", file=sys.stderr)
            for i, cnt in enumerate(sorted_contours[:5]):  # Top 5
                area = cv2.contourArea(cnt)
                area_pct = area / (src.shape[0] * src.shape[1]) * 100
                print(f"   Contour {i+1}: area={area:.0f} ({area_pct:.1f}%)", file=sys.stderr)
        
        # Check for multiple significant contours (e.g., 2-page newspaper)
        significant_contours = []
        for cnt in sorted_contours:
            area = cv2.contourArea(cnt)
            area_ratio = area / (src.shape[0] * src.shape[1])
            if area_ratio > 0.05:  # At least 5% of image
                significant_contours.append(cnt)
            if len(significant_contours) >= 3:  # Max 3 regions
                break
        
        if debug:
            print(f"✅ Found {len(significant_contours)} significant contours (>5% of image)", file=sys.stderr)
        
        # If multiple significant contours, combine them (multi-page document case)
        if len(significant_contours) > 1:
            if debug:
                print(f"⚠️  Multiple document regions detected - combining to avoid content loss", file=sys.stderr)
            # Combine all significant contours
            all_points = np.vstack([cnt for cnt in significant_contours])
            x, y, w, h = cv2.boundingRect(all_points)
            combined = True
        else:
            # Single contour
            largest_contour = significant_contours[0]
            x, y, w, h = cv2.boundingRect(largest_contour)
            combined = False
        
        if debug:
            print(f"✅ Bounding rectangle: x={x}, y={y}, w={w}, h={h}", file=sys.stderr)
            if combined:
                print(f"   (Combined from {len(significant_contours)} regions)", file=sys.stderr)
        
        # Calculate area statistics
        bbox_area = w * h
        img_area = src.shape[0] * src.shape[1]
        area_ratio = bbox_area / img_area
        
        if debug:
            print(f"   Area ratio: {area_ratio*100:.2f}% of image", file=sys.stderr)
        
        # Confidence scoring system
        confidence = 0.0
        confidence_factors = {}
        
        # Factor 1: Area ratio (0-0.35 points)
        if area_ratio > 0.85:
            confidence_factors['area'] = 0.35  # High confidence - document fills image
        elif area_ratio > 0.50:
            confidence_factors['area'] = 0.30  # Good confidence
        elif area_ratio > 0.20:
            confidence_factors['area'] = 0.20  # Moderate confidence
        else:
            confidence_factors['area'] = 0.10  # Low confidence
        
        # Factor 2: Mask quality (0-0.25 points)
        mask_fill_ratio = np.count_nonzero(mask_cleaned) / mask_cleaned.size
        if mask_fill_ratio > 0.30:
            confidence_factors['mask_quality'] = 0.25
        elif mask_fill_ratio > 0.15:
            confidence_factors['mask_quality'] = 0.20
        else:
            confidence_factors['mask_quality'] = 0.10
        
        # Factor 3: Aspect ratio (0-0.20 points)
        aspect_ratio = w / h if h > 0 else 0
        # Document typically has reasonable aspect ratio (0.3 to 3.0)
        if 0.5 <= aspect_ratio <= 2.0:
            confidence_factors['aspect'] = 0.20  # Standard document proportions
        elif 0.3 <= aspect_ratio <= 3.0:
            confidence_factors['aspect'] = 0.15  # Acceptable
        else:
            confidence_factors['aspect'] = 0.05  # Unusual aspect ratio
        
        # Factor 4: Multiple contours penalty/bonus (0-0.20 points)
        if combined:
            # Multiple regions - lower confidence but prevented content loss
            confidence_factors['multi_region'] = 0.10
        else:
            # Single clear region - higher confidence
            confidence_factors['multi_region'] = 0.20
        
        # Calculate total confidence
        confidence = sum(confidence_factors.values())
        
        if debug:
            print(f"\n✅ Confidence Scoring:", file=sys.stderr)
            for factor, score in confidence_factors.items():
                print(f"   {factor}: {score:.2f}", file=sys.stderr)
            print(f"   TOTAL CONFIDENCE: {confidence:.2f} ({confidence*100:.0f}%)", file=sys.stderr)
        
        # V4 Advanced: Three-tier decision logic with adaptive threshold
        should_skip_crop = False
        skip_reason = ""
        verification_used = False
        verification_passed = False
        
        # Rule 1: Document occupies most of image (>92%) - likely no background
        if area_ratio > 0.92:
            should_skip_crop = True
            skip_reason = "no background detected (>92% coverage)"
            confidence = max(confidence, 0.85)  # High confidence in "no crop" decision
        
        # Rule 2: Three-tier confidence system
        # Tier 1: High confidence - Auto crop
        elif confidence >= confidence_threshold:
            should_skip_crop = False
            if debug:
                print(f"\n✅ HIGH CONFIDENCE ({confidence:.2f} >= {confidence_threshold:.2f}): Auto crop", file=sys.stderr)
        
        # Tier 2: Medium confidence (60-90%) - Use content verification if enabled
        elif confidence >= 0.60 and use_content_verify:
            verification_used = True
            if debug:
                print(f"\n⚠️  MEDIUM CONFIDENCE ({confidence:.2f}): Using content verification", file=sys.stderr)
            
            # Import content verification function
            try:
                from kukalib.training_export import verify_removed_content
                
                # Get the crop bbox
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Create bbox for verification
                bbox = (x, y, w, h)
                
                # Verify removed regions don't contain text
                is_suspicious, details = verify_removed_content(src, mask_binary, bbox, debug=debug)
                
                if is_suspicious:
                    should_skip_crop = True
                    skip_reason = f"content verification failed: {details.get('reason', 'text detected in removed region')}"
                    verification_passed = False
                    if debug:
                        print(f"   ❌ Verification FAILED: {skip_reason}", file=sys.stderr)
                else:
                    should_skip_crop = False
                    verification_passed = True
                    if debug:
                        print(f"   ✅ Verification PASSED: Safe to crop", file=sys.stderr)
            except Exception as e:
                if debug:
                    print(f"   ⚠️  Content verification error: {e}", file=sys.stderr)
                # If verification fails, be conservative - skip crop
                should_skip_crop = True
                skip_reason = "verification error (being conservative)"
        
        # Tier 3: Low confidence (<60%) or medium without verification - Skip
        elif confidence < 0.60:
            should_skip_crop = True
            skip_reason = f"confidence too low ({confidence:.2f}, minimum 0.60)"
        else:
            # Medium confidence without verification enabled
            should_skip_crop = True
            skip_reason = f"confidence below threshold ({confidence:.2f}, need {confidence_threshold:.2f})"
        
        # Rule 3: Would remove too much (>70% removal) - extra safety check
        if not should_skip_crop and area_ratio < 0.30:
            should_skip_crop = True
            skip_reason = f"suspicious crop (would remove {(1-area_ratio)*100:.0f}% of image)"
        
        if should_skip_crop:
            if debug:
                print(f"\n⚠️  SKIPPING CROP: {skip_reason}", file=sys.stderr)
                print(f"   Returning original image to preserve all content", file=sys.stderr)
            
            time_ms = (time.time() - start_time) * 1000
            
            # Return original image
            lineImg = src.copy()
            h_img, w_img = src.shape[:2]
            corners = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]])
            cv2.rectangle(lineImg, (0, 0), (w_img, h_img), (0, 255, 0), 3)
            cv2.putText(lineImg, f"NO CROP: {skip_reason}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(lineImg, f"Confidence: {confidence:.2f} (threshold: {confidence_threshold:.2f})", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if verification_used:
                status = "PASSED" if verification_passed else "FAILED"
                cv2.putText(lineImg, f"Content Verify: {status}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint = corners
            return src.copy(), lineImg, (topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint), True, time_ms, confidence
        
        # Proceed with intelligent tight cropping
        if debug:
            print(f"\n✅ Proceeding with INTELLIGENT TIGHT CROP", file=sys.stderr)
            print(f"   Confidence: {confidence:.2f}", file=sys.stderr)
        
        # Intelligent margin calculation (adaptive, NOT fixed 5-10%)
        # Analyze edges to determine if we can crop tighter
        edge_margin_pixels = 3  # Minimum pixels to keep from edge
        
        # Check boundary densities to see if we can crop tighter
        # Left edge
        left_strip = mask_cleaned[y:y+h, max(0, x-20):x+5]
        left_density = np.count_nonzero(left_strip) / left_strip.size if left_strip.size > 0 else 0
        
        # Right edge
        right_strip = mask_cleaned[y:y+h, x+w-5:min(mask_cleaned.shape[1], x+w+20)]
        right_density = np.count_nonzero(right_strip) / right_strip.size if right_strip.size > 0 else 0
        
        # Top edge
        top_strip = mask_cleaned[max(0, y-20):y+5, x:x+w]
        top_density = np.count_nonzero(top_strip) / top_strip.size if top_strip.size > 0 else 0
        
        # Bottom edge
        bottom_strip = mask_cleaned[y+h-5:min(mask_cleaned.shape[0], y+h+20), x:x+w]
        bottom_density = np.count_nonzero(bottom_strip) / bottom_strip.size if bottom_strip.size > 0 else 0
        
        if debug:
            print(f"\n✅ Edge Analysis (density of content near boundaries):", file=sys.stderr)
            print(f"   Left: {left_density:.2f}, Right: {right_density:.2f}", file=sys.stderr)
            print(f"   Top: {top_density:.2f}, Bottom: {bottom_density:.2f}", file=sys.stderr)
        
        # Adaptive margins based on edge density
        # If edge density is very low (<0.05), we can crop tighter
        # If edge density is high (>0.20), add small safety margin
        
        margin_left = edge_margin_pixels if left_density < 0.05 else int(w * 0.01) if left_density < 0.20 else int(w * 0.02)
        margin_right = edge_margin_pixels if right_density < 0.05 else int(w * 0.01) if right_density < 0.20 else int(w * 0.02)
        margin_top = edge_margin_pixels if top_density < 0.05 else int(h * 0.01) if top_density < 0.20 else int(h * 0.02)
        margin_bottom = edge_margin_pixels if bottom_density < 0.05 else int(h * 0.01) if bottom_density < 0.20 else int(h * 0.02)
        
        if debug:
            print(f"\n✅ Adaptive Margins Calculated:", file=sys.stderr)
            print(f"   Left: {margin_left}px, Right: {margin_right}px", file=sys.stderr)
            print(f"   Top: {margin_top}px, Bottom: {margin_bottom}px", file=sys.stderr)
            print(f"   (Based on edge content density)", file=sys.stderr)
        
        # Apply intelligent margins
        x_crop = max(0, x - margin_left)
        y_crop = max(0, y - margin_top)
        w_crop = min(src.shape[1] - x_crop, w + margin_left + margin_right)
        h_crop = min(src.shape[0] - y_crop, h + margin_top + margin_bottom)
        
        # Final content safety check
        crop_area_final = w_crop * h_crop
        crop_ratio_final = crop_area_final / img_area
        
        if debug:
            print(f"\n✅ Final Crop Rectangle:", file=sys.stderr)
            print(f"   x={x_crop}, y={y_crop}, w={w_crop}, h={h_crop}", file=sys.stderr)
            print(f"   Final crop area: {crop_ratio_final*100:.1f}% of original", file=sys.stderr)
        
        # Safety check: if we're still removing too much with low confidence, don't crop
        if crop_ratio_final < 0.25 and confidence < 0.55:
            if debug:
                print(f"\n⚠️  FINAL SAFETY CHECK FAILED", file=sys.stderr)
                print(f"   Would remove {(1-crop_ratio_final)*100:.0f}% with confidence {confidence:.2f}", file=sys.stderr)
                print(f"   Returning original to preserve content", file=sys.stderr)
            
            time_ms = (time.time() - start_time) * 1000
            lineImg = src.copy()
            h_img, w_img = src.shape[:2]
            corners = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]])
            cv2.rectangle(lineImg, (0, 0), (w_img, h_img), (255, 0, 0), 3)
            cv2.putText(lineImg, "NO CROP: Safety check", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(lineImg, f"Confidence: {confidence:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint = corners
            return src.copy(), lineImg, (topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint), True, time_ms, confidence
        
        # Apply the intelligent crop
        cropedImg = src[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop].copy()
        
        if debug:
            print(f"\n✅ INTELLIGENT TIGHT CROP APPLIED", file=sys.stderr)
            print(f"   Output shape: {cropedImg.shape}", file=sys.stderr)
            print(f"   Removed: {(1-crop_ratio_final)*100:.1f}% of original", file=sys.stderr)
        
        # Create corners for the bounding box
        corners_ordered = np.array([
            [x_crop, y_crop],  # top-left
            [x_crop + w_crop, y_crop],  # top-right
            [x_crop + w_crop, y_crop + h_crop],  # bottom-right
            [x_crop, y_crop + h_crop]  # bottom-left
        ])
        
        # Create debug visualization
        lineImg = src.copy()
        
        # Draw all significant contours if multiple
        if len(significant_contours) > 1:
            for i, cnt in enumerate(significant_contours):
                color = [(255, 0, 0), (0, 255, 255), (255, 0, 255)][i % 3]
                cv2.drawContours(lineImg, [cnt], -1, color, 2)
        
        # Draw final bounding box
        pt1 = tuple(map(int, corners_ordered[0]))
        pt2 = tuple(map(int, corners_ordered[2]))
        cv2.rectangle(lineImg, pt1, pt2, (0, 255, 0), 3)
        
        # Draw corners
        for i, corner in enumerate(corners_ordered):
            corner_int = tuple(map(int, corner))
            cv2.circle(lineImg, corner_int, 8, (0, 0, 255), -1)
        
        # Add confidence and info text
        cv2.putText(lineImg, f"Confidence: {confidence:.2f} ({confidence*100:.0f}%)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if combined:
            cv2.putText(lineImg, f"Multi-region: {len(significant_contours)} areas", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw mask overlay (lighter for better visibility)
        mask_color = cv2.applyColorMap(mask_cleaned, cv2.COLORMAP_JET)
        lineImg = cv2.addWeighted(lineImg, 0.7, mask_color, 0.3, 0)
        
        # Calculate timing
        time_ms = (time.time() - start_time) * 1000
        
        if debug:
            print(f"\n{'='*70}", file=sys.stderr)
            print(f"U2-Net Method: SUCCESS", file=sys.stderr)
            print(f"Document area: {area_ratio*100:.2f}% of original image", file=sys.stderr)
            print(f"Cropped size: {cropedImg.shape[1]}x{cropedImg.shape[0]}", file=sys.stderr)
            print(f"Confidence: {confidence:.2f} ({confidence*100:.0f}%)", file=sys.stderr)
            print(f"Detection time: {time_ms:.1f}ms", file=sys.stderr)
            print(f"{'='*70}\n", file=sys.stderr)
        
        topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint = corners_ordered
        return cropedImg, lineImg, (topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint), True, time_ms, confidence
        
    except ImportError as e:
        time_ms = (time.time() - start_time) * 1000
        if debug:
            print(f"❌ Import error: {e}", file=sys.stderr)
            print("   Make sure onnxruntime is installed: pip install onnxruntime", file=sys.stderr)
        return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False, time_ms, 0.0
    except Exception as e:
        time_ms = (time.time() - start_time) * 1000
        if debug:
            print(f"❌ U2-Net Method: Error occurred: {str(e)}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False, time_ms, 0.0


def detectDocumentPage_DeepLabV3(src: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, tuple, bool, float, float]:
    """
    Detect document page using DeepLabV3 semantic segmentation.
    
    This method uses a pre-trained DeepLabV3 model for robust segmentation.
    Best for: Complex backgrounds, various lighting conditions
    
    Parameters:
    -----------
    src : np.ndarray
        Input image (BGR format)
    debug : bool
        If True, print debug information
        
    Returns:
    --------
    tuple: (cropped_img, debug_img, corners, success, time_ms, confidence)
    """
    import time
    start_time = time.time()
    
    try:
        import torch
        import torchvision.transforms as T
        from torchvision.models.segmentation import deeplabv3_resnet50
        
        if debug:
            print("DeepLabV3 Method: Using semantic segmentation", file=sys.stderr)
        
        # Load pre-trained model
        try:
            # Try new API first (PyTorch 0.13+)
            from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
            model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        except ImportError:
            # Fallback to old API
            model = deeplabv3_resnet50(pretrained=True)
        model.eval()
        
        # Preprocess image
        preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        input_tensor = preprocess(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        input_batch = input_tensor.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()
        
        # Create binary mask (assuming background class is 0)
        mask = (output_predictions > 0).astype(np.uint8) * 255
        mask = cv2.resize(mask, (src.shape[1], src.shape[0]))
        
        # Morphological operations
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            time_ms = (time.time() - start_time) * 1000
            if debug:
                print("DeepLabV3 Method: No contours found", file=sys.stderr)
            return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False, time_ms, 0.0
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest_contour) / (src.shape[0] * src.shape[1])
        
        if area_ratio < 0.15:
            time_ms = (time.time() - start_time) * 1000
            if debug:
                print(f"DeepLabV3 Method: Area too small ({area_ratio*100:.1f}%)", file=sys.stderr)
            return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False, time_ms, 0.0
        
        # Approximate to quadrilateral
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            corners = approx.reshape(4, 2)
        else:
            x, y, w, h = cv2.boundingRect(largest_contour)
            corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        
        corners_ordered = order_points(corners)
        destination_corners = find_dest(corners_ordered)
        
        M = cv2.getPerspectiveTransform(np.float32(corners_ordered), np.float32(destination_corners))
        cropedImg = cv2.warpPerspective(src, M, (destination_corners[2][0], destination_corners[2][1]),
                                       flags=cv2.INTER_LINEAR)
        
        # Create debug visualization
        lineImg = src.copy()
        cv2.polylines(lineImg, [corners_ordered], True, (255, 0, 255), 3)
        for corner in corners_ordered:
            cv2.circle(lineImg, tuple(corner), 8, (255, 0, 0), -1)
        
        time_ms = (time.time() - start_time) * 1000
        confidence = 0.6  # Default confidence for DeepLabV3
        
        if debug:
            print(f"DeepLabV3 Method: Success (area: {area_ratio*100:.1f}%)", file=sys.stderr)
            print(f"Detection time: {time_ms:.1f}ms", file=sys.stderr)
        
        topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint = corners_ordered
        return cropedImg, lineImg, (topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint), True, time_ms, confidence
        
    except Exception as e:
        time_ms = (time.time() - start_time) * 1000
        if debug:
            print(f"DeepLabV3 Method: Error - {str(e)}", file=sys.stderr)
        return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False, time_ms, 0.0


def detectDocumentPage_ContentBased(src: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, tuple, bool, float, float]:
    """
    Detect document page by finding the content-rich region.
    
    This method looks for where text/content is concentrated.
    Best for: Real-world scanned documents with margins
    
    Parameters:
    -----------
    src : np.ndarray
        Input image (BGR format)
    debug : bool
        If True, print debug information
        
    Returns:
    --------
    tuple: (cropped_img, debug_img, corners, success, time_ms, confidence)
    """
    import time
    start_time = time.time()
    
    try:
        if debug:
            print("Content-Based Method: Processing", file=sys.stderr)
        
        # Convert to grayscale
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        
        # Invert to make text white
        inv = 255 - gray
        
        # Threshold to find dark content (text, etc)
        _, binary = cv2.threshold(inv, 40, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to connect nearby text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilated = cv2.dilate(binary, kernel, iterations=3)
        
        # Find projection profiles (horizontal and vertical)
        h_projection = np.sum(dilated, axis=1)
        v_projection = np.sum(dilated, axis=0)
        
        # Find content boundaries with adaptive threshold
        # Use a higher threshold to find more concentrated content
        h_threshold = np.max(h_projection) * 0.25
        v_threshold = np.max(v_projection) * 0.25
        
        # Find top and bottom
        top = 0
        for i in range(len(h_projection)):
            if h_projection[i] > h_threshold:
                top = i
                break
        
        bottom = len(h_projection) - 1
        for i in range(len(h_projection) - 1, -1, -1):
            if h_projection[i] > h_threshold:
                bottom = i
                break
        
        # Find left and right
        left = 0
        for i in range(len(v_projection)):
            if v_projection[i] > v_threshold:
                left = i
                break
        
        right = len(v_projection) - 1
        for i in range(len(v_projection) - 1, -1, -1):
            if v_projection[i] > v_threshold:
                right = i
                break
        
        # Add small margin (2%)
        margin_h = int((bottom - top) * 0.02)
        margin_w = int((right - left) * 0.02)
        
        top = max(0, top - margin_h)
        bottom = min(src.shape[0] - 1, bottom + margin_h)
        left = max(0, left - margin_w)
        right = min(src.shape[1] - 1, right + margin_w)
        
        # Check if detection is valid
        width = right - left
        height = bottom - top
        area_ratio = (width * height) / (src.shape[0] * src.shape[1])
        
        if area_ratio < 0.1 or area_ratio > 0.95:
            time_ms = (time.time() - start_time) * 1000
            if debug:
                print(f"Content-Based Method: Invalid area ratio {area_ratio:.2f}", file=sys.stderr)
            return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False, time_ms, 0.0
        
        # Create corners
        corners = np.array([
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom]
        ], dtype=np.int32)
        
        # Crop the image
        cropped = src[top:bottom, left:right]
        
        # Create debug visualization
        lineImg = src.copy()
        cv2.rectangle(lineImg, (left, top), (right, bottom), (0, 255, 0), 3)
        for corner in corners:
            cv2.circle(lineImg, tuple(corner), 8, (0, 0, 255), -1)
        
        time_ms = (time.time() - start_time) * 1000
        confidence = 0.5  # Default confidence for Content-Based
        
        if debug:
            print(f"Content-Based Method: Success (area: {area_ratio*100:.1f}%)", file=sys.stderr)
            print(f"  Bounds: x={left}-{right}, y={top}-{bottom}", file=sys.stderr)
            print(f"Detection time: {time_ms:.1f}ms", file=sys.stderr)
        
        topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint = corners
        return cropped, lineImg, (topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint), True, time_ms, confidence
        
    except Exception as e:
        time_ms = (time.time() - start_time) * 1000
        if debug:
            print(f"Content-Based Method: Error - {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False, time_ms, 0.0


def detectAndCropDocumentPage(src: np.ndarray, method: str = 'u2net', model_name: str = 'u2netp', debug: bool = False,
                              use_content_verify: bool = False, confidence_threshold: float = 0.85,
                              source_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, tuple, str, float, float]:
    """
    Main function to detect and crop document page from image.
    
    V4 Final Unified Logic: Both U2-Net and YOLO use same threshold (0.85) and verification logic.
    
    Automatically selects the best method or uses specified method.
    Uses simple bounding box cropping (NO perspective transform/dewarp).
    
    Parameters:
    -----------
    src : np.ndarray
        Input image (BGR format)
    method : str
        Detection method: 'u2net', 'yolo', 'content' (auto mode removed in V4)
    model_name : str
        For U2-Net: 'u2netp' (4.4MB, only option in V4 Final)
        For YOLO: 'yolov11s-seg' (22MB, default), 'yolov11n-seg' (11MB), or custom model path
    debug : bool
        If True, print debug information
    use_content_verify : bool
        If True, use content-based verification for medium confidence (60-85%)
    confidence_threshold : float
        Threshold for auto-crop decision (default 0.85, unified for both models)
        
    Returns:
    --------
    tuple: (cropped_img, debug_img, corners, method_used, time_ms, confidence)
        cropped_img: Cropped document image (bounding box, no warp)
        debug_img: Visualization of detection process
        corners: Tuple of (tl, tr, br, bl) corner points
        method_used: String indicating which method was used
        time_ms: Detection time in milliseconds
        confidence: Confidence score (0-1) for detection quality
    """
    if debug:
        import sys
        print(f"\n=== Document Page Detection ===", file=sys.stderr)
        print(f"Input image size: {src.shape[1]}x{src.shape[0]}", file=sys.stderr)
        print(f"Requested method: {method}", file=sys.stderr)
        if source_path:
            print(f"Processing file: {source_path}", file=sys.stderr)
        if method == 'u2net' or method == 'auto':
            print(f"U2-Net model: {model_name}", file=sys.stderr)
    
    cropped = None
    debugImg = None
    corners = ([], [], [], [])
    method_used = "none"
    success = False
    time_ms = 0.0
    confidence = 0.0
    
    # Method selection - U2-Net, YOLO, and Content-Based supported (DeepLabV3 removed, auto mode removed in V4)
    if method == 'u2net':
        cropped, debugImg, corners, success, time_ms, confidence = detectDocumentPage_U2Net(
            src, model_name=model_name, debug=debug, 
            use_content_verify=use_content_verify, confidence_threshold=confidence_threshold,
            source_path=source_path
        )
        method_used = "u2net" if success else "u2net_failed"
    elif method == 'yolo':
        # Try YOLO detection
        from kukalib.yolo_detector import detectDocumentPage_YOLO
        # Use YOLO model - default to ONNX format for speed
        yolo_model = f"{model_name}.onnx" if not model_name.endswith('.onnx') and not model_name.endswith('.pt') else model_name
        cropped, debugImg, corners, success, time_ms, confidence = detectDocumentPage_YOLO(src, model_name=yolo_model, confidence_threshold=confidence_threshold, debug=debug, source_path=source_path)
        method_used = "yolo" if success else "yolo_failed"
    elif method == 'content':
        cropped, debugImg, corners, success, time_ms, confidence = detectDocumentPage_ContentBased(src, debug=debug)
        method_used = "content" if success else "content_failed"
    else:  # fallback for any other method name
        # V4: Priority order: U2-Net → YOLO → Content (DeepLabV3 removed)
        # Focus on U2-Net and YOLO as primary AI methods
        
        # 1. Try U2-Net (PRIMARY - AI-based, best for real-world documents)
        cropped, debugImg, corners, success, time_ms, confidence = detectDocumentPage_U2Net(src, model_name=model_name, debug=debug, source_path=source_path)
        if success and confidence > 0.5:
            method_used = "u2net"
        else:
            # 2. Try YOLO (FAST - good for multi-page and CPU)
            if debug:
                print("\nAuto: U2-Net low confidence or failed, trying YOLO method", file=sys.stderr)
            from kukalib.yolo_detector import detectDocumentPage_YOLO
            cropped_yolo, debugImg_yolo, corners_yolo, success_yolo, time_ms_yolo, confidence_yolo = detectDocumentPage_YOLO(src, model_name='yolov11n-seg.onnx', confidence_threshold=confidence_threshold, debug=debug, source_path=source_path)
            
            # Use YOLO if better confidence
            if success_yolo and confidence_yolo > confidence:
                cropped, debugImg, corners, success, time_ms, confidence = cropped_yolo, debugImg_yolo, corners_yolo, success_yolo, time_ms_yolo, confidence_yolo
                method_used = "yolo"
            elif success:
                method_used = "u2net"  # Keep U2-Net result
            else:
                # 3. Try Content-Based (final fallback for docs with clear margins)
                if debug:
                    print("\nAuto: Both U2-Net and YOLO failed, trying Content-Based method", file=sys.stderr)
                cropped, debugImg, corners, success, time_ms, confidence = detectDocumentPage_ContentBased(src, debug=debug)
                if success:
                    method_used = "content"
                else:
                    method_used = "failed"
    
    # If detection failed, return original image
    if not success or cropped is None or cropped.size == 0:
        if debug:
            print("\n⚠️  Detection failed, returning original image", file=sys.stderr)
        cropped = src.copy()
        debugImg = src.copy()
        corners = ([], [], [], [])
        method_used = "failed"
        time_ms = 0.0
        confidence = 0.0
    
    if debug:
        import sys
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"Final method used: {method_used}", file=sys.stderr)
        print(f"Output image size: {cropped.shape[1]}x{cropped.shape[0]}", file=sys.stderr)
        print(f"Detection time: {time_ms:.1f}ms", file=sys.stderr)
        print(f"Confidence: {confidence:.2f} ({confidence*100:.0f}%)", file=sys.stderr)
        print(f"={'='*70}", file=sys.stderr)
        print("=== Detection Complete ===\n", file=sys.stderr)
    
    return cropped, debugImg, corners, method_used, time_ms, confidence


def convertPdfToImages(pdf_path: str, output_dir: Optional[str] = None, dpi: int = 200) -> List[np.ndarray]:
    """
    Convert PDF file to list of images (one per page).
    
    Requires PyMuPDF (fitz) to be installed: pip install PyMuPDF
    
    Parameters:
    -----------
    pdf_path : str
        Path to PDF file
    output_dir : str, optional
        If provided, save images to this directory
    dpi : int
        Resolution for rendering PDF pages (default: 200)
        
    Returns:
    --------
    list: List of numpy arrays (images in BGR format). 
          Empty list if PyMuPDF is not installed or conversion fails.
    """
    try:
        import fitz  # PyMuPDF
        
        images = []
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Render page to image
            mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 is default DPI
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to numpy array
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            images.append(img)
            
            # Save if output directory specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"page_{page_num + 1}.jpg")
                cv2.imwrite(output_path, img)
        
        pdf_document.close()
        return images
        
    except ImportError:
        print("Error: PyMuPDF (fitz) not installed. Install with: pip install PyMuPDF")
        return []
    except Exception as e:
        print(f"Error converting PDF: {str(e)}")
        return []


def processPdfDocument(pdf_path: str, output_dir: str, method: str = 'auto', model_name: str = 'u2net', debug: bool = False) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Process all pages in a PDF document.
    
    Parameters:
    -----------
    pdf_path : str
        Path to PDF file
    output_dir : str
        Directory to save processed images
    method : str
        Detection method: 'auto', 'u2net', 'deeplabv3', 'content'
    model_name : str
        For U2-Net: 'u2net' (176MB, default) or 'u2netp' (4.4MB)
    debug : bool
        If True, print debug information
        
    Returns:
    --------
    list: List of (cropped_image, debug_image, time_ms) tuples for each page
    """
    # Convert PDF to images
    images = convertPdfToImages(pdf_path)
    
    if not images:
        print(f"Failed to convert PDF: {pdf_path}")
        return []
    
    results = []
    os.makedirs(output_dir, exist_ok=True)
    total_time = 0.0
    
    for i, img in enumerate(images):
        if debug:
            print(f"\nProcessing page {i + 1}/{len(images)}...")
        
        # Detect and crop
        cropped, debugImg, corners, method_used, time_ms, confidence = detectAndCropDocumentPage(
            img, method=method, model_name=model_name, debug=debug
        )
        
        total_time += time_ms
        
        # Save results
        cropped_path = os.path.join(output_dir, f"page_{i + 1}_cropped.jpg")
        debug_path = os.path.join(output_dir, f"page_{i + 1}_debug.jpg")
        
        cv2.imwrite(cropped_path, cropped)
        cv2.imwrite(debug_path, debugImg)
        
        results.append((cropped, debugImg, time_ms, confidence))
        
        if debug:
            print(f"Saved: {cropped_path} ({time_ms:.1f}ms, confidence: {confidence:.2f})")
    
    if debug:
        print(f"\nTotal detection time: {total_time:.1f}ms ({total_time/1000:.2f}s)")
        print(f"Average per page: {total_time/len(images):.1f}ms")
    
    return results


if __name__ == '__main__':
    import time
    
    if len(sys.argv) < 2:
        print('Usage: python doc_page_crop.py [image_path or pdf_path] [method] [model]')
        print('Methods: auto (default), u2net, deeplabv3, content')
        print('Models (for u2net): u2net (176MB, default), u2netp (4.4MB)')
        sys.exit()
    
    path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'auto'
    model_name = sys.argv[3] if len(sys.argv) > 3 else 'u2netp'  # V3: Default to lightweight model
    
    if not os.path.exists(path):
        print(f'Error: Path does not exist: {path}')
        sys.exit()
    
    # Check if PDF or image
    if path.lower().endswith('.pdf'):
        print(f"Processing PDF: {path}")
        output_dir = os.path.join(os.path.dirname(path), "output_pdf")
        results = processPdfDocument(path, output_dir, method=method, model_name=model_name, debug=True)
        print(f"\nProcessed {len(results)} pages")
        print(f"Results saved to: {output_dir}")
    else:
        # Process single image
        src = cv2.imread(path)
        
        if src is None:
            print(f'Error: Failed to load image: {path}')
            sys.exit()
        
        print(f"Processing image: {path}")
        
        cropped, debugImg, corners, method_used, time_ms, confidence = detectAndCropDocumentPage(
            src, method=method, model_name=model_name, debug=True
        )
        
        # Save results
        output_dir = os.path.dirname(path)
        base_name = os.path.splitext(os.path.basename(path))[0]
        
        cropped_path = os.path.join(output_dir, f"{base_name}_cropped.jpg")
        debug_path = os.path.join(output_dir, f"{base_name}_debug.jpg")
        
        cv2.imwrite(cropped_path, cropped)
        cv2.imwrite(debug_path, debugImg)
        
        print(f"\nProcessing completed!")
        print(f"Method used: {method_used}")
        print(f"Detection time: {time_ms:.1f}ms")
        print(f"Confidence: {confidence:.2f} ({confidence*100:.0f}%)")
        print(f"Cropped image saved to: {cropped_path}")
        print(f"Debug image saved to: {debug_path}")
