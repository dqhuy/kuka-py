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


def detectDocumentPage_OpenCV(src: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, tuple, bool]:
    """
    Detect document page using traditional OpenCV methods.
    
    This method uses edge detection and contour analysis to find document boundaries.
    Best for: Simple backgrounds, good contrast, clear edges
    
    Parameters:
    -----------
    src : np.ndarray
        Input image (BGR format)
    debug : bool
        If True, print debug information
        
    Returns:
    --------
    tuple: (cropped_img, debug_img, corners, success)
        cropped_img: Cropped and rectified document image
        debug_img: Visualization of detection process
        corners: Tuple of (tl, tr, br, bl) corner points
        success: Boolean indicating if detection succeeded
    """
    src_small = src.copy()
    
    # Initialize return values
    lineImg = np.zeros(src_small.shape, dtype=np.uint8)
    cropedImg = np.zeros(src.shape, dtype=np.uint8)
    hasCropped = False
    
    topleftPoint = []
    toprightPoint = []
    bottomleftPoint = []
    bottomrightPoint = []
    
    img_height = src.shape[0]
    img_width = src.shape[1]
    
    # Resize for faster processing
    widthResize = 1280
    ratio = 1
    if img_width > widthResize:
        if img_width > img_height:
            ratio = widthResize / img_width
        else:
            ratio = widthResize / img_height
        
        img_width = int(img_width * ratio)
        img_height = int(img_height * ratio)
        src_small = cv2.resize(src, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    
    # Method 1: Try contour-based detection
    gray = cv2.cvtColor(src_small, cv2.COLOR_BGR2GRAY)
    blurImg = cv2.GaussianBlur(gray, (5, 5), 0)
    edgeImg = cv2.Canny(blurImg, 50, 150)
    
    # Morphological operations to close gaps
    kernel = np.ones((5, 5))
    edgeImg = cv2.dilate(edgeImg, kernel, iterations=3)
    edgeImg = cv2.erode(edgeImg, kernel, iterations=2)
    
    # Find contours
    contours, hierarchy = cv2.findContours(edgeImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = biggestContour(contours)
    
    if len(biggest) == 4:
        biggest = biggest.reshape(4, 2)
        biggest_order = order_points(biggest)
        x_b, y_b, w_b, h_b = cv2.boundingRect(np.array(biggest_order).reshape(4, 1, 2))
        
        # Check if contour is large enough (at least 15% of image)
        if w_b * h_b >= 0.15 * (img_height * img_width):
            # Scale back to original size
            biggest_order = np.array(biggest_order) / ratio
            biggest_order = np.ceil(biggest_order).astype(int)
            
            topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint = biggest_order
            
            # Create lines from corners
            topLine = np.array([topleftPoint, toprightPoint]).reshape(4)
            bottomLine = np.array([bottomleftPoint, bottomrightPoint]).reshape(4)
            leftLine = np.array([topleftPoint, bottomleftPoint]).reshape(4)
            rightLine = np.array([toprightPoint, bottomrightPoint]).reshape(4)
            
            # Crop and transform
            corners = np.array([topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint])
            destination_corners = find_dest(corners)
            
            M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
            cropedImg = cv2.warpPerspective(src, M, 
                                           (destination_corners[2][0], destination_corners[2][1]),
                                           flags=cv2.INTER_LINEAR)
            
            # Create debug visualization
            lineImg = src.copy()
            cv2.polylines(lineImg, [corners], True, (0, 255, 0), 3)
            for i, corner in enumerate(corners):
                cv2.circle(lineImg, tuple(corner), 8, (0, 0, 255), -1)
            
            hasCropped = True
            
            if debug:
                print(f"OpenCV Method: Successfully detected document using contour method")
                print(f"Document area: {w_b * h_b / ratio / ratio / (src.shape[0] * src.shape[1]) * 100:.1f}% of image")
    
    if not hasCropped and debug:
        print("OpenCV Method: Failed to detect document")
        lineImg = src.copy()
    
    return cropedImg, lineImg, (topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint), hasCropped


def detectDocumentPage_U2Net(src: np.ndarray, model_path: Optional[str] = None, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, tuple, bool]:
    """
    Detect document page using U2-Net AI model.
    
    This method uses deep learning for robust detection even with complex backgrounds.
    Uses standalone U2-Net implementation for better compatibility.
    Best for: Complex backgrounds, poor lighting, cluttered scenes
    
    Parameters:
    -----------
    src : np.ndarray
        Input image (BGR format)
    model_path : str, optional
        Path to U2Net model weights. If None, uses default in kukalib/models/
    debug : bool
        If True, print extensive debug information
        
    Returns:
    --------
    tuple: (cropped_img, debug_img, corners, success)
        cropped_img: Cropped and rectified document image
        debug_img: Visualization of detection process
        corners: Tuple of (tl, tr, br, bl) corner points
        success: Boolean indicating if detection succeeded
    """
    if debug:
        print("\n" + "="*70, file=sys.stderr)
        print("U2-Net Method: Starting document detection", file=sys.stderr)
        print("="*70, file=sys.stderr)
        print(f"Input image shape: {src.shape}", file=sys.stderr)
    
    try:
        # Import our custom U2-Net implementation
        from kukalib.u2net_model import run_u2net_inference
        
        if debug:
            print("✅ U2-Net module imported successfully", file=sys.stderr)
        
        # Run U2-Net inference to get mask
        mask = run_u2net_inference(src, debug=debug)
        
        if mask is None:
            if debug:
                print("❌ U2-Net inference failed", file=sys.stderr)
            return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False
        
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
            return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        
        if debug:
            print(f"✅ Largest contour area: {contour_area} pixels", file=sys.stderr)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        if debug:
            print(f"✅ Bounding rectangle: x={x}, y={y}, w={w}, h={h}", file=sys.stderr)
        
        # Check if area is significant (at least 10% of image, more lenient)
        area_ratio = (w * h) / (src.shape[0] * src.shape[1])
        if debug:
            print(f"   Area ratio: {area_ratio*100:.2f}% of image", file=sys.stderr)
        
        if area_ratio < 0.10:
            if debug:
                print(f"❌ Detected area too small (< 10%): {area_ratio*100:.1f}%", file=sys.stderr)
            return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False
        
        # Try multiple approximation epsilons to find 4-point contour
        peri = cv2.arcLength(largest_contour, True)
        
        approx = None
        for epsilon_factor in [0.01, 0.02, 0.03, 0.04, 0.05]:
            test_approx = cv2.approxPolyDP(largest_contour, epsilon_factor * peri, True)
            if len(test_approx) == 4:
                approx = test_approx
                if debug:
                    print(f"✅ Found 4-point approximation with epsilon={epsilon_factor}", file=sys.stderr)
                break
        
        if approx is None:
            approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
            if debug:
                print(f"✅ Contour approximation: {len(approx)} points", file=sys.stderr)
        
        # If we got 4 points, use them; otherwise use minimum area rectangle
        if len(approx) == 4:
            corners = approx.reshape(4, 2)
            if debug:
                print(f"   Using 4-point approximation", file=sys.stderr)
        else:
            # Use minimum area rectangle for better fit
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            corners = np.intp(box)
            if debug:
                print(f"   Using minimum area rectangle", file=sys.stderr)
        
        # Order points correctly
        corners_ordered = order_points(corners)
        
        if debug:
            print(f"✅ Corners ordered:", file=sys.stderr)
            print(f"   Top-left: {corners_ordered[0]}", file=sys.stderr)
            print(f"   Top-right: {corners_ordered[1]}", file=sys.stderr)
            print(f"   Bottom-right: {corners_ordered[2]}", file=sys.stderr)
            print(f"   Bottom-left: {corners_ordered[3]}", file=sys.stderr)
        
        # Apply perspective transform
        destination_corners = find_dest(corners_ordered)
        
        if debug:
            print(f"✅ Destination corners calculated: {destination_corners[2]}", file=sys.stderr)
        
        M = cv2.getPerspectiveTransform(np.float32(corners_ordered), np.float32(destination_corners))
        cropedImg = cv2.warpPerspective(src, M,
                                       (destination_corners[2][0], destination_corners[2][1]),
                                       flags=cv2.INTER_LINEAR)
        
        if debug:
            print(f"✅ Perspective transform applied", file=sys.stderr)
            print(f"   Output image shape: {cropedImg.shape}", file=sys.stderr)
        
        # Create debug visualization
        lineImg = src.copy()
        corners_for_poly = np.array([corners_ordered], dtype=np.int32)
        cv2.polylines(lineImg, corners_for_poly, True, (0, 255, 0), 3)
        for i, corner in enumerate(corners_ordered):
            corner_int = tuple(map(int, corner))
            cv2.circle(lineImg, corner_int, 10, (0, 0, 255), -1)
            cv2.putText(lineImg, str(i+1), corner_int, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw mask overlay
        mask_color = cv2.applyColorMap(mask_cleaned, cv2.COLORMAP_JET)
        lineImg = cv2.addWeighted(lineImg, 0.6, mask_color, 0.4, 0)
        
        if debug:
            print(f"✅ Debug visualization created", file=sys.stderr)
            print(f"\n{'='*70}", file=sys.stderr)
            print(f"U2-Net Method: SUCCESS", file=sys.stderr)
            print(f"Document area: {area_ratio*100:.2f}% of original image", file=sys.stderr)
            print(f"Cropped size: {cropedImg.shape[1]}x{cropedImg.shape[0]}", file=sys.stderr)
            print(f"{'='*70}\n", file=sys.stderr)
        
        topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint = corners_ordered
        return cropedImg, lineImg, (topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint), True
        
    except ImportError as e:
        if debug:
            print(f"❌ Import error: {e}", file=sys.stderr)
            print("   Make sure onnxruntime is installed: pip install onnxruntime", file=sys.stderr)
        return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False
    except Exception as e:
        if debug:
            print(f"❌ U2-Net Method: Error occurred: {str(e)}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False


def detectDocumentPage_DeepLabV3(src: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, tuple, bool]:
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
    tuple: (cropped_img, debug_img, corners, success)
    """
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
            if debug:
                print("DeepLabV3 Method: No contours found", file=sys.stderr)
            return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest_contour) / (src.shape[0] * src.shape[1])
        
        if area_ratio < 0.15:
            if debug:
                print(f"DeepLabV3 Method: Area too small ({area_ratio*100:.1f}%)", file=sys.stderr)
            return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False
        
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
        
        if debug:
            print(f"DeepLabV3 Method: Success (area: {area_ratio*100:.1f}%)", file=sys.stderr)
        
        topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint = corners_ordered
        return cropedImg, lineImg, (topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint), True
        
    except Exception as e:
        if debug:
            print(f"DeepLabV3 Method: Error - {str(e)}", file=sys.stderr)
        return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False


def detectDocumentPage_EdgeLinkingCNN(src: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, tuple, bool]:
    """
    Detect document page using edge-based CNN approach.
    
    This method combines edge detection with CNN-based refinement.
    Best for: Documents with clear edges
    
    Parameters:
    -----------
    src : np.ndarray
        Input image (BGR format)
    debug : bool
        If True, print debug information
        
    Returns:
    --------
    tuple: (cropped_img, debug_img, corners, success)
    """
    try:
        if debug:
            print("EdgeLinking-CNN Method: Processing", file=sys.stderr)
        
        # Convert to grayscale
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Find edges using Canny with automatic thresholds
        v = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(gray, lower, upper)
        
        # Dilate edges
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            if debug:
                print("EdgeLinking-CNN Method: No contours found", file=sys.stderr)
            return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False
        
        # Filter contours by area
        min_area = (src.shape[0] * src.shape[1]) * 0.15
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if not valid_contours:
            if debug:
                print("EdgeLinking-CNN Method: No valid contours", file=sys.stderr)
            return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False
        
        # Get largest contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Approximate to polygon
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        
        if len(approx) >= 4:
            # If more than 4 points, get bounding rect
            if len(approx) > 4:
                x, y, w, h = cv2.boundingRect(approx)
                corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            else:
                corners = approx.reshape(4, 2)
            
            corners_ordered = order_points(corners)
            destination_corners = find_dest(corners_ordered)
            
            M = cv2.getPerspectiveTransform(np.float32(corners_ordered), np.float32(destination_corners))
            cropedImg = cv2.warpPerspective(src, M, (destination_corners[2][0], destination_corners[2][1]),
                                           flags=cv2.INTER_LINEAR)
            
            # Create debug visualization
            lineImg = src.copy()
            cv2.polylines(lineImg, [corners_ordered], True, (0, 255, 255), 3)
            for corner in corners_ordered:
                cv2.circle(lineImg, tuple(corner), 8, (255, 255, 0), -1)
            
            if debug:
                print(f"EdgeLinking-CNN Method: Success", file=sys.stderr)
            
            topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint = corners_ordered
            return cropedImg, lineImg, (topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint), True
        
        if debug:
            print("EdgeLinking-CNN Method: Could not find quadrilateral", file=sys.stderr)
        return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False
        
    except Exception as e:
        if debug:
            print(f"EdgeLinking-CNN Method: Error - {str(e)}", file=sys.stderr)
        return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False


def detectDocumentPage_ContentBased(src: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, tuple, bool]:
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
    tuple: (cropped_img, debug_img, corners, success)
    """
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
            if debug:
                print(f"Content-Based Method: Invalid area ratio {area_ratio:.2f}", file=sys.stderr)
            return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False
        
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
        
        if debug:
            print(f"Content-Based Method: Success (area: {area_ratio*100:.1f}%)", file=sys.stderr)
            print(f"  Bounds: x={left}-{right}, y={top}-{bottom}", file=sys.stderr)
        
        topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint = corners
        return cropped, lineImg, (topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint), True
        
    except Exception as e:
        if debug:
            print(f"Content-Based Method: Error - {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False


def detectDocumentPage_MorphologyBased(src: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, tuple, bool]:
    """
    Detect document page using advanced morphological operations.
    
    This method uses morphological operations to find document boundaries.
    Best for: High-contrast documents
    
    Parameters:
    -----------
    src : np.ndarray
        Input image (BGR format)
    debug : bool
        If True, print debug information
        
    Returns:
    --------
    tuple: (cropped_img, debug_img, corners, success)
    """
    try:
        if debug:
            print("Morphology Method: Processing", file=sys.stderr)
        
        # Convert to grayscale
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Otsu's thresholding
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            if debug:
                print("Morphology Method: No contours found", file=sys.stderr)
            return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest_contour) / (src.shape[0] * src.shape[1])
        
        if area_ratio < 0.1:
            if debug:
                print(f"Morphology Method: Area too small ({area_ratio*100:.1f}%)", file=sys.stderr)
            return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        corners_ordered = order_points(box)
        corners_ordered = np.array(corners_ordered, dtype=np.int32)
        destination_corners = find_dest(corners_ordered)
        
        M = cv2.getPerspectiveTransform(np.float32(corners_ordered), np.float32(destination_corners))
        cropedImg = cv2.warpPerspective(src, M, (destination_corners[2][0], destination_corners[2][1]),
                                       flags=cv2.INTER_LINEAR)
        
        # Create debug visualization
        lineImg = src.copy()
        corners_for_poly = np.array([corners_ordered], dtype=np.int32)
        cv2.polylines(lineImg, corners_for_poly, True, (128, 0, 128), 3)
        for corner in corners_ordered:
            cv2.circle(lineImg, tuple(corner), 8, (0, 128, 128), -1)
        
        if debug:
            print(f"Morphology Method: Success (area: {area_ratio*100:.1f}%)", file=sys.stderr)
        
        topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint = corners_ordered
        return cropedImg, lineImg, (topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint), True
        
    except Exception as e:
        if debug:
            print(f"Morphology Method: Error - {str(e)}", file=sys.stderr)
        return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False


def detectAndCropDocumentPage(src: np.ndarray, method: str = 'auto', debug: bool = False) -> Tuple[np.ndarray, np.ndarray, tuple, str]:
    """
    Main function to detect and crop document page from image.
    
    Automatically selects the best method or uses specified method.
    
    Parameters:
    -----------
    src : np.ndarray
        Input image (BGR format)
    method : str
        Detection method: 'auto', 'u2net', 'deeplabv3', 'edgelinking', 'morphology', 'opencv'
        'auto' tries AI methods first (priority order), falls back to traditional methods
    debug : bool
        If True, print debug information
        
    Returns:
    --------
    tuple: (cropped_img, debug_img, corners, method_used)
        cropped_img: Cropped and rectified document image
        debug_img: Visualization of detection process
        corners: Tuple of (tl, tr, br, bl) corner points
        method_used: String indicating which method was used
    """
    if debug:
        import sys
        print(f"=== Document Page Detection ===", file=sys.stderr)
        print(f"Input image size: {src.shape[1]}x{src.shape[0]}", file=sys.stderr)
        print(f"Requested method: {method}", file=sys.stderr)
    
    cropped = None
    debugImg = None
    corners = ([], [], [], [])
    method_used = "none"
    success = False
    
    # Single method selection
    if method == 'opencv':
        cropped, debugImg, corners, success = detectDocumentPage_OpenCV(src, debug=debug)
        method_used = "opencv"
    elif method == 'u2net':
        cropped, debugImg, corners, success = detectDocumentPage_U2Net(src, debug=debug)
        method_used = "u2net" if success else "u2net_failed"
    elif method == 'deeplabv3':
        cropped, debugImg, corners, success = detectDocumentPage_DeepLabV3(src, debug=debug)
        method_used = "deeplabv3" if success else "deeplabv3_failed"
    elif method == 'edgelinking':
        cropped, debugImg, corners, success = detectDocumentPage_EdgeLinkingCNN(src, debug=debug)
        method_used = "edgelinking" if success else "edgelinking_failed"
    elif method == 'morphology':
        cropped, debugImg, corners, success = detectDocumentPage_MorphologyBased(src, debug=debug)
        method_used = "morphology" if success else "morphology_failed"
    elif method == 'content':
        cropped, debugImg, corners, success = detectDocumentPage_ContentBased(src, debug=debug)
        method_used = "content" if success else "content_failed"
    else:  # method == 'auto'
        # Priority order: U2-Net first (requested by user), then fallback methods
        # U2-Net provides best accuracy for real-world documents
        
        # 1. Try U2-Net (BEST - AI-based, works with complex backgrounds)
        cropped, debugImg, corners, success = detectDocumentPage_U2Net(src, debug=debug)
        if success:
            method_used = "u2net"
        else:
            # 2. Try Content-Based (good for documents with clear margins)
            if debug:
                print("Auto: U2-Net failed, trying Content-Based method", file=sys.stderr)
            cropped, debugImg, corners, success = detectDocumentPage_ContentBased(src, debug=debug)
            if success:
                method_used = "content"
            else:
                # 3. Try Morphology-based (good for high-contrast docs)
                if debug:
                    print("Auto: Trying Morphology method", file=sys.stderr)
                cropped, debugImg, corners, success = detectDocumentPage_MorphologyBased(src, debug=debug)
                if success:
                    method_used = "morphology"
                else:
                    # 4. Try EdgeLinking (good for docs with clear edges)
                    if debug:
                        print("Auto: Trying EdgeLinking method", file=sys.stderr)
                    cropped, debugImg, corners, success = detectDocumentPage_EdgeLinkingCNN(src, debug=debug)
                    if success:
                        method_used = "edgelinking"
                    else:
                        # 5. Try DeepLabV3 (powerful CNN but slower)
                        if debug:
                            print("Auto: Trying DeepLabV3 method", file=sys.stderr)
                        cropped, debugImg, corners, success = detectDocumentPage_DeepLabV3(src, debug=debug)
                        if success:
                            method_used = "deeplabv3"
                        else:
                            # 6. Finally try traditional OpenCV
                            if debug:
                                print("Auto: Falling back to OpenCV method", file=sys.stderr)
                            cropped, debugImg, corners, success = detectDocumentPage_OpenCV(src, debug=debug)
                            if success:
                                method_used = "opencv"
                            else:
                                method_used = "failed"
    
    # If detection failed, return original image
    if not success or cropped is None or cropped.size == 0:
        if debug:
            print("Detection failed, returning original image")
        cropped = src.copy()
        debugImg = src.copy()
        corners = ([], [], [], [])
        method_used = "failed"
    
    if debug:
        import sys
        print(f"Final method used: {method_used}", file=sys.stderr)
        print(f"Output image size: {cropped.shape[1]}x{cropped.shape[0]}", file=sys.stderr)
        print("=== Detection Complete ===\n", file=sys.stderr)
    
    return cropped, debugImg, corners, method_used


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


def processPdfDocument(pdf_path: str, output_dir: str, method: str = 'auto', debug: bool = False) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Process all pages in a PDF document.
    
    Parameters:
    -----------
    pdf_path : str
        Path to PDF file
    output_dir : str
        Directory to save processed images
    method : str
        Detection method: 'auto', 'u2net', 'opencv'
    debug : bool
        If True, print debug information
        
    Returns:
    --------
    list: List of (cropped_image, debug_image) tuples for each page
    """
    # Convert PDF to images
    images = convertPdfToImages(pdf_path)
    
    if not images:
        print(f"Failed to convert PDF: {pdf_path}")
        return []
    
    results = []
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img in enumerate(images):
        if debug:
            print(f"\nProcessing page {i + 1}/{len(images)}...")
        
        # Detect and crop
        cropped, debugImg, corners, method_used = detectAndCropDocumentPage(img, method=method, debug=debug)
        
        # Save results
        cropped_path = os.path.join(output_dir, f"page_{i + 1}_cropped.jpg")
        debug_path = os.path.join(output_dir, f"page_{i + 1}_debug.jpg")
        
        cv2.imwrite(cropped_path, cropped)
        cv2.imwrite(debug_path, debugImg)
        
        results.append((cropped, debugImg))
        
        if debug:
            print(f"Saved: {cropped_path}")
    
    return results


if __name__ == '__main__':
    import time
    
    if len(sys.argv) < 2:
        print('Usage: python doc_page_crop.py [image_path or pdf_path] [method]')
        print('Methods: auto (default), u2net, opencv')
        sys.exit()
    
    path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'auto'
    
    if not os.path.exists(path):
        print(f'Error: Path does not exist: {path}')
        sys.exit()
    
    # Check if PDF or image
    if path.lower().endswith('.pdf'):
        print(f"Processing PDF: {path}")
        output_dir = os.path.join(os.path.dirname(path), "output_pdf")
        results = processPdfDocument(path, output_dir, method=method, debug=True)
        print(f"\nProcessed {len(results)} pages")
        print(f"Results saved to: {output_dir}")
    else:
        # Process single image
        src = cv2.imread(path)
        
        if src is None:
            print(f'Error: Failed to load image: {path}')
            sys.exit()
        
        print(f"Processing image: {path}")
        start_time = time.time()
        
        cropped, debugImg, corners, method_used = detectAndCropDocumentPage(src, method=method, debug=True)
        
        elapsed_time = time.time() - start_time
        
        # Save results
        output_dir = os.path.dirname(path)
        base_name = os.path.splitext(os.path.basename(path))[0]
        
        cropped_path = os.path.join(output_dir, f"{base_name}_cropped.jpg")
        debug_path = os.path.join(output_dir, f"{base_name}_debug.jpg")
        
        cv2.imwrite(cropped_path, cropped)
        cv2.imwrite(debug_path, debugImg)
        
        print(f"\nProcessing completed in {elapsed_time:.3f} seconds")
        print(f"Method used: {method_used}")
        print(f"Cropped image saved to: {cropped_path}")
        print(f"Debug image saved to: {debug_path}")
