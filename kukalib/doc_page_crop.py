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
    Best for: Complex backgrounds, poor lighting, cluttered scenes
    
    Parameters:
    -----------
    src : np.ndarray
        Input image (BGR format)
    model_path : str, optional
        Path to U2Net model weights. If None, uses default or falls back.
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
    try:
        # Try to import rembg (which uses U2-Net internally)
        from rembg import remove, new_session
        
        if debug:
            print("U2-Net Method: Using rembg for background removal")
        
        # Remove background
        output = remove(src)
        
        # Convert to grayscale and threshold to get mask
        if len(output.shape) == 3 and output.shape[2] == 4:
            # Has alpha channel
            mask = output[:, :, 3]
        else:
            # Convert to grayscale
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            mask = gray
        
        # Threshold to get binary mask
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean mask
        kernel = np.ones((9, 9), np.uint8)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours in mask
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            if debug:
                print("U2-Net Method: No contours found in mask")
            return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Check if area is significant (at least 20% of image)
        area_ratio = (w * h) / (src.shape[0] * src.shape[1])
        if area_ratio < 0.2:
            if debug:
                print(f"U2-Net Method: Detected area too small ({area_ratio*100:.1f}%)")
            return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False
        
        # Approximate contour to quadrilateral
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        
        # If we got 4 points, use them; otherwise use bounding rect
        if len(approx) == 4:
            corners = approx.reshape(4, 2)
        else:
            # Use bounding rectangle corners
            corners = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ])
        
        # Order points correctly
        corners_ordered = order_points(corners)
        
        # Apply perspective transform
        destination_corners = find_dest(corners_ordered)
        M = cv2.getPerspectiveTransform(np.float32(corners_ordered), np.float32(destination_corners))
        cropedImg = cv2.warpPerspective(src, M,
                                       (destination_corners[2][0], destination_corners[2][1]),
                                       flags=cv2.INTER_LINEAR)
        
        # Create debug visualization
        lineImg = src.copy()
        cv2.polylines(lineImg, [corners_ordered], True, (0, 255, 0), 3)
        for corner in corners_ordered:
            cv2.circle(lineImg, tuple(corner), 8, (0, 0, 255), -1)
        
        # Draw mask overlay
        mask_color = cv2.applyColorMap(mask_binary, cv2.COLORMAP_JET)
        lineImg = cv2.addWeighted(lineImg, 0.7, mask_color, 0.3, 0)
        
        if debug:
            print(f"U2-Net Method: Successfully detected document")
            print(f"Document area: {area_ratio*100:.1f}% of image")
            print(f"Corners: {len(approx)} points approximated")
        
        topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint = corners_ordered
        return cropedImg, lineImg, (topleftPoint, toprightPoint, bottomrightPoint, bottomleftPoint), True
        
    except ImportError:
        if debug:
            print("U2-Net Method: rembg not installed, falling back to OpenCV")
        return np.zeros(src.shape, dtype=np.uint8), src.copy(), ([], [], [], []), False
    except Exception as e:
        if debug:
            print(f"U2-Net Method: Error occurred: {str(e)}")
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
        Detection method: 'auto', 'u2net', 'opencv'
        'auto' tries U2-Net first, falls back to OpenCV
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
        print(f"=== Document Page Detection ===")
        print(f"Input image size: {src.shape[1]}x{src.shape[0]}")
        print(f"Requested method: {method}")
    
    cropped = None
    debugImg = None
    corners = ([], [], [], [])
    method_used = "none"
    success = False
    
    if method == 'opencv':
        # Use only OpenCV
        cropped, debugImg, corners, success = detectDocumentPage_OpenCV(src, debug=debug)
        method_used = "opencv"
        
    elif method == 'u2net':
        # Use only U2-Net
        cropped, debugImg, corners, success = detectDocumentPage_U2Net(src, debug=debug)
        if success:
            method_used = "u2net"
        else:
            method_used = "u2net_failed"
            
    else:  # method == 'auto'
        # Try U2-Net first
        cropped, debugImg, corners, success = detectDocumentPage_U2Net(src, debug=debug)
        if success:
            method_used = "u2net"
        else:
            # Fallback to OpenCV
            if debug:
                print("Auto mode: Falling back to OpenCV method")
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
        print(f"Final method used: {method_used}")
        print(f"Output image size: {cropped.shape[1]}x{cropped.shape[0]}")
        print("=== Detection Complete ===\n")
    
    return cropped, debugImg, corners, method_used


def convertPdfToImages(pdf_path: str, output_dir: Optional[str] = None, dpi: int = 200) -> List[np.ndarray]:
    """
    Convert PDF file to list of images (one per page).
    
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
    list: List of numpy arrays (images in BGR format)
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
