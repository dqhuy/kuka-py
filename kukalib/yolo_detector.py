"""
YOLO Document Detection Module

This module provides YOLO-based document detection as an alternative to U2-Net.
Uses YOLOv8 for fast, accurate document segmentation.

CPU-optimized and .NET compatible via ONNX export.

Author: KUKA-LIB Team
Version: 1.0.0
"""

import cv2
import numpy as np
import os
import sys
from typing import Tuple, Optional, List

def check_yolo_available() -> bool:
    """Check if YOLO (ultralytics) is installed"""
    try:
        import ultralytics
        return True
    except ImportError:
        return False


def detectDocumentPage_YOLO(src: np.ndarray, 
                           model_name: str = 'yolov8n-seg', 
                           confidence_threshold: float = 0.5,
                           debug: bool = False) -> Tuple[np.ndarray, np.ndarray, tuple, bool, float, float]:
    """
    Detect document page using YOLO segmentation model.
    
    YOLO offers:
    - Fast inference: 50-200ms on CPU
    - High accuracy: 90-95% with proper training
    - Multi-document detection: Can detect multiple pages in one image
    - CPU-optimized: Better CPU performance than U2-Net
    - .NET compatible: Easy ONNX export
    
    Best for: Fast processing, multi-page documents, CPU-only environments
    
    Parameters:
    -----------
    src : np.ndarray
        Input image (BGR format)
    model_name : str
        YOLO model name: 'yolov8n-seg' (nano), 'yolov8s-seg' (small), etc.
        Or path to custom trained model (.pt or .onnx)
    confidence_threshold : float
        Minimum confidence for detection (default: 0.5)
    debug : bool
        If True, print extensive debug information
        
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
        print(f"YOLO Method: Starting document detection (model: {model_name})", file=sys.stderr)
        print("="*70, file=sys.stderr)
        print(f"Input image shape: {src.shape}", file=sys.stderr)
    
    # Check if YOLO is available
    if not check_yolo_available():
        if debug:
            print("❌ YOLO (ultralytics) not installed", file=sys.stderr)
            print("   Install with: pip install ultralytics", file=sys.stderr)
        return src, src, None, False, 0, 0.0
    
    try:
        from ultralytics import YOLO
        
        if debug:
            print("✅ YOLO module imported successfully", file=sys.stderr)
        
        # Determine model path
        if os.path.exists(model_name):
            model_path = model_name
        elif model_name.endswith('.pt') or model_name.endswith('.onnx'):
            model_path = f"kukalib/models/{model_name}"
        else:
            model_path = model_name  # Use pretrained model name
        
        if debug:
            print(f"📂 Loading model: {model_path}", file=sys.stderr)
        
        # Load model
        model = YOLO(model_path)
        
        if debug:
            print("✅ Model loaded successfully", file=sys.stderr)
        
        # Run inference
        if debug:
            print(f"🔍 Running inference with confidence threshold: {confidence_threshold}", file=sys.stderr)
        
        results = model(src, conf=confidence_threshold, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            if debug:
                print("❌ No documents detected", file=sys.stderr)
            return src, src, None, False, (time.time() - start_time) * 1000, 0.0
        
        result = results[0]
        
        if debug:
            print(f"✅ Found {len(result.boxes)} detection(s)", file=sys.stderr)
        
        # Handle multiple detections (multi-page documents)
        if len(result.boxes) > 1:
            if debug:
                print(f"📄 Multiple documents detected: {len(result.boxes)} pages", file=sys.stderr)
                print("   Combining all detections into single bounding box", file=sys.stderr)
            
            # Get all bboxes
            all_boxes = result.boxes.xyxy.cpu().numpy()
            
            # Combine into single bbox
            x1 = int(np.min(all_boxes[:, 0]))
            y1 = int(np.min(all_boxes[:, 1]))
            x2 = int(np.max(all_boxes[:, 2]))
            y2 = int(np.max(all_boxes[:, 3]))
            
            # Average confidence
            confidence = float(np.mean(result.boxes.conf.cpu().numpy()))
            
            # Get mask if available
            if hasattr(result, 'masks') and result.masks is not None:
                # Combine all masks
                combined_mask = np.zeros((src.shape[0], src.shape[1]), dtype=np.uint8)
                for mask_data in result.masks.data:
                    mask = mask_data.cpu().numpy()
                    mask = cv2.resize(mask, (src.shape[1], src.shape[0]))
                    combined_mask = np.maximum(combined_mask, (mask > 0.5).astype(np.uint8) * 255)
                mask = combined_mask
            else:
                mask = None
        else:
            # Single detection
            bbox = result.boxes[0].xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = bbox.astype(int)
            confidence = float(result.boxes[0].conf)
            
            # Get mask if available
            if hasattr(result, 'masks') and result.masks is not None:
                mask = result.masks[0].data.cpu().numpy()[0]
                mask = cv2.resize(mask, (src.shape[1], src.shape[0]))
                mask = (mask > 0.5).astype(np.uint8) * 255
            else:
                mask = None
        
        if debug:
            print(f"📊 Detection confidence: {confidence:.3f}", file=sys.stderr)
            print(f"📦 Bounding box: x={x1}, y={y1}, w={x2-x1}, h={y2-y1}", file=sys.stderr)
        
        # Add small adaptive margin (2% by default)
        margin_percent = 0.02
        margin_x = int((x2 - x1) * margin_percent)
        margin_y = int((y2 - y1) * margin_percent)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(src.shape[1], x2 + margin_x)
        y2 = min(src.shape[0], y2 + margin_y)
        
        if debug:
            print(f"   With margins: x={x1}, y={y1}, w={x2-x1}, h={y2-y1}", file=sys.stderr)
        
        # Check if cropping is needed (document > 92% of image)
        doc_area = (x2 - x1) * (y2 - y1)
        img_area = src.shape[0] * src.shape[1]
        area_ratio = doc_area / img_area
        
        if debug:
            print(f"📐 Area ratio: {area_ratio:.2%} of image", file=sys.stderr)
        
        if area_ratio > 0.92:
            if debug:
                print("⚠️ Document occupies >92% of image - skipping crop", file=sys.stderr)
            return src, src, None, False, (time.time() - start_time) * 1000, confidence
        
        # Crop image
        cropped = src[y1:y2, x1:x2]
        
        if debug:
            print(f"✂️ Cropped image shape: {cropped.shape}", file=sys.stderr)
        
        # Create debug visualization
        debug_img = src.copy()
        
        # Draw bounding box
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw mask if available
        if mask is not None:
            # Create colored mask overlay
            mask_overlay = np.zeros_like(src)
            mask_overlay[mask > 0] = [0, 255, 0]  # Green for document
            debug_img = cv2.addWeighted(debug_img, 0.7, mask_overlay, 0.3, 0)
        
        # Add text annotations
        cv2.putText(debug_img, f"YOLO: {confidence:.2f}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Calculate corners
        corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        
        # Calculate time
        time_ms = (time.time() - start_time) * 1000
        
        if debug:
            print("="*70, file=sys.stderr)
            print("YOLO Method: SUCCESS", file=sys.stderr)
            print(f"Detection confidence: {confidence:.3f}", file=sys.stderr)
            print(f"Processing time: {time_ms:.1f}ms", file=sys.stderr)
            print(f"Cropped size: {cropped.shape[1]}x{cropped.shape[0]}", file=sys.stderr)
            print("="*70, file=sys.stderr)
        
        return cropped, debug_img, tuple(corners), True, time_ms, confidence
        
    except Exception as e:
        if debug:
            print(f"❌ YOLO Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
        
        return src, src, None, False, (time.time() - start_time) * 1000, 0.0


def download_pretrained_yolo_model(model_name: str = 'yolov8n-seg', output_dir: str = 'kukalib/models'):
    """
    Download pretrained YOLO model
    
    Args:
        model_name: Model name (e.g., 'yolov8n-seg', 'yolov8s-seg')
        output_dir: Directory to save model
    """
    if not check_yolo_available():
        print("❌ YOLO (ultralytics) not installed")
        print("   Install with: pip install ultralytics")
        return False
    
    try:
        from ultralytics import YOLO
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📥 Downloading {model_name}...")
        model = YOLO(model_name)
        
        # Export to ONNX for CPU optimization
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        print(f"📦 Exporting to ONNX: {onnx_path}")
        model.export(format='onnx', dynamic=True, simplify=True)
        
        print(f"✅ Model ready: {onnx_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("YOLO Document Detector - Test Mode")
    
    if not check_yolo_available():
        print("\n⚠️ ultralytics package not installed")
        print("Install with: pip install ultralytics")
        sys.exit(1)
    
    # Test with sample image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\n📷 Loading image: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            sys.exit(1)
        
        print(f"   Image shape: {img.shape}")
        
        # Run detection
        cropped, debug_img, corners, success, time_ms, confidence = detectDocumentPage_YOLO(
            img, debug=True
        )
        
        if success:
            print(f"\n✅ Detection successful!")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Time: {time_ms:.1f}ms")
            print(f"   Cropped shape: {cropped.shape}")
            
            # Save results
            cv2.imwrite("yolo_cropped.jpg", cropped)
            cv2.imwrite("yolo_debug.jpg", debug_img)
            print(f"\n💾 Saved: yolo_cropped.jpg, yolo_debug.jpg")
        else:
            print("\n❌ Detection failed")
    else:
        print("\nUsage: python yolo_detector.py <image_path>")
        print("\nOr download pretrained model:")
        print("   from kukalib.yolo_detector import download_pretrained_yolo_model")
        print("   download_pretrained_yolo_model('yolov8n-seg')")
