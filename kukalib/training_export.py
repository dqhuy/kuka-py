"""
Training Data Export Module for Document Detection

This module provides functionality to export training data in COCO format
for fine-tuning U2-Net or training YOLO models.

Supports export of:
- Original images
- Binary masks
- Bounding box annotations
- COCO format JSON

Author: KUKA-LIB Team
Version: 1.0.0
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class TrainingDataExporter:
    """Export training data for model fine-tuning"""
    
    def __init__(self, output_dir: str = "training_data"):
        """
        Initialize exporter
        
        Args:
            output_dir: Directory to save exported data
        """
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.masks_dir = os.path.join(output_dir, "masks")
        self.annotations_file = os.path.join(output_dir, "annotations.json")
        
        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)
        
        # Initialize COCO format structure
        self.coco_data = {
            "info": {
                "description": "Document Detection Training Dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "document",
                    "supercategory": "object"
                }
            ]
        }
        
        self.image_id = 1
        self.annotation_id = 1
    
    def export_sample(self, 
                     original_image: np.ndarray,
                     mask: np.ndarray,
                     bbox: List[int],
                     filename_prefix: str,
                     confidence: float = 1.0,
                     metadata: Optional[Dict] = None) -> Dict:
        """
        Export a single training sample
        
        Args:
            original_image: Original input image (BGR)
            mask: Binary mask (0=background, 255=document)
            bbox: Bounding box [x, y, width, height]
            filename_prefix: Prefix for saved files
            confidence: Detection confidence
            metadata: Additional metadata
        
        Returns:
            dict: Export information
        """
        # Generate filenames
        image_filename = f"{filename_prefix}_{self.image_id:06d}.jpg"
        mask_filename = f"{filename_prefix}_{self.image_id:06d}_mask.png"
        
        image_path = os.path.join(self.images_dir, image_filename)
        mask_path = os.path.join(self.masks_dir, mask_filename)
        
        # Save original image
        cv2.imwrite(image_path, original_image)
        
        # Save mask
        cv2.imwrite(mask_path, mask)
        
        # Get image dimensions
        height, width = original_image.shape[:2]
        
        # Add to COCO images
        image_info = {
            "id": self.image_id,
            "file_name": image_filename,
            "width": width,
            "height": height,
            "date_captured": datetime.now().isoformat(),
            "confidence": confidence
        }
        
        if metadata:
            image_info.update(metadata)
        
        self.coco_data["images"].append(image_info)
        
        # Create segmentation from mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        segmentation = []
        if len(contours) > 0:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            # Flatten to [x1, y1, x2, y2, ...]
            seg = largest_contour.flatten().tolist()
            segmentation.append(seg)
        
        # Calculate area
        area = int(bbox[2] * bbox[3])
        
        # Add to COCO annotations
        annotation = {
            "id": self.annotation_id,
            "image_id": self.image_id,
            "category_id": 1,
            "segmentation": segmentation,
            "area": area,
            "bbox": bbox,  # [x, y, width, height]
            "iscrowd": 0,
            "confidence": confidence
        }
        
        self.coco_data["annotations"].append(annotation)
        
        # Increment IDs
        self.image_id += 1
        self.annotation_id += 1
        
        return {
            "image_path": image_path,
            "mask_path": mask_path,
            "image_id": self.image_id - 1,
            "annotation_id": self.annotation_id - 1
        }
    
    def save_annotations(self):
        """Save COCO annotations to JSON file"""
        with open(self.annotations_file, 'w') as f:
            json.dump(self.coco_data, f, indent=2)
        
        return self.annotations_file
    
    def get_stats(self) -> Dict:
        """Get export statistics"""
        return {
            "total_images": len(self.coco_data["images"]),
            "total_annotations": len(self.coco_data["annotations"]),
            "output_dir": self.output_dir,
            "images_dir": self.images_dir,
            "masks_dir": self.masks_dir,
            "annotations_file": self.annotations_file
        }


def verify_removed_content(image: np.ndarray, 
                          mask: np.ndarray, 
                          bbox: List[int],
                          debug: bool = False) -> Tuple[bool, Dict]:
    """
    Verify if removed region (outside bbox) contains important content
    
    CRITICAL V4 HOTFIX 2: Enhanced morphological text detection
    Uses morphological operations (dilate/erode) to detect text structures
    in removed regions - prevents false crops on 2-page documents.
    
    Args:
        image: Original image
        mask: Binary mask (255=document, 0=background)
        bbox: Bounding box [x, y, width, height]
        debug: Enable debug output
    
    Returns:
        tuple: (suspicious, details)
            suspicious: True if removed region contains text/content
            details: Dictionary with verification metrics
    """
    import sys
    
    x, y, w, h = bbox
    
    # Create removed region (outside bbox)
    removed_region = image.copy()
    removed_region[y:y+h, x:x+w] = 0  # Zero out document region
    
    # Get document region
    doc_region = image[y:y+h, x:x+w]
    
    details = {}
    suspicious_count = 0
    
    # Convert to grayscale
    gray_removed = cv2.cvtColor(removed_region, cv2.COLOR_BGR2GRAY)
    
    # ==============================================================================
    # CRITICAL FIX 1: Morphological Text Detection
    # ==============================================================================
    if debug:
        print(f"\n🔍 Morphological Text Detection:", file=sys.stderr)
    
    # Binarize removed region
    _, binary_removed = cv2.threshold(gray_removed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Detect horizontal text lines (common in documents)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_h = cv2.morphologyEx(binary_removed, cv2.MORPH_CLOSE, kernel_h)
    detected_h = cv2.morphologyEx(detected_h, cv2.MORPH_OPEN, kernel_h)
    
    # Detect vertical text/columns
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    detected_v = cv2.morphologyEx(binary_removed, cv2.MORPH_CLOSE, kernel_v)
    detected_v = cv2.morphologyEx(detected_v, cv2.MORPH_OPEN, kernel_v)
    
    # Combine detections
    text_pattern = cv2.bitwise_or(detected_h, detected_v)
    
    # Calculate text ratio
    text_pixels = np.count_nonzero(text_pattern)
    total_removed_pixels = np.count_nonzero(gray_removed)
    text_ratio = text_pixels / total_removed_pixels if total_removed_pixels > 0 else 0
    
    details['text_pattern_ratio'] = text_ratio
    
    # VERY SENSITIVE: Any text pattern is suspicious
    if text_ratio > 0.02:  # 2% text - CRITICAL
        suspicious_count += 3  # Triple weight for strong text signal
        if debug:
            print(f"   ❌ TEXT DETECTED: {text_ratio*100:.2f}% text patterns!", file=sys.stderr)
    elif text_ratio > 0.01:  # 1% text - suspicious
        suspicious_count += 2
        if debug:
            print(f"   ⚠️  Possible text: {text_ratio*100:.2f}% patterns", file=sys.stderr)
    elif text_ratio > 0.005:  # 0.5% - minor concern
        suspicious_count += 1
        if debug:
            print(f"   ⚠️  Minor text patterns: {text_ratio*100:.2f}%", file=sys.stderr)
    else:
        if debug:
            print(f"   ✅ Minimal text: {text_ratio*100:.2f}%", file=sys.stderr)
    
    # ==============================================================================
    # CRITICAL FIX 2: Margin Analysis
    # ==============================================================================
    # Check each margin (top, bottom, left, right) separately
    margin_size = 50  # pixels
    h_img, w_img = image.shape[:2]
    
    margins = []
    if y > margin_size:  # Top
        margins.append(('top', gray_removed[max(0, y-margin_size):y, :]))
    if y + h < h_img - margin_size:  # Bottom
        margins.append(('bottom', gray_removed[y+h:min(h_img, y+h+margin_size), :]))
    if x > margin_size:  # Left
        margins.append(('left', gray_removed[:, max(0, x-margin_size):x]))
    if x + w < w_img - margin_size:  # Right
        margins.append(('right', gray_removed[:, x+w:min(w_img, x+w+margin_size)]))
    
    margin_text_detected = False
    for margin_name, margin_region in margins:
        if margin_region.size < 100:  # Too small to analyze
            continue
        
        # Detect text in margin using morphological ops
        _, binary_margin = cv2.threshold(margin_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        text_in_margin = cv2.morphologyEx(binary_margin, cv2.MORPH_CLOSE, kernel_text)
        
        text_density = np.count_nonzero(text_in_margin) / margin_region.size
        
        if debug:
            print(f"   Margin '{margin_name}': text_density={text_density*100:.2f}%", file=sys.stderr)
        
        # If ANY text in margins, CRITICAL
        if text_density > 0.08:  # 8% threshold
            margin_text_detected = True
            suspicious_count += 3  # Critical weight
            if debug:
                print(f"   ❌ CRITICAL: Text in {margin_name} margin!", file=sys.stderr)
            break
        elif text_density > 0.05:  # 5% threshold
            margin_text_detected = True
            suspicious_count += 2
            if debug:
                print(f"   ⚠️  Suspicious: Content in {margin_name} margin", file=sys.stderr)
            break
    
    details['margin_text_detected'] = margin_text_detected
    
    # ==============================================================================
    # 3. Connected Components Analysis
    # ==============================================================================
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_removed, connectivity=8)
    
    # Count significant components
    significant_components = 0
    large_components = 0
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 50:
            significant_components += 1
        if area > 500:
            large_components += 1
    
    details['significant_components'] = significant_components
    details['large_components'] = large_components
    
    # Large components = likely content
    if large_components > 2:
        suspicious_count += 2
        if debug:
            print(f"   ⚠️  {large_components} large components", file=sys.stderr)
    elif significant_components > 10:
        suspicious_count += 1
        if debug:
            print(f"   ⚠️  {significant_components} components", file=sys.stderr)
    
    # ==============================================================================
    # 4. Mask Coverage Check
    # ==============================================================================
    mask_coverage = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
    details['mask_coverage'] = mask_coverage
    
    if mask_coverage > 0.92:  # >92% coverage
        suspicious_count += 2  # Critical
        if debug:
            print(f"   ❌ HIGH mask coverage: {mask_coverage*100:.1f}%", file=sys.stderr)
    elif mask_coverage > 0.85:  # 85-92%
        suspicious_count += 1
        if debug:
            print(f"   ⚠️  Moderate coverage: {mask_coverage*100:.1f}%", file=sys.stderr)
    
    # ==============================================================================
    # DECISION: Lowered threshold with weighted scoring
    # ==============================================================================
    # With triple-weight on text detection, even 1 text indicator triggers suspicious
    suspicious = suspicious_count >= 2
    details['suspicious_count'] = suspicious_count
    details['is_suspicious'] = suspicious
    details['reason'] = "text detected in removed region" if (margin_text_detected or text_ratio > 0.01) else "suspicious content"
    
    if debug:
        print(f"\n📊 Content Verification Summary:", file=sys.stderr)
        print(f"   Text pattern ratio: {text_ratio*100:.2f}%", file=sys.stderr)
        print(f"   Margin text: {margin_text_detected}", file=sys.stderr)
        print(f"   Components: {significant_components} ({large_components} large)", file=sys.stderr)
        print(f"   Mask coverage: {mask_coverage*100:.1f}%", file=sys.stderr)
        print(f"   Suspicious score: {suspicious_count}", file=sys.stderr)
        print(f"   🚨 DECISION: {'SUSPICIOUS - SKIP CROP' if suspicious else 'SAFE TO CROP'}", file=sys.stderr)
    
    return suspicious, details


def create_yolo_dataset(coco_json_path: str, output_dir: str = "yolo_dataset"):
    """
    Convert COCO format to YOLO format for training
    
    Args:
        coco_json_path: Path to COCO annotations.json
        output_dir: Output directory for YOLO dataset
    """
    import json
    import shutil
    
    # Load COCO data
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create YOLO directory structure
    train_images_dir = os.path.join(output_dir, "images", "train")
    train_labels_dir = os.path.join(output_dir, "labels", "train")
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    
    # Get images directory from COCO path
    coco_dir = os.path.dirname(coco_json_path)
    images_source_dir = os.path.join(coco_dir, "images")
    
    # Convert each annotation
    for image_info in coco_data["images"]:
        image_id = image_info["id"]
        image_filename = image_info["file_name"]
        width = image_info["width"]
        height = image_info["height"]
        
        # Find annotations for this image
        annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]
        
        if not annotations:
            continue
        
        # Copy image
        source_image = os.path.join(images_source_dir, image_filename)
        dest_image = os.path.join(train_images_dir, image_filename)
        shutil.copy(source_image, dest_image)
        
        # Create YOLO label file
        label_filename = os.path.splitext(image_filename)[0] + ".txt"
        label_path = os.path.join(train_labels_dir, label_filename)
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                # Get bbox [x, y, width, height]
                bbox = ann["bbox"]
                
                # Convert to YOLO format [class_id, x_center, y_center, width, height] (normalized)
                x_center = (bbox[0] + bbox[2] / 2) / width
                y_center = (bbox[1] + bbox[3] / 2) / height
                norm_width = bbox[2] / width
                norm_height = bbox[3] / height
                
                # Class ID 0 for document
                f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    # Create dataset.yaml
    yaml_content = f"""# YOLO Dataset Configuration
path: {os.path.abspath(output_dir)}
train: images/train
val: images/train  # Use same for now, split later

names:
  0: document
"""
    
    with open(os.path.join(output_dir, "dataset.yaml"), 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ YOLO dataset created: {output_dir}")
    print(f"   Images: {len(coco_data['images'])}")
    print(f"   Annotations: {len(coco_data['annotations'])}")
    print(f"   Config: {os.path.join(output_dir, 'dataset.yaml')}")


if __name__ == "__main__":
    # Test example
    print("Training Data Exporter - Test Mode")
    
    # Create sample data
    exporter = TrainingDataExporter("test_training_data")
    
    # Create dummy image and mask
    img = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)
    mask = np.zeros((1000, 800), dtype=np.uint8)
    mask[100:900, 100:700] = 255
    
    bbox = [100, 100, 600, 800]
    
    # Export
    result = exporter.export_sample(img, mask, bbox, "sample", confidence=0.85)
    print(f"Exported: {result}")
    
    # Save annotations
    ann_file = exporter.save_annotations()
    print(f"Annotations saved: {ann_file}")
    
    # Get stats
    stats = exporter.get_stats()
    print(f"Stats: {stats}")
