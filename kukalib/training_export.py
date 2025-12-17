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
    
    This is a smart content verification to avoid cropping document text or images.
    
    Args:
        image: Original image
        mask: Binary mask (255=document, 0=background)
        bbox: Bounding box [x, y, width, height]
        debug: Enable debug output
    
    Returns:
        tuple: (suspicious, details)
            suspicious: True if removed region might contain document content
            details: Dictionary with verification metrics
    """
    x, y, w, h = bbox
    
    # Create removed region (outside bbox)
    removed_region = image.copy()
    removed_region[y:y+h, x:x+w] = 0  # Zero out document region
    
    # Get document region
    doc_region = image[y:y+h, x:x+w]
    
    details = {}
    suspicious_count = 0
    
    # 1. Check for text in removed region using edge density
    # (Simple heuristic: text has high edge density)
    gray_removed = cv2.cvtColor(removed_region, cv2.COLOR_BGR2GRAY)
    edges_removed = cv2.Canny(gray_removed, 50, 150)
    edge_density_removed = np.count_nonzero(edges_removed) / (image.shape[0] * image.shape[1])
    
    details['edge_density_removed'] = edge_density_removed
    
    # If significant edges in removed region, suspicious
    if edge_density_removed > 0.01:  # 1% edge density threshold
        suspicious_count += 1
        if debug:
            print(f"⚠️ High edge density in removed region: {edge_density_removed:.4f}", file=sys.stderr)
    
    # 2. Check visual similarity between document and removed region
    doc_mean = cv2.mean(doc_region)[:3]
    
    # Only check non-zero parts of removed region
    removed_nonzero = removed_region[np.any(removed_region != 0, axis=2)]
    if len(removed_nonzero) > 100:  # Need some pixels to compare
        removed_mean = np.mean(removed_nonzero, axis=0)
        color_diff = np.linalg.norm(np.array(doc_mean) - np.array(removed_mean))
        
        details['color_difference'] = color_diff
        
        # If similar colors, might be same content
        if color_diff < 40:  # Threshold for color similarity
            suspicious_count += 1
            if debug:
                print(f"⚠️ Similar colors between document and removed region: diff={color_diff:.2f}", file=sys.stderr)
    
    # 3. Check for connected components in removed region
    # (Multiple components might indicate text/content)
    _, binary_removed = cv2.threshold(gray_removed, 30, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_removed, connectivity=8)
    
    # Filter small components (noise)
    significant_components = 0
    for i in range(1, num_labels):  # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 100:  # Significant size threshold
            significant_components += 1
    
    details['significant_components'] = significant_components
    
    if significant_components > 5:  # Multiple components suggest content
        suspicious_count += 1
        if debug:
            print(f"⚠️ Multiple components in removed region: {significant_components}", file=sys.stderr)
    
    # 4. Check mask coverage
    # If mask doesn't cover much of the image, cropping might cut content
    mask_coverage = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
    details['mask_coverage'] = mask_coverage
    
    if mask_coverage > 0.95:  # Document covers >95% of image
        # Almost no background, cropping is suspicious
        suspicious_count += 1
        if debug:
            print(f"⚠️ High mask coverage: {mask_coverage:.2f} - minimal background", file=sys.stderr)
    
    # Decision: suspicious if 2 or more indicators
    suspicious = suspicious_count >= 2
    details['suspicious_count'] = suspicious_count
    details['is_suspicious'] = suspicious
    
    if debug:
        print(f"\n📊 Content Verification Results:", file=sys.stderr)
        print(f"   Edge density removed: {edge_density_removed:.4f}", file=sys.stderr)
        print(f"   Significant components: {significant_components}", file=sys.stderr)
        print(f"   Mask coverage: {mask_coverage:.2f}", file=sys.stderr)
        print(f"   Suspicious indicators: {suspicious_count}/4", file=sys.stderr)
        print(f"   🚨 Suspicious: {suspicious}", file=sys.stderr)
    
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
