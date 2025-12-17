# YOLO Integration Guide for Document Segmentation

## Overview

This guide explains how to use YOLO (You Only Look Once) for document detection and segmentation as an alternative to U2-Net. YOLO offers fast, accurate object detection with excellent CPU performance.

## Why YOLO for Document Segmentation?

### Advantages
- ✅ **Fast inference**: 50-200ms on CPU (vs 385ms for U2-Net)
- ✅ **High accuracy**: 90-95% with proper training
- ✅ **CPU-optimized**: Excellent CPU performance
- ✅ **.NET compatible**: ONNX export works seamlessly
- ✅ **Roboflow support**: Easy training with pre-annotated datasets
- ✅ **Instance segmentation**: Can detect multiple documents

### Comparison

| Feature | YOLO | U2-Net |
|---------|------|--------|
| Speed (CPU) | 50-200ms | 385ms |
| Accuracy | 90-95% | 85-90% |
| Model Size | 6-25MB | 4.4-176MB |
| Training | Easy (Roboflow) | Complex |
| Multi-object | Yes | No |
| .NET Support | Excellent | Good |

## Roboflow Document Segmentation Project

### Reference Project

**URL**: https://universe.roboflow.com/maulvi-zm/document-segmentation-j6olp/model/1

This project demonstrates excellent document detection:
- **Dataset**: 1000+ annotated images
- **Classes**: Document (foreground) vs Background
- **Performance**: 92% mAP@0.5
- **Model**: YOLOv8 segmentation

### Key Learnings
- Annotates document boundaries precisely
- Handles multi-page documents
- Works with various backgrounds
- CPU-compatible ONNX export

## Implementation

### 1. Using Pre-trained YOLO Model

#### Install Dependencies

```bash
pip install ultralytics>=8.0.0  # YOLOv8
pip install onnxruntime>=1.15.0
```

#### Python Implementation

```python
from ultralytics import YOLO
import cv2
import numpy as np

class YOLODocumentDetector:
    def __init__(self, model_path="yolov8n-seg.pt"):
        """
        Initialize YOLO model for document segmentation
        
        Args:
            model_path: Path to YOLO model (.pt or .onnx)
        """
        self.model = YOLO(model_path)
    
    def detect_document(self, image_path, confidence_threshold=0.5):
        """
        Detect document in image
        
        Returns:
            mask: Binary mask of document region
            bbox: Bounding box [x, y, w, h]
            confidence: Detection confidence
        """
        # Run inference
        results = self.model(image_path, conf=confidence_threshold)
        
        if len(results) == 0 or len(results[0].masks) == 0:
            return None, None, 0.0
        
        # Get first detection (highest confidence)
        result = results[0]
        mask = result.masks[0].data.cpu().numpy()
        bbox = result.boxes[0].xyxy.cpu().numpy()
        confidence = float(result.boxes[0].conf)
        
        # Convert mask to binary
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Convert bbox to [x, y, w, h]
        x1, y1, x2, y2 = bbox[0]
        bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
        
        return mask, bbox, confidence
    
    def crop_document(self, image, mask, bbox, margin_percent=0.02):
        """
        Crop document with adaptive margins
        
        Args:
            image: Input image
            mask: Binary mask
            bbox: Bounding box [x, y, w, h]
            margin_percent: Safety margin (default 2%)
        
        Returns:
            cropped_image: Cropped document
            confidence: Detection confidence
        """
        x, y, w, h = bbox
        
        # Add small safety margin
        margin_w = int(w * margin_percent)
        margin_h = int(h * margin_percent)
        
        x = max(0, x - margin_w)
        y = max(0, y - margin_h)
        w = min(image.shape[1] - x, w + 2 * margin_w)
        h = min(image.shape[0] - y, h + 2 * margin_h)
        
        # Crop
        cropped = image[y:y+h, x:x+w]
        
        return cropped

# Usage
detector = YOLODocumentDetector("yolov8n-seg.pt")
mask, bbox, confidence = detector.detect_document("document.jpg")

if confidence > 0.5:
    image = cv2.imread("document.jpg")
    cropped = detector.crop_document(image, mask, bbox)
    cv2.imwrite("cropped.jpg", cropped)
```

### 2. Training Custom YOLO Model

#### Using Roboflow

1. **Create Project**
   - Go to [Roboflow](https://roboflow.com/)
   - Create new instance segmentation project
   - Name: "Document Segmentation"

2. **Upload and Annotate**
   ```python
   from roboflow import Roboflow
   
   rf = Roboflow(api_key="YOUR_API_KEY")
   project = rf.workspace().project("document-segmentation")
   project.upload("images/", annotation_format="coco-segmentation")
   ```

3. **Generate Dataset**
   - Apply augmentations (rotation, brightness, blur)
   - Split: 70% train, 20% val, 10% test
   - Export in YOLOv8 format

4. **Train Model**
   ```python
   from ultralytics import YOLO
   
   # Load pretrained model
   model = YOLO('yolov8n-seg.pt')
   
   # Train on custom dataset
   results = model.train(
       data='dataset.yaml',  # From Roboflow export
       epochs=100,
       imgsz=640,
       batch=16,
       name='document_seg',
       device='cpu'  # or 'cuda'
   )
   
   # Export to ONNX
   model.export(format='onnx')
   ```

#### Dataset YAML Format

```yaml
# dataset.yaml
path: /path/to/dataset
train: images/train
val: images/val

names:
  0: document

# Optional: augmentation
augment: true
```

### 3. ONNX Deployment (CPU-optimized)

#### Export Model

```python
from ultralytics import YOLO

model = YOLO('best.pt')  # Your trained model
model.export(
    format='onnx',
    dynamic=True,
    simplify=True,
    opset=12
)
```

#### Use ONNX Model

```python
import onnxruntime as ort
import numpy as np
import cv2

class YOLOONNXDetector:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
    
    def preprocess(self, image):
        """Preprocess image for YOLO"""
        # Resize to 640x640
        img = cv2.resize(image, (640, 640))
        # Normalize
        img = img.astype(np.float32) / 255.0
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        # Add batch dimension
        img = np.expand_dims(img, 0)
        return img
    
    def detect(self, image):
        """Run detection"""
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Postprocess
        boxes = outputs[0]  # [1, num_boxes, 6] -> [x, y, w, h, conf, class]
        masks = outputs[1]  # [1, num_masks, H, W]
        
        return boxes, masks

# Usage
detector = YOLOONNXDetector("best.onnx")
image = cv2.imread("document.jpg")
boxes, masks = detector.detect(image)
```

## Integration with .NET Core

### Using ONNX Runtime in C#

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

public class YOLODocumentDetector
{
    private InferenceSession session;
    
    public YOLODocumentDetector(string modelPath)
    {
        var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        session = new InferenceSession(modelPath, options);
    }
    
    public (byte[,] mask, Rectangle bbox, float confidence) DetectDocument(string imagePath)
    {
        // Load image
        using var image = Image.Load<Rgb24>(imagePath);
        var resized = image.Clone(x => x.Resize(640, 640));
        
        // Convert to tensor
        var input = new DenseTensor<float>(new[] { 1, 3, 640, 640 });
        for (int y = 0; y < 640; y++)
        {
            for (int x = 0; x < 640; x++)
            {
                var pixel = resized[x, y];
                input[0, 0, y, x] = pixel.R / 255.0f;
                input[0, 1, y, x] = pixel.G / 255.0f;
                input[0, 2, y, x] = pixel.B / 255.0f;
            }
        }
        
        // Run inference
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("images", input)
        };
        
        using var results = session.Run(inputs);
        
        // Parse outputs
        var boxes = results[0].AsTensor<float>();
        var masks = results[1].AsTensor<float>();
        
        // Extract first detection
        float x = boxes[0, 0, 0];
        float y = boxes[0, 0, 1];
        float w = boxes[0, 0, 2];
        float h = boxes[0, 0, 3];
        float conf = boxes[0, 0, 4];
        
        // Extract mask
        var mask = new byte[640, 640];
        for (int i = 0; i < 640; i++)
        {
            for (int j = 0; j < 640; j++)
            {
                mask[i, j] = masks[0, 0, i, j] > 0.5f ? (byte)255 : (byte)0;
            }
        }
        
        var bbox = new Rectangle((int)x, (int)y, (int)w, (int)h);
        return (mask, bbox, conf);
    }
}
```

### NuGet Packages

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.17.0" />
<PackageReference Include="SixLabors.ImageSharp" Version="3.1.0" />
```

## Advanced Features

### 1. Content Verification

```python
def verify_content_in_removed_region(image, mask, bbox):
    """
    Check if removed region contains text or important content
    
    Returns:
        suspicious: True if region might contain document content
    """
    import pytesseract
    
    # Get removed region (outside bbox)
    removed_region = image.copy()
    x, y, w, h = bbox
    removed_region[y:y+h, x:x+w] = 0  # Zero out document region
    
    # OCR on removed region
    text = pytesseract.image_to_string(removed_region)
    word_count = len(text.split())
    
    # If significant text found, suspicious
    if word_count > 5:
        return True
    
    # Check visual similarity
    doc_region = image[y:y+h, x:x+w]
    doc_mean = cv2.mean(doc_region)[:3]
    removed_mean = cv2.mean(removed_region)[:3]
    
    # If similar colors, suspicious
    color_diff = np.linalg.norm(np.array(doc_mean) - np.array(removed_mean))
    if color_diff < 30:
        return True
    
    return False
```

### 2. Multi-Document Detection

```python
def detect_multiple_documents(image, confidence_threshold=0.5):
    """
    Detect multiple documents in single image (e.g., 2-page spread)
    """
    results = model(image, conf=confidence_threshold)
    
    documents = []
    for result in results[0]:
        mask = result.masks[0].data.cpu().numpy()
        bbox = result.boxes[0].xyxy.cpu().numpy()
        confidence = float(result.boxes[0].conf)
        
        documents.append({
            'mask': mask,
            'bbox': bbox,
            'confidence': confidence
        })
    
    return documents
```

## Performance Optimization

### CPU Optimization

```python
# Use optimized ONNX providers
import onnxruntime as ort

session = ort.InferenceSession(
    'model.onnx',
    providers=[
        'CPUExecutionProvider'
    ]
)

# Enable graph optimizations
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.intra_op_num_threads = 4  # Use 4 CPU threads
```

### Model Size Optimization

```python
# Use smaller YOLO variant
model = YOLO('yolov8n-seg.pt')  # Nano: 6MB
# vs
model = YOLO('yolov8s-seg.pt')  # Small: 23MB
# vs
model = YOLO('yolov8m-seg.pt')  # Medium: 52MB
```

## Integration with Current System

### Add to doc_page_crop.py

```python
def detectDocumentPage_YOLO(img, model_name='yolov8n-seg', debug=False):
    """
    Detect document page using YOLO
    
    Args:
        img: Input image (BGR)
        model_name: YOLO model name
        debug: Enable debug output
    
    Returns:
        cropped_img, debug_img, corners, success, time_ms, confidence
    """
    import time
    from ultralytics import YOLO
    
    start_time = time.time()
    
    try:
        # Load model
        model_path = f"kukalib/models/{model_name}.onnx"
        model = YOLO(model_path)
        
        # Run detection
        results = model(img, conf=0.5)
        
        if len(results) == 0 or len(results[0].masks) == 0:
            print("YOLO: No document detected")
            return img, img, None, False, 0, 0.0
        
        # Get detection
        result = results[0]
        mask = result.masks[0].data.cpu().numpy()[0]
        bbox = result.boxes[0].xyxy.cpu().numpy()[0]
        confidence = float(result.boxes[0].conf)
        
        # Resize mask to image size
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Convert bbox
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Add small margin (2%)
        margin_x = int((x2 - x1) * 0.02)
        margin_y = int((y2 - y1) * 0.02)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(img.shape[1], x2 + margin_x)
        y2 = min(img.shape[0], y2 + margin_y)
        
        # Crop
        cropped = img[y1:y2, x1:x2]
        
        # Create debug image
        debug_img = img.copy()
        debug_img = cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Calculate time
        time_ms = (time.time() - start_time) * 1000
        
        corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        
        return cropped, debug_img, corners, True, time_ms, confidence
        
    except Exception as e:
        print(f"YOLO Error: {e}")
        return img, img, None, False, 0, 0.0
```

## Recommendation

### Use YOLO When:
- ✅ Need faster inference (<200ms)
- ✅ Multiple documents in image
- ✅ Easy training with Roboflow
- ✅ Deploying to .NET applications
- ✅ CPU-only environment

### Use U2-Net When:
- ✅ Need pixel-perfect segmentation
- ✅ Complex backgrounds
- ✅ Already have U2-Net training pipeline
- ✅ Single document per image

### Hybrid Approach (Recommended):
1. Try YOLO first (fast, 50-100ms)
2. If confidence < 0.7, fallback to U2-Net
3. Skip cropping if both methods have low confidence

## Resources

- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **Roboflow Universe**: https://universe.roboflow.com/
- **Document Segmentation Project**: https://universe.roboflow.com/maulvi-zm/document-segmentation-j6olp
- **ONNX Runtime**: https://onnxruntime.ai/
- **Ultralytics**: https://github.com/ultralytics/ultralytics

## Summary

YOLO provides:
- ✅ Fast inference (50-200ms on CPU)
- ✅ High accuracy (90-95%)
- ✅ Easy training with Roboflow
- ✅ Excellent .NET compatibility
- ✅ Multi-document support

Perfect for production deployment in CPU-based .NET Core applications!
