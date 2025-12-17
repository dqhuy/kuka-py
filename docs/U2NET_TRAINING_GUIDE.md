# U2-Net Fine-Tuning Training Guide

## Overview

This guide explains how to fine-tune the U2-Net model with custom data to improve document detection accuracy. The workflow supports iterative training using platforms like Roboflow.

## Training Workflow

### 1. Data Collection Phase

Run the system to generate training data:

```bash
# Process images and export training data
python kukalib/doc_page_crop.py input.pdf --export-training-data
```

The system will generate:
- **Original images**: Raw input images
- **Mask images**: Binary masks showing document regions (white=document, black=background)
- **Annotations**: COCO format JSON with bounding boxes

### 2. Data Review and Correction

1. Download exported data from GUI
2. Review mask images for accuracy
3. Correct incorrect masks using annotation tools:
   - [Roboflow](https://roboflow.com/) - Recommended, free tier available
   - [CVAT](https://cvat.org/) - Open source
   - [LabelMe](http://labelme.csail.mit.edu/) - Open source

### 3. Dataset Format

#### COCO Segmentation Format

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image001.jpg",
      "width": 1024,
      "height": 768
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 123456,
      "bbox": [x, y, width, height],
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "document",
      "supercategory": "object"
    }
  ]
}
```

### 4. Training with PyTorch

#### Setup Environment

```bash
# Install training dependencies
pip install torch torchvision
pip install opencv-python pillow numpy
pip install albumentations  # For data augmentation
```

#### Training Script Structure

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from kukalib.u2net_model import U2NET, U2NETP

class DocumentDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted(glob.glob(f"{image_dir}/*.jpg"))
        self.mask_paths = sorted(glob.glob(f"{mask_dir}/*.png"))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = cv2.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], 0)
        
        # Resize to 320x320 for U2-Net
        image = cv2.resize(image, (320, 320))
        mask = cv2.resize(mask, (320, 320))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        # Convert to torch tensors
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask

def train_u2net(model, train_loader, epochs=100, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss (U2-Net has multiple outputs)
            loss = 0
            for output in outputs:
                loss += criterion(output, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"u2net_checkpoint_{epoch+1}.pth")

# Usage
dataset = DocumentDataset("training_images", "training_masks")
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = U2NETP()  # or U2NET() for full model
train_u2net(model, train_loader, epochs=100)
```

### 5. Convert to ONNX for Deployment

```python
import torch
from kukalib.u2net_model import U2NETP

# Load trained model
model = U2NETP()
model.load_state_dict(torch.load("u2net_checkpoint_100.pth"))
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 320, 320)
torch.onnx.export(
    model,
    dummy_input,
    "kukalib/models/u2netp_custom.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

### 6. Using Roboflow (Recommended)

#### Advantages
- Free tier available
- Web-based annotation tools
- Auto-augmentation
- Version control for datasets
- One-click export to multiple formats
- API for programmatic access

#### Workflow

1. **Create Project**: Sign up at [Roboflow](https://roboflow.com/)
2. **Upload Data**: Upload images and masks
3. **Annotate**: Use web-based tools to correct masks
4. **Augment**: Apply augmentations (rotation, brightness, etc.)
5. **Export**: Download in COCO format
6. **Train**: Use exported data for training

```python
# Download dataset from Roboflow
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("your-workspace").project("document-segmentation")
dataset = project.version(1).download("coco-segmentation")
```

## Integration with .NET Core

### Using ONNX Runtime in C#

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

public class DocumentDetector
{
    private InferenceSession session;
    
    public DocumentDetector(string modelPath)
    {
        session = new InferenceSession(modelPath);
    }
    
    public byte[,] DetectDocument(string imagePath)
    {
        // Load and preprocess image
        using var image = Image.Load<Rgb24>(imagePath);
        var resized = image.Clone(x => x.Resize(320, 320));
        
        // Convert to tensor
        var input = new DenseTensor<float>(new[] { 1, 3, 320, 320 });
        for (int y = 0; y < 320; y++)
        {
            for (int x = 0; x < 320; x++)
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
            NamedOnnxValue.CreateFromTensor("input", input)
        };
        
        using var results = session.Run(inputs);
        var output = results.First().AsTensor<float>();
        
        // Convert to binary mask
        var mask = new byte[320, 320];
        for (int y = 0; y < 320; y++)
        {
            for (int x = 0; x < 320; x++)
            {
                mask[y, x] = output[0, 0, y, x] > 0.5f ? (byte)255 : (byte)0;
            }
        }
        
        return mask;
    }
}
```

### NuGet Packages for .NET

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.17.0" />
<PackageReference Include="SixLabors.ImageSharp" Version="3.1.0" />
```

## Iterative Training Loop

### Recommended Process

1. **Initial Training** (100 images)
   - Use system to detect documents
   - Export training data
   - Review and correct 10-20% of data
   - Train model

2. **Test and Evaluate**
   - Run on validation set
   - Calculate accuracy metrics
   - Identify failure cases

3. **Refinement** (Additional 50-100 images)
   - Focus on failure cases
   - Add diverse samples
   - Retrain model

4. **Repeat** until accuracy > 90%
   - Typical: 3-5 iterations
   - Dataset size: 200-500 images

### Accuracy Metrics

```python
def calculate_iou(pred_mask, true_mask):
    """Intersection over Union"""
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union > 0 else 0

def calculate_dice(pred_mask, true_mask):
    """Dice Coefficient"""
    intersection = np.logical_and(pred_mask, true_mask).sum()
    return 2 * intersection / (pred_mask.sum() + true_mask.sum())
```

Target metrics:
- **IoU > 0.90**: Excellent segmentation
- **Dice > 0.95**: High precision
- **Confidence > 0.80**: Reliable detection

## Best Practices

### Data Collection
- ✅ Diverse backgrounds (white, colored, textured)
- ✅ Various lighting conditions
- ✅ Different document types (single page, 2-page spread, torn pages)
- ✅ Multiple orientations and perspectives
- ✅ Include edge cases (shadows, creases, stains)

### Annotation Quality
- ✅ Precise mask boundaries
- ✅ Include all document regions (text + images)
- ✅ Exclude pure background
- ✅ Handle multi-page documents correctly

### Training Tips
- ✅ Start with small learning rate (0.001)
- ✅ Use data augmentation (rotation ±10°, brightness ±20%)
- ✅ Monitor validation loss (stop if overfitting)
- ✅ Save checkpoints every 10 epochs
- ✅ Use GPU for faster training (optional)

## Troubleshooting

### Common Issues

**Problem**: Model crops document content  
**Solution**: Add more training examples with similar documents, emphasize text regions in masks

**Problem**: Model misses small backgrounds  
**Solution**: Increase negative examples (documents with minimal background)

**Problem**: Poor performance on torn pages  
**Solution**: Add training samples with torn/damaged documents

## Resources

- **U2-Net Paper**: https://arxiv.org/abs/2005.09007
- **Roboflow**: https://roboflow.com/
- **ONNX Runtime**: https://onnxruntime.ai/
- **PyTorch Tutorial**: https://pytorch.org/tutorials/
- **COCO Format**: https://cocodataset.org/#format-data

## Summary

This workflow enables continuous improvement of document detection through:
1. Easy data export in standard formats
2. Web-based annotation and correction
3. Iterative training with PyTorch
4. Deployment via ONNX (CPU-compatible)
5. Integration with .NET Core applications

Target: **>90% accuracy** through 3-5 training iterations with 200-500 images.
