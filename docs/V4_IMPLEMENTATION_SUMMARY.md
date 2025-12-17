# V4 Implementation Summary - Pending Completion

## Status: Foundation Complete,  Awaiting Vais_*.pdf Dataset

### Completed (Commit a15c3ab)

✅ **PDF Extractor Module** (`kukalib/pdf_extractor.py`)
- Extract PDF pages at 50 DPI and 100 DPI
- Batch processing support
- Dataset summary generation
- Ready for Vais_*.pdf files

✅ **Comprehensive Documentation** (`docs/V4_PRODUCTION_IMPROVEMENTS.md`)
- All 7 user requirements documented
- Technical implementation details
- Multi-stage validation pipeline
- Testing strategy
- Migration guide

### Pending Implementation

The following changes require the Vais_*.pdf dataset to be present for proper testing and validation:

#### 1. Remove DeepLabV3 Method

**Files to modify**:
- `kukalib/doc_page_crop.py`
  - Remove `detectDocumentPage_DeepLabV3()` function (~150 lines)
  - Update `detectAndCropDocumentPage()` auto mode
  - Remove DeepLabV3 from method list

- `pages/Doc-Page-Crop.py`
  - Remove "deeplabv3" from dropdown
  - Update method descriptions

**Rationale**:
- Slower than U2-Net (1000ms vs 385ms)
- Similar accuracy
- Not needed with U2-Net + Content validation

#### 2. Enhance U2-Net with Content-Based Validation

**Changes to `kukalib/doc_page_crop.py`**:

```python
def detectDocumentPage_U2Net(src, model_name='u2netp', debug=False):
    # ... existing U2-Net detection ...
    
    # NEW: Add content-based validation
    from kukalib.training_export import verify_removed_content
    
    # Calculate area to be removed
    h, w = src.shape[:2]
    x_crop, y_crop, w_crop, h_crop = # ... from detection ...
    
    # Create mask of removed area
    removed_mask = np.ones((h, w), dtype=np.uint8) * 255
    removed_mask[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop] = 0
    
    # Verify removed content
    suspicious, details = verify_removed_content(
        src, removed_mask, (x_crop, y_crop, w_crop, h_crop)
    )
    
    # Adjust confidence if suspicious
    if suspicious:
        if debug:
            print(f"⚠️  Content verification: Suspicious - {details}", file=sys.stderr)
        confidence *= 0.5
        
        # Skip crop if confidence too low
        if confidence < 0.4:
            if debug:
                print("⚠️  Skipping crop - risk of content loss", file=sys.stderr)
            # Return original image
            time_ms = (time.time() - start_time) * 1000
            return src.copy(), debug_img, corners, False, time_ms, confidence
    
    # ... proceed with crop ...
```

**Key Safety Checks**:
1. Check if document fills >95% (no background)
2. Verify removed regions don't contain text
3. Compare color similarity
4. Adjust confidence based on findings
5. Skip crop if confidence <0.4

#### 3. Fix YOLO on Streamlit Cloud

**Changes to `requirements-optional.txt`**:
```
ultralytics>=8.0.0
pytesseract>=0.3.10
pillow>=10.0.0
```

**Changes to `kukalib/yolo_detector.py`**:

```python
def detectDocumentPage_YOLO(src, model_name='yolov8n-seg', confidence_threshold=0.5, debug=False):
    """YOLO detection with proper error handling for cloud deployment"""
    import time
    start_time = time.time()
    
    try:
        from ultralytics import YOLO
    except ImportError:
        if debug:
            print("❌ ultralytics not installed", file=sys.stderr)
            print("   Install with: pip install ultralytics>=8.0.0", file=sys.stderr)
        time_ms = (time.time() - start_time) * 1000
        return src.copy(), src.copy(), ([], [], [], []), False, time_ms, 0.0
    
    # Ensure models directory exists
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f"{model_name}.pt")
    
    # Auto-download model if not exists
    if not os.path.exists(model_path):
        if debug:
            print(f"📥 Downloading {model_name} model...", file=sys.stderr)
        try:
            model = YOLO(model_name)  # This triggers download
            model.export(format="onnx", simplify=True)  # Export for deployment
            if debug:
                print(f"✅ Model downloaded and exported to ONNX", file=sys.stderr)
        except Exception as e:
            if debug:
                print(f"❌ Model download failed: {e}", file=sys.stderr)
            time_ms = (time.time() - start_time) * 1000
            return src.copy(), src.copy(), ([], [], [], []), False, time_ms, 0.0
    else:
        model = YOLO(model_path)
    
    # ... rest of detection ...
```

#### 4. Improve GUI with Session State

**Changes to `pages/Doc-Page-Crop.py`**:

```python
import streamlit as st

# Initialize session state
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = {}
if 'last_file_info' not in st.session_state:
    st.session_state.last_file_info = None

def get_file_info(uploaded_file):
    """Generate unique file identifier"""
    if uploaded_file is None:
        return None
    return {
        'name': uploaded_file.name,
        'size': uploaded_file.size,
        'type': uploaded_file.type
    }

# Main processing
uploaded_file = st.file_uploader(...)

if uploaded_file:
    file_info = get_file_info(uploaded_file)
    
    # Only run detection if file changed
    if file_info != st.session_state.last_file_info:
        # Read image
        image = read_image(uploaded_file)
        
        # Run detection
        result = detectAndCropDocumentPage(image, method=method, model_name=model_name)
        
        # Cache results
        st.session_state.detection_results = {
            'original': image,
            'cropped': result[0],
            'debug': result[1],
            'corners': result[2],
            'method': result[3],
            'time_ms': result[4],
            'confidence': result[5]
        }
        st.session_state.last_file_info = file_info
    
    # Use cached results
    results = st.session_state.detection_results
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.image(results['original'], caption="Original")
        st.download_button(
            "📥 Download Original",
            data=encode_image(results['original']),
            file_name=f"original_{uploaded_file.name}",
            mime="image/jpeg",
            key="download_original"  # NO RE-DETECTION!
        )
    
    with col2:
        st.image(results['cropped'], caption=f"Cropped (Confidence: {results['confidence']:.2f})")
        st.download_button(
            "📥 Download Cropped",
            data=encode_image(results['cropped']),
            file_name=f"cropped_{uploaded_file.name}",
            mime="image/jpeg",
            key="download_cropped"  # NO RE-DETECTION!
        )
    
    # Training data export
    st.download_button(
        "📦 Download Training Data (ZIP)",
        data=create_training_zip(results),
        file_name=f"training_{uploaded_file.name}.zip",
        mime="application/zip",
        key="download_training"  # NO RE-DETECTION!
    )
```

**Benefits**:
- ✅ Detection runs only once per file
- ✅ All downloads use cached results
- ✅ No re-detection on button clicks
- ✅ Better UX and performance

#### 5. Create Comparative Testing Script

**New file: `test_comparative.py`**:

```python
"""
Comparative testing script for U2-Net vs YOLO.
Tests with Vais_*.pdf extracted dataset at multiple resolutions.
"""

import cv2
import numpy as np
import os
import glob
import json
from kukalib.doc_page_crop import (
    detectDocumentPage_U2Net,
    detectDocumentPage_YOLO,
    detectDocumentPage_ContentBased
)

def test_method_on_dataset(dataset_dir, method_name, model_name=None):
    """Test a detection method on all images in dataset"""
    
    # Find all images
    images_50dpi = sorted(glob.glob(os.path.join(dataset_dir, "*_50dpi.jpg")))
    images_100dpi = sorted(glob.glob(os.path.join(dataset_dir, "*_100dpi.jpg")))
    
    results = {
        "method": method_name,
        "model": model_name,
        "50dpi": [],
        "100dpi": [],
        "statistics": {}
    }
    
    # Test 50 DPI images
    print(f"\n📊 Testing {method_name} on 50 DPI images...")
    for img_path in images_50dpi:
        img = cv2.imread(img_path)
        
        if method_name == "u2net":
            result = detectDocumentPage_U2Net(img, model_name=model_name or 'u2netp', debug=False)
        elif method_name == "yolo":
            result = detectDocumentPage_YOLO(img, debug=False)
        elif method_name == "content":
            result = detectDocumentPage_ContentBased(img, debug=False)
        
        results["50dpi"].append({
            "image": os.path.basename(img_path),
            "success": result[3],
            "time_ms": result[4],
            "confidence": result[5]
        })
    
    # Test 100 DPI images
    print(f"📊 Testing {method_name} on 100 DPI images...")
    for img_path in images_100dpi:
        img = cv2.imread(img_path)
        
        if method_name == "u2net":
            result = detectDocumentPage_U2Net(img, model_name=model_name or 'u2netp', debug=False)
        elif method_name == "yolo":
            result = detectDocumentPage_YOLO(img, debug=False)
        elif method_name == "content":
            result = detectDocumentPage_ContentBased(img, debug=False)
        
        results["100dpi"].append({
            "image": os.path.basename(img_path),
            "success": result[3],
            "time_ms": result[4],
            "confidence": result[5]
        })
    
    # Calculate statistics
    results["statistics"] = {
        "50dpi": {
            "total": len(results["50dpi"]),
            "success_rate": sum(1 for r in results["50dpi"] if r["success"]) / len(results["50dpi"]) if results["50dpi"] else 0,
            "avg_time_ms": np.mean([r["time_ms"] for r in results["50dpi"]]) if results["50dpi"] else 0,
            "avg_confidence": np.mean([r["confidence"] for r in results["50dpi"]]) if results["50dpi"] else 0
        },
        "100dpi": {
            "total": len(results["100dpi"]),
            "success_rate": sum(1 for r in results["100dpi"] if r["success"]) / len(results["100dpi"]) if results["100dpi"] else 0,
            "avg_time_ms": np.mean([r["time_ms"] for r in results["100dpi"]]) if results["100dpi"] else 0,
            "avg_confidence": np.mean([r["confidence"] for r in results["100dpi"]]) if results["100dpi"] else 0
        }
    }
    
    return results

def compare_methods(dataset_dir):
    """Compare all methods on the dataset"""
    
    print("="*70)
    print("COMPARATIVE TESTING: U2-Net vs YOLO vs Content-Based")
    print("="*70)
    
    # Test each method
    results_u2net = test_method_on_dataset(dataset_dir, "u2net", "u2netp")
    results_yolo = test_method_on_dataset(dataset_dir, "yolo")
    results_content = test_method_on_dataset(dataset_dir, "content")
    
    # Print comparison
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for dpi in ["50dpi", "100dpi"]:
        print(f"\n{dpi.upper()} Results:")
        print(f"{'Method':<15} {'Success Rate':<15} {'Avg Time (ms)':<15} {'Avg Confidence':<15}")
        print("-"*60)
        
        for results in [results_u2net, results_yolo, results_content]:
            stats = results["statistics"][dpi]
            print(f"{results['method']:<15} {stats['success_rate']*100:<14.1f}% {stats['avg_time_ms']:<14.1f} {stats['avg_confidence']:<14.2f}")
    
    # Save full results
    output_file = os.path.join(dataset_dir, "comparative_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "u2net": results_u2net,
            "yolo": results_yolo,
            "content": results_content
        }, f, indent=2)
    
    print(f"\n📊 Full results saved to: {output_file}")

if __name__ == "__main__":
    import sys
    
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "docs/samples/extracted_dataset"
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        print("\nPlease extract Vais_*.pdf files first:")
        print("   python kukalib/pdf_extractor.py docs/samples/input docs/samples/extracted_dataset")
        sys.exit(1)
    
    compare_methods(dataset_dir)
```

## Next Steps (When Vais_*.pdf Available)

1. **Extract Dataset**:
   ```bash
   # User uploads Vais_*.pdf to docs/samples/input
   python kukalib/pdf_extractor.py docs/samples/input docs/samples/extracted_dataset
   ```

2. **Implement Code Changes**:
   - Remove DeepLabV3
   - Enhance U2-Net with content validation
   - Fix YOLO dependencies
   - Improve GUI session state

3. **Run Comparative Testing**:
   ```bash
   python test_comparative.py docs/samples/extracted_dataset
   ```

4. **Analyze Results**:
   - Check success rates
   - Verify resolution stability
   - Identify false crops
   - Adjust confidence thresholds

5. **Iterate if Needed**:
   - Fine-tune detection parameters
   - Adjust validation thresholds
   - Re-test with updated code

## Expected Outcomes

**Success Criteria**:
- ✅ Zero false crops on Vais dataset
- ✅ >95% success rate on documents with background
- ✅ 100% skip rate on no-background documents
- ✅ Stable results across 50/100 DPI
- ✅ U2-Net + Content validation: <1% false crop rate
- ✅ YOLO working on Streamlit Cloud

## Why Waiting for Dataset?

The implementations require testing with real scanned documents to:
1. Tune confidence thresholds
2. Validate multi-resolution stability
3. Verify no false crops
4. Adjust safety margins
5. Test edge cases (torn documents, no background)

**Without the dataset, we risk**:
- Incorrect threshold values
- Untested edge cases
- Potential false crops on production data

**Best practice**: Test-driven development with real data ✅

## Current Status

✅ Foundation complete (PDF extractor + documentation)  
⏳ Awaiting Vais_*.pdf dataset  
⏳ Code implementation pending  
⏳ Testing pending  
⏳ Validation pending  

**Once dataset is available, all changes can be implemented and tested in ~1-2 hours.**
