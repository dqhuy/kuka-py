"""
U2-Net Model Implementation for Document Detection
Standalone implementation to avoid rembg compatibility issues
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional

# Model will be downloaded to this directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'u2netp.onnx')
MODEL_URL = 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx'

def download_model(debug=False):
    """Download U2-Net model if not exists"""
    if os.path.exists(MODEL_PATH):
        if debug:
            print(f"U2-Net model already exists at: {MODEL_PATH}")
        return True
    
    try:
        import urllib.request
        if debug:
            print(f"Downloading U2-Net model to: {MODEL_PATH}")
            print(f"This may take a few minutes...")
        
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        
        if debug:
            print(f"✅ Model downloaded successfully!")
            print(f"Model size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB")
        
        return True
    except Exception as e:
        if debug:
            print(f"❌ Failed to download model: {e}")
        return False


def load_u2net_model(debug=False):
    """Load U2-Net model using ONNX runtime for better compatibility"""
    try:
        import onnxruntime as ort
        
        if not os.path.exists(MODEL_PATH):
            if debug:
                print("Model not found, attempting download...")
            if not download_model(debug):
                return None
        
        if debug:
            print(f"Loading U2-Net model from: {MODEL_PATH}")
        
        # Create ONNX runtime session
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        
        if debug:
            print("✅ U2-Net model loaded successfully")
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            print(f"Input name: {input_name}, shape: {input_shape}")
        
        return session
    
    except ImportError:
        if debug:
            print("❌ onnxruntime not installed. Install with: pip install onnxruntime")
        return None
    except Exception as e:
        if debug:
            print(f"❌ Error loading U2-Net model: {e}")
            import traceback
            traceback.print_exc()
        return None


def preprocess_image(image: np.ndarray, target_size=(320, 320), debug=False) -> np.ndarray:
    """Preprocess image for U2-Net"""
    if debug:
        print(f"Preprocessing image: {image.shape}")
    
    # Resize
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    
    # Transpose to CHW format
    chw = np.transpose(normalized, (2, 0, 1))
    
    # Add batch dimension
    batch = np.expand_dims(chw, axis=0)
    
    if debug:
        print(f"Preprocessed shape: {batch.shape}")
    
    return batch


def postprocess_mask(mask: np.ndarray, original_shape: Tuple[int, int], debug=False) -> np.ndarray:
    """Postprocess U2-Net output mask"""
    if debug:
        print(f"Postprocessing mask: {mask.shape}")
    
    # Remove batch dimension and get first channel
    if len(mask.shape) == 4:
        mask = mask[0, 0, :, :]
    elif len(mask.shape) == 3:
        mask = mask[0, :, :]
    
    # Normalize to [0, 255]
    mask = (mask * 255).astype(np.uint8)
    
    # Resize to original shape
    h, w = original_shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    
    if debug:
        print(f"Postprocessed mask shape: {mask_resized.shape}")
        print(f"Mask value range: [{mask_resized.min()}, {mask_resized.max()}]")
    
    return mask_resized


def run_u2net_inference(image: np.ndarray, debug=False) -> Optional[np.ndarray]:
    """Run U2-Net inference on image"""
    if debug:
        print("="*60)
        print("U2-Net Inference Starting")
        print("="*60)
    
    # Load model
    session = load_u2net_model(debug=debug)
    if session is None:
        if debug:
            print("❌ Failed to load U2-Net model")
        return None
    
    # Preprocess
    input_data = preprocess_image(image, debug=debug)
    
    try:
        # Run inference
        if debug:
            print("Running inference...")
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        outputs = session.run([output_name], {input_name: input_data})
        mask = outputs[0]
        
        if debug:
            print(f"✅ Inference complete. Output shape: {mask.shape}")
        
        # Postprocess
        mask_final = postprocess_mask(mask, image.shape, debug=debug)
        
        if debug:
            print("="*60)
        
        return mask_final
    
    except Exception as e:
        if debug:
            print(f"❌ Inference error: {e}")
            import traceback
            traceback.print_exc()
        return None


def test_u2net(debug=True):
    """Test U2-Net model availability and basic functionality"""
    print("\n" + "="*60)
    print("U2-Net Model Test")
    print("="*60)
    
    # Check if model exists
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model file exists: {MODEL_PATH}")
        print(f"   Size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB")
    else:
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("   Attempting to download...")
        if download_model(debug=True):
            print("✅ Model downloaded successfully")
        else:
            print("❌ Model download failed")
            return False
    
    # Check if onnxruntime is installed
    try:
        import onnxruntime
        print(f"✅ onnxruntime installed: version {onnxruntime.__version__}")
    except ImportError:
        print("❌ onnxruntime not installed")
        print("   Install with: pip install onnxruntime")
        return False
    
    # Try to load model
    session = load_u2net_model(debug=True)
    if session is None:
        print("❌ Failed to load model")
        return False
    
    print("✅ U2-Net model ready for use!")
    print("="*60 + "\n")
    return True


if __name__ == "__main__":
    # Run test
    test_u2net()
