# kuka-py
KUKA Image Processing Lib in Python

## Features

### 1. Document Page Detection & Cropping 🆕
Automatically detect and crop document pages from images, removing background and applying perspective correction.

**Location:** `kukalib/doc_page_crop.py`, Page: `pages/Doc-Page-Crop.py`

**Features:**
- Automatic document boundary detection
- Background removal
- Perspective correction and deskewing
- PDF support (auto-converts each page to image)
- Multiple detection methods:
  - **Auto**: Automatically selects the best method
  - **OpenCV**: Fast traditional method for simple backgrounds
  - **U2-Net**: AI-based detection for complex backgrounds (requires `rembg`)

**Use cases:**
- Smart document scanning
- Document digitization
- Pre-processing for OCR
- Archive management

**Technical Details:** See [docs/doc_page_detection_solution.md](docs/doc_page_detection_solution.md)

### 2. Card Crop
Detect and crop ID cards, bank cards, and similar documents from images.

**Location:** `kukalib/cardcrop.py`, Page: `pages/Card-Crop.py`

### 3. Document Deskew
Automatically detect and correct skewed document images.

**Location:** `kukalib/docdeskew.py`, Page: `pages/Doc-deskew.py`

### 4. Text Detection
Detect and locate text regions in images.

**Location:** `kukalib/detect.py`, Page: `pages/Detect-text.py`

## Installation

```bash
# Clone the repository
git clone https://github.com/dqhuy/kuka-py.git
cd kuka-py

# Install dependencies
pip install -r requirements.txt

# Optional: For AI-based document detection (U2-Net)
pip install rembg
```

## Usage

### Command Line

#### Document Page Crop
```bash
# Process a single image
python kukalib/doc_page_crop.py input.jpg

# Process with specific method
python kukalib/doc_page_crop.py input.jpg opencv
python kukalib/doc_page_crop.py input.jpg u2net

# Process PDF file
python kukalib/doc_page_crop.py document.pdf
```

### Python API

```python
from kukalib.doc_page_crop import detectAndCropDocumentPage
import cv2

# Load image
img = cv2.imread('document.jpg')

# Detect and crop (auto method selection)
cropped, debug, corners, method = detectAndCropDocumentPage(img, method='auto')

# Save result
cv2.imwrite('cropped.jpg', cropped)
cv2.imwrite('debug.jpg', debug)

print(f"Used method: {method}")
```

### Streamlit Demo App

```bash
streamlit run demo-app.py
```

Then navigate to:
- **Document Page Crop**: http://localhost:8501/Doc-Page-Crop
- **Card Crop**: http://localhost:8501/Card-Crop
- **Document Deskew**: http://localhost:8501/Doc-deskew
- **Text Detection**: http://localhost:8501/Detect-text

## Sample Data

Sample images for testing are available in `docs/samples/`:
- `input/` - Sample input images
- `output/` - Sample output results

For comprehensive testing, download the **Extended SmartDoc Dataset**:
- Sample: [Google Drive](https://drive.google.com/file/d/1RhfZsyZL-x3Z70kgXOarVWvaiVpxeXFi/view)
- Full: [Google Drive](https://drive.google.com/drive/folders/1G3xbBFrkNKKKnqlkGvDA4quG19rUmCnZ)
- GitHub: [Extended-Smartdoc-Dataset](https://github.com/ricardobnjunior/Extended-Smartdoc-Dataset)

## Dependencies

Core dependencies:
- numpy
- opencv-python
- matplotlib
- scikit-image
- streamlit
- PyMuPDF (for PDF support)
- Pillow

Optional (for AI-based detection):
- rembg (includes U2-Net model)
- torch
- torchvision

## Documentation

- [Document Page Detection Solution](docs/doc_page_detection_solution.md) - Technical specification and architecture
- [Sample Images README](docs/samples/README.md) - Information about test samples

## Performance

| Method | Accuracy | Speed (CPU) | Model Size | Use Case |
|--------|----------|-------------|------------|----------|
| OpenCV | 60-70% | 50-100ms | 0MB | Simple backgrounds |
| U2-Net | 85-95% | 200-500ms | 4.7MB | Complex backgrounds |
| Auto (Hybrid) | 85-95% | 200-600ms | 4.7MB | General use |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license here]

## Contact

dinhquanghuy.work@gmail.com
