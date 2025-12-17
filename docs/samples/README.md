# Sample Images for Document Page Detection

## Input Samples

This directory contains sample images for testing the document page detection and cropping feature.

### Sample Images

1. **sample1_simple_document.jpg**
   - Simple white document on dark background
   - Slight rotation (~5 degrees)
   - Clear contrast between document and background
   - Good test case for basic detection

2. **sample2_textured_background.jpg**
   - Document on textured, noisy background
   - More rotation (~8 degrees)
   - Tests robustness against background texture
   - Medium difficulty

3. **sample3_perspective_document.jpg**
   - Document with perspective distortion
   - Simulates angled camera view
   - Gradient background
   - Tests perspective correction capability
   - Higher difficulty

## About SmartDoc Dataset

For more comprehensive testing, you can use the **Extended SmartDoc Dataset**, which is the standard benchmark for document detection research.

### Download Extended SmartDoc Dataset

- **Sample Dataset**: [Google Drive Link](https://drive.google.com/file/d/1RhfZsyZL-x3Z70kgXOarVWvaiVpxeXFi/view?usp=sharing)
- **Full Dataset**: [Google Drive Link](https://drive.google.com/drive/folders/1G3xbBFrkNKKKnqlkGvDA4quG19rUmCnZ?usp=sharing)
- **GitHub Repository**: [Extended-Smartdoc-Dataset](https://github.com/ricardobnjunior/Extended-Smartdoc-Dataset)

### Dataset Characteristics

The Extended SmartDoc dataset includes:
- 213 varied backgrounds
- Multiple smartphone devices
- Various lighting conditions
- Different rotations and perspectives
- Real-world challenging scenarios

### Citation

If you use the Extended SmartDoc dataset, please cite:

```bibtex
@article{jemni2019extended,
  title={Extended SmartDoc Dataset},
  author={Jemni, Skander K and others},
  journal={GitHub repository},
  year={2019}
}
```

## Output Samples

The `output/` directory will contain processed results showing:
- Original image
- Detection visualization (corners/edges)
- Cropped and perspective-corrected document

## Usage

To test with these samples:

```python
from kukalib.doc_page_crop import detectAndCropDocumentPage
import cv2

# Load a sample image
img = cv2.imread('docs/samples/input/sample1_simple_document.jpg')

# Detect and crop
cropped, debug, corners, method = detectAndCropDocumentPage(img)

# Save results
cv2.imwrite('docs/samples/output/sample1_output.jpg', cropped)
cv2.imwrite('docs/samples/output/sample1_debug.jpg', debug)
```

## Performance Expectations

| Sample | Expected Method | Expected Accuracy | Notes |
|--------|----------------|-------------------|-------|
| Sample 1 | Either method works | > 95% | Easy case |
| Sample 2 | U2-Net preferred | > 85% | Texture challenges OpenCV |
| Sample 3 | U2-Net required | > 90% | Perspective needs AI |

## Additional Test Cases

For thorough testing, consider:
1. **Different document types**: Forms, receipts, letters, invoices
2. **Various backgrounds**: Wood tables, carpets, marble, outdoor scenes
3. **Lighting conditions**: Bright, dim, side-lit, back-lit
4. **Orientations**: Portrait, landscape, rotated
5. **Quality**: High-res, low-res, blurry, sharp
6. **Multi-page PDFs**: Test PDF conversion and batch processing
