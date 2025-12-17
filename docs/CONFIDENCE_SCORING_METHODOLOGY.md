# Confidence Scoring Methodology

## Overview

This document describes how the document detection system calculates confidence scores and makes cropping decisions.

**V4 Update**: Confidence threshold increased to **>90%** for maximum safety (per user feedback).

## Confidence Score Calculation

The confidence score is calculated as a weighted average of 4 factors:

### 1. Area Ratio (35% weight)

**Measures**: How much of the image the detected document occupies

```python
area_ratio = detected_document_area / total_image_area

if area_ratio > 0.80:
    area_factor = 0.35  # High confidence - document fills image
elif area_ratio > 0.50:
    area_factor = 0.30  # Good confidence
elif area_ratio > 0.20:
    area_factor = 0.20  # Medium confidence
else:
    area_factor = 0.10  # Low confidence - small detection
```

**Rationale**: Larger detections are more likely to be correct. Very small detections might be noise or artifacts.

### 2. Mask Quality (25% weight)

**Measures**: Quality of the U2-Net segmentation mask

```python
mask_fill_ratio = white_pixels_in_mask / bbox_area

if mask_fill_ratio > 0.85:
    mask_factor = 0.25  # High quality - mask fills bbox well
elif mask_fill_ratio > 0.70:
    mask_factor = 0.20  # Good quality
elif mask_fill_ratio > 0.50:
    mask_factor = 0.15  # Medium quality
else:
    mask_factor = 0.05  # Low quality - sparse mask
```

**Rationale**: A good mask should fill most of the bounding box. Sparse masks indicate uncertain detection.

### 3. Aspect Ratio (20% weight)

**Measures**: Whether the detected shape has document-like proportions

```python
aspect_ratio = width / height
# Standard document ratios: A4 (1.41), Letter (1.29), etc.

if 1.0 < aspect_ratio < 2.0:
    aspect_factor = 0.20  # Standard document ratio
elif 0.5 < aspect_ratio < 2.5:
    aspect_factor = 0.15  # Acceptable ratio
else:
    aspect_factor = 0.05  # Unusual ratio
```

**Rationale**: Documents typically have aspect ratios between 1:1 and 2:1. Extreme ratios suggest false detection.

### 4. Multi-Region Handling (20% weight)

**Measures**: Whether detection is single or multi-page

```python
if num_regions == 1:
    region_factor = 0.20  # Single document - high confidence
else:
    region_factor = 0.15  # Multiple regions - moderate confidence
```

**Rationale**: Single-region detections are simpler and more reliable. Multi-region (e.g., 2-page spreads) are handled but with slightly lower confidence.

### Final Confidence Score

```python
confidence = area_factor + mask_factor + aspect_factor + region_factor
# Range: 0.0 to 1.0
```

## Cropping Decision Logic

### V4 Rules (Current - VERY CONSERVATIVE)

**Rule 1**: Document occupies >92% of image → **SKIP CROP**
- Reason: No background to remove
- Confidence: Set to 0.85+

**Rule 2**: Confidence < 0.90 → **SKIP CROP** ⭐
- Reason: **V4 requirement - must be >90% confident**
- This is the primary safety check

**Rule 3**: Would remove >70% of image → **SKIP CROP**
- Reason: Too aggressive, likely wrong detection
- Extra safety against false crops

### Previous Versions (for reference)

**V3**: Confidence < 0.30 to skip (much more aggressive)
**V2**: Confidence < 0.30 OR (remove >70% AND confidence < 0.50)
**V1**: Fixed 5-10% margins, no confidence scoring

## Alternative Confidence Metrics (Advanced)

### Option 1: Content-Based Cross-Validation

Add a 5th factor based on Content-Based method validation:

```python
# Run Content-Based detection
content_confidence = detectDocumentPage_ContentBased(img)

# Compare results
if abs(u2net_bbox - content_bbox) < threshold:
    validation_factor = 0.20  # Results agree
else:
    validation_factor = 0.05  # Results disagree - suspicious

# New confidence calculation
confidence = area_factor + mask_factor + aspect_factor + 
             region_factor + validation_factor
# Range: 0.0 to 1.2 (then normalize to 0.0-1.0)
```

**Pros**: Cross-validation catches errors
**Cons**: Slower (2x processing), complex logic

### Option 2: Edge-Based Verification

Check if removed regions contain edges (potential text):

```python
# Analyze edges in removed regions
edge_density = count_edges_in_removed_area / removed_area

if edge_density < 0.05:
    edge_factor = 0.20  # Low edge density - safe to remove
else:
    edge_factor = 0.05  # High edge density - might contain content

confidence = (confidence + edge_factor) / 1.2  # Normalize
```

**Pros**: Directly checks for content in removed areas
**Cons**: Sensitive to noise, thresholding challenges

### Option 3: Color Similarity Analysis

Compare color distributions between document and removed regions:

```python
# Calculate color histograms
doc_hist = cv2.calcHist(document_region)
bg_hist = cv2.calcHist(background_region)

# Compare similarity
similarity = cv2.compareHist(doc_hist, bg_hist, cv2.HISTCMP_CORREL)

if similarity < 0.3:
    color_factor = 0.20  # Different colors - clear boundary
else:
    color_factor = 0.05  # Similar colors - unclear boundary

confidence = (confidence + color_factor) / 1.2  # Normalize
```

**Pros**: Detects subtle boundaries
**Cons**: Fails with uniform-color documents

### Option 4: Machine Learning Confidence

Train a classifier on correctly/incorrectly cropped images:

```python
# Extract features
features = [
    area_ratio,
    mask_quality,
    aspect_ratio,
    edge_density,
    color_variance,
    ...
]

# Trained model prediction
ml_confidence = trained_model.predict_proba(features)[1]

# Combine with geometric confidence
final_confidence = 0.7 * geometric_confidence + 0.3 * ml_confidence
```

**Pros**: Learns from data, adapts over time
**Cons**: Requires training data, more complex

## Recommendations

### Current V4 Approach (90% threshold)

**Strengths**:
- Very conservative - minimizes false crops
- Simple to understand and tune
- No dependencies on training data

**Weaknesses**:
- Might skip too many valid crops
- No cross-validation
- Doesn't analyze removed content

### Recommended Improvements

1. **Short-term**: Implement Option 2 (Edge-Based Verification)
   - Add edge density check to existing pipeline
   - Minimal code changes
   - Significant safety improvement

2. **Medium-term**: Implement Option 1 (Content-Based Cross-Validation)
   - Use existing Content-Based method
   - Catches errors U2-Net misses
   - Moderate complexity increase

3. **Long-term**: Implement Option 4 (ML Confidence)
   - Train on collected dataset
   - Continuous improvement
   - Requires infrastructure

## Testing Strategy

### Validation Metrics

1. **False Positive Rate**: Crops that shouldn't have been made
2. **False Negative Rate**: Valid crops that were skipped
3. **Precision**: Accuracy of bounding boxes
4. **Consistency**: Same result at different resolutions

### Test Suite

```bash
# Run comprehensive test on Vais dataset
python test_vais_dataset.py

# Expected results with 90% threshold:
# - Most images should SKIP (conservative)
# - Zero false crops into content
# - Consistent between 50 DPI and 100 DPI
```

### Threshold Tuning

If too conservative (skips valid crops):
- Reduce threshold: 0.90 → 0.85 → 0.80
- Add cross-validation to increase confidence
- Implement edge-based verification

If too aggressive (false crops):
- Increase threshold: 0.90 → 0.95
- Add more safety rules
- Implement content analysis

## Conclusion

The V4 confidence scoring system uses a **4-factor weighted average** with a **90% threshold** for maximum safety.

Future improvements should focus on **cross-validation** and **content analysis** to increase confidence in valid detections while maintaining zero false crop rate.

**Key Principle**: When in doubt, DON'T crop. Preserving content is more important than perfect cropping.
