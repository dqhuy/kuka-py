from kukalib.doc_page_crop import detectAndCropDocumentPage
import cv2

img_path = 'docs/samples/extracted_dataset/Vais_Test_Crop_lo6_page002_100dpi.jpg'
img = cv2.imread(img_path)
if img is None:
    print('Failed to load image:', img_path)
    exit(1)

# Call detectAndCropDocumentPage with high confidence threshold to test forwarding
cropped, debug_img, corners, method_used, time_ms, confidence = detectAndCropDocumentPage(
    img, method='yolo', model_name='yolov11s-seg', debug=True, use_content_verify=True, confidence_threshold=0.85, source_path=img_path
)

print('\nRESULT SUMMARY:')
print('method_used =', method_used)
print('time_ms =', time_ms)
print('confidence =', confidence)
print('corners =', corners)

# Save debug image
if debug_img is not None:
    cv2.imwrite('debug_yolo_test_debug.jpg', debug_img)
    print('Saved debug image: debug_yolo_test_debug.jpg')
if cropped is not None and cropped.size > 0:
    cv2.imwrite('debug_yolo_test_cropped.jpg', cropped)
    print('Saved cropped image: debug_yolo_test_cropped.jpg')
