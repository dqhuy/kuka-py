from kukalib.u2net_model import run_u2net_inference
import cv2

img_path = 'docs/samples/extracted_dataset/Vais_Test_Crop_lo6_page002_100dpi.jpg'
img = cv2.imread(img_path)
if img is None:
    print('Failed to load image:', img_path)
    exit(1)

mask = run_u2net_inference(img, model_name='u2netp', debug=True, source_path=img_path)
if mask is not None:
    cv2.imwrite('debug_u2net_mask.jpg', mask)
    print('Saved mask: debug_u2net_mask.jpg')
else:
    print('U2-Net inference failed')
