import cv2
import numpy as np

# Đọc ảnh đầu vào
input_path = 'input_sample_1.jpg'
output_path = 'grok_processed_'+input_path
input_image = cv2.imread(input_path)

# 1. Khử nhiễu (Denoise)
# Sử dụng Non-Local Means Denoising để giữ chi tiết văn bản
denoised_image = cv2.fastNlMeansDenoisingColored(input_image, None, 10, 10, 7, 21)

# 2. Giảm mờ (Deblur)
# Áp dụng Wiener Filter đơn giản để khôi phục chi tiết (giả định blur nhẹ)
# Tạo kernel deblur (có thể điều chỉnh kích thước kernel tùy theo mức độ blur)
kernel = np.ones((3, 3), np.float32) / 9
deblurred_image = cv2.filter2D(denoised_image, -1, kernel)

# 3. Loại bỏ watermark/nền/bóng
# Chuyển sang không gian màu HSV để tách nền đỏ
hsv_image = cv2.cvtColor(deblurred_image, cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv_image, lower_red, upper_red)
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
mask = mask1 + mask2

# Tạo ảnh không nền đỏ (giữ lại văn bản và dấu)
result_without_background = cv2.bitwise_and(deblurred_image, deblurred_image, mask=cv2.bitwise_not(mask))

# Loại bỏ watermark/bóng bằng Inpainting (điền vùng bị ảnh hưởng)
inpaint_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
cleaned_image = cv2.inpaint(result_without_background, inpaint_mask, 3, cv2.INPAINT_TELEA)

# 4. Binarization (Chuyển trắng đen)
# Áp dụng Otsu's thresholding để chuyển sang nhị phân
gray_image = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Lưu ảnh kết quả
cv2.imwrite(output_path, binary_image)

# (Tùy chọn) Hiển thị kết quả (bỏ qua nếu không chạy trực tiếp)
# cv2.imshow('Binary Output', binary_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()