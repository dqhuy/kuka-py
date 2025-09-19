import cv2
import numpy as np
from skimage import filters, util

def sauvola_binarization(image):
    """
    Áp dụng thuật toán nhị phân hóa cục bộ Sauvola để tạo ảnh nền trắng chữ đen.
    """
    # Chuyển ảnh sang thang độ xám nếu chưa phải
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Áp dụng thuật toán Sauvola để tạo ảnh nhị phân
    window_size = 11  # Kích thước cửa sổ cục bộ
    k = 0.15          # Tham số điều chỉnh độ nhạy
    
    # Skimage thực hiện nhị phân hóa Sauvola rất hiệu quả
    image_sk = util.img_as_float(gray)
    t = filters.threshold_sauvola(image_sk, window_size=window_size, k=k)
    binarized_image = image_sk > t
    
    # Chuyển đổi kiểu dữ liệu boolean sang uint8 (0 hoặc 255) trước khi sử dụng
    binarized_image_uint8 = binarized_image.astype(np.uint8) * 255
    
    # Trực tiếp trả về ảnh đã được chuyển đổi
    return binarized_image_uint8

def process_document(input_path, output_path):
    """
    Tải ảnh, xử lý và lưu kết quả.
    """
    # Đọc ảnh đầu vào
    image = cv2.imread(input_path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn {input_path}")
        return

    # Chuyển đổi không gian màu sang LAB để xử lý ánh sáng và màu sắc riêng biệt
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization) cho kênh L
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    
    # Chuyển đổi ngược lại không gian màu
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Giảm nhiễu hiệu quả bằng bộ lọc Bilateral
    denoised_image = cv2.bilateralFilter(enhanced_image, d=9, sigmaColor=75, sigmaSpace=75)

    # Thực hiện nhị phân hóa Sauvola
    final_output = sauvola_binarization(denoised_image)

    # Lưu ảnh đầu ra
    cv2.imwrite(output_path, final_output)
    
    print(f"Hoàn thành xử lý ảnh. Ảnh đầu ra đã được lưu tại {output_path}")

    # Hiển thị ảnh gốc và ảnh kết quả
    cv2.imshow('Anh Goc', cv2.resize(cv2.imread(input_path), (600, 800)))
    cv2.imshow('Ket qua cuoi cung', cv2.resize(final_output, (600, 800)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    input_image_path = 'input_sample_2.jpg'
    output_image_path = 'processed_'+input_image_path
    process_document(input_image_path, output_image_path)
