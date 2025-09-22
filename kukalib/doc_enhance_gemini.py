import cv2
import numpy as np
from skimage import filters, util
import os
import glob

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
    window_size = 9  # Kích thước cửa sổ cục bộ // Đang test với sổ đỏ thì thấy window 9 là ổn vì nó giữ lại chi tiết tốt các nét chữ in
    k = 0.2          # Tham số điều chỉnh độ nhạy
    
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
    
    denoised_image = cv2.bilateralFilter(enhanced_image, d=9, sigmaColor=75, sigmaSpace=75)

   # Giảm nhiễu hiệu quả bằng bộ lọc Non-local Means Denoising
    # Các tham số h, hColor, templateWindowSize, searchWindowSize có thể được tinh chỉnh
    #denoised_image = cv2.fastNlMeansDenoisingColored(enhanced_image, None, 10, 10, 7, 21)
    # Thực hiện nhị phân hóa Sauvola
    #final_output = sauvola_binarization(denoised_image)
    final_output = denoised_image
    # Lưu ảnh đầu ra
    cv2.imwrite(output_path, final_output)
    
    print(f"Hoàn thành xử lý ảnh. Ảnh đầu ra đã được lưu tại {output_path}")

def process_document2(input_path, output_path):
    """
    Tải ảnh, xử lý và lưu kết quả.
    """
    # Đọc ảnh đầu vào
    image = cv2.imread(input_path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn {input_path}")
        return

    # Chuyển đổi không gian màu sang thang độ xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Áp dụng làm sắc nét (Unsharp Masking)
    # Bước này giúp tăng cường độ tương phản của các cạnh và làm nổi bật chữ
    gaussian = cv2.GaussianBlur(gray_image, (0, 0), 3.0)
    sharpened_image = cv2.addWeighted(gray_image, 1.5, gaussian, -0.5, 0)
    
    # Áp dụng bộ lọc Gaussian để làm mịn nhẹ
    denoised_image = cv2.GaussianBlur(sharpened_image, (3, 3), 0)

    # Chuyển đổi lại sang ảnh màu để phù hợp với hàm sauvola_confidence_binarization
    denoised_image_color = cv2.cvtColor(denoised_image, cv2.COLOR_GRAY2BGR)

    # Thực hiện nhị phân hóa bằng phương pháp Sauvola Confidence
    final_output = denoised_image_color

    # Lưu ảnh đầu ra
    cv2.imwrite(output_path, final_output)
    
    print(f"Hoàn thành xử lý ảnh. Ảnh đầu ra đã được lưu tại {output_path}")

    # Hiển thị ảnh gốc và ảnh kết quả
    # cv2.imshow('Anh Goc', cv2.resize(cv2.imread(input_path), (600, 800)))
    # cv2.imshow('Ket qua cuoi cung', cv2.resize(final_output, (600, 800)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
def process_document3(input_path, output_path):
    """
    Tải ảnh, xử lý và lưu kết quả.
    """
    # Đọc ảnh đầu vào
    image = cv2.imread(input_path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn {input_path}")
        return

    # Chuyển đổi không gian màu sang thang độ xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Bước 1: Giảm nhiễu nền bằng Gaussian Blur
    # Sử dụng một kernel lớn để làm mờ nền hiệu quả
    blurred_background = cv2.GaussianBlur(gray_image, (15, 15), 0)
    
    # Bước 2: Tăng cường độ sắc nét cục bộ của văn bản
    # Bằng cách trừ đi phần nền đã làm mờ
    sharpened_text = cv2.addWeighted(gray_image, 1.5, blurred_background, -0.5, 0)
    
    # Bước 3: Áp dụng CLAHE để tăng cường độ tương phản cuối cùng cho văn bản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    final_image = clahe.apply(sharpened_text)

    # Lưu ảnh đầu ra
    cv2.imwrite(output_path, final_image)
    
    print(f"Hoàn thành xử lý ảnh. Ảnh đầu ra đã được lưu tại {output_path}")

    # Hiển thị ảnh gốc và ảnh kết quả
    # cv2.imshow('Anh Goc', cv2.resize(cv2.imread(input_path), (600, 800)))
    # cv2.imshow('Ket qua cuoi cung', cv2.resize(final_image, (600, 800)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
def process_document4(input_path, output_path):
    """
    Tải ảnh, xử lý và lưu kết quả.
    """
    # Đọc ảnh đầu vào
    image = cv2.imread(input_path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn {input_path}")
        return

    # Chuyển đổi sang không gian màu HSV để xử lý màu nền
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Định nghĩa dải màu đỏ
    # OpenCV sử dụng dải màu 0-179, nên màu đỏ nằm ở hai đầu
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([179, 255, 255])

    # Tạo mask để phát hiện màu đỏ
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Đảo ngược mask để giữ lại các phần không phải màu đỏ (văn bản)
    non_red_mask = cv2.bitwise_not(red_mask)
    
    # Chuyển đổi ảnh gốc sang thang độ xám để xử lý
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Áp dụng làm sắc nét (Unsharp Masking)
    gaussian = cv2.GaussianBlur(gray_image, (0, 0), 3.0)
    sharpened_image = cv2.addWeighted(gray_image, 1.5, gaussian, -0.5, 0)
    
    # Tạo một ảnh nền trắng
    final_image = np.ones_like(sharpened_image) * 255
    
    # Sao chép các pixel từ ảnh đã làm sắc nét vào ảnh nền trắng, chỉ ở các vùng không phải màu đỏ
    cv2.copyTo(sharpened_image, non_red_mask, final_image)
    
    # Lưu ảnh đầu ra
    cv2.imwrite(output_path, final_image)
    
    print(f"Hoàn thành xử lý ảnh. Ảnh đầu ra đã được lưu tại {output_path}")

    # Hiển thị ảnh gốc và ảnh kết quả
    # cv2.imshow('Anh Goc', cv2.resize(cv2.imread(input_path), (600, 800)))
    # cv2.imshow('Ket qua cuoi cung', cv2.resize(final_image, (600, 800)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    # Đường dẫn tới thư mục chứa ảnh
    input_folder = "D:\\CSDL_SOHOA\\so-do\\30775-GCN-41-79_images"
    
    # Lấy toàn bộ các file ảnh trong thư mục
    # glob.glob hỗ trợ tìm kiếm file với wildcard
    image_files = glob.glob(os.path.join(input_folder, '*'))

    processCase = 2
    # Duyệt qua từng file
    for file_path in image_files:
        # Lấy tên file
        filename = os.path.basename(file_path)

        # Bỏ qua các file đã xử lý
        if filename.startswith('processed_'):
            continue

        # Đặt tên file đầu ra
        output_filename = 'processed_' + str(processCase) + '_' + filename
        output_path = os.path.join(input_folder, output_filename)

        # Xử lý file ảnh
        print(f"Đang xử lý file: {file_path}")
        if processCase == 1:
            process_document(file_path, output_path)
        elif processCase == 2:
            process_document2(file_path, output_path)
        elif processCase == 3:
            process_document3(file_path, output_path)
        elif processCase == 4:
            process_document4(file_path, output_path)
