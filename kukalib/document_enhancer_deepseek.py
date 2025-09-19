import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

class AdvancedDocumentEnhancer:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        
    def load_image(self, image_path):
        """Tải ảnh từ đường dẫn"""
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Không thể tải ảnh từ {image_path}")
        self.processed_image = self.original_image.copy()
        return self.original_image
    
    def remove_red_background(self, image):
        """
        Loại bỏ nền đỏ và các màu nền không mong muốn
        """
        # Chuyển sang HSV để dễ dàng xác định màu đỏ
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Xác định phạm vi màu đỏ trong HSV
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Tạo mask cho màu đỏ
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Mở rộng mask để chắc chắn bao phủ hết nền đỏ
        kernel = np.ones((5,5), np.uint8)
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)
        
        # Tạo ảnh trắng để thay thế nền đỏ
        white_background = np.ones_like(image) * 255
        
        # Kết hợp: giữ nguyên ảnh gốc ở vùng không phải màu đỏ, thay bằng trắng ở vùng màu đỏ
        result = np.where(red_mask[:,:,np.newaxis].astype(bool), white_background, image)
        
        return result.astype(np.uint8)
    
    def enhance_text_contrast(self, image):
        """
        Tăng cường độ tương phản cho văn bản
        """
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # CLAHE để cải thiện độ tương phản cục bộ
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def advanced_denoise(self, image):
        """
        Giảm nhiễu nâng cao bảo toàn biên của chữ
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Non-local means denoising cho ảnh xám
        denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
        
        return denoised
    
    def sharpen_text(self, image):
        """
        Làm sắc nét văn bản đặc biệt
        """
        # Kernel làm sắc nét mạnh cho văn bản
        sharpen_kernel = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
        
        sharpened = cv2.filter2D(image, -1, sharpen_kernel)
        
        return sharpened
    
    def adaptive_binarization(self, image):
        """
        Nhị phân hóa thích nghi nâng cao cho văn bản
        """
        # Adaptive Gaussian thresholding
        binary = cv2.adaptiveThreshold(
            image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            15, 
            10
        )
        
        # Morphological operations để làm sạch kết quả
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def remove_background_noise(self, image):
        """
        Loại bỏ nhiễu nền và làm sạch tài liệu
        """
        # Làm mờ để giảm nhiễu
        blurred = cv2.GaussianBlur(image, (3,3), 0)
        
        # Phát hiện biên Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Tìm contours và lọc các contours nhỏ (nhiễu)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Tạo mask để giữ lại chỉ các vùng lớn (văn bản)
        mask = np.zeros_like(image)
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Chỉ giữ contours có diện tích lớn
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Áp dụng mask
        result = cv2.bitwise_and(image, image, mask=mask)
        
        return result
    
    def complete_enhancement_pipeline(self, image_path, output_path=None):
        """
        Pipeline xử lý hoàn chỉnh cho tài liệu có nền đỏ
        """
        print("Đang tải ảnh...")
        image = self.load_image(image_path)
        
        # Bước 1: Loại bỏ nền đỏ
        print("Đang loại bỏ nền đỏ...")
        no_red_bg = self.remove_red_background(image)
        
        # Bước 2: Tăng cường độ tương phản văn bản
        print("Đang tăng cường độ tương phản...")
        contrast_enhanced = self.enhance_text_contrast(no_red_bg)
        
        # Bước 3: Giảm nhiễu
        print("Đang giảm nhiễu...")
        denoised = self.advanced_denoise(contrast_enhanced)
        
        # Bước 4: Làm sắc nét văn bản
        print("Đang làm sắc nét văn bản...")
        sharpened = self.sharpen_text(denoised)
        
        # Bước 5: Loại bỏ nhiễu nền
        print("Đang loại bỏ nhiễu nền...")
        clean_bg = self.remove_background_noise(sharpened)
        
        # Bước 6: Nhị phân hóa
        print("Đang nhị phân hóa...")
        binary = self.adaptive_binarization(clean_bg)
        
        # Hiển thị kết quả
        plt.figure(figsize=(20, 12))
        
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Ảnh gốc')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(no_red_bg, cv2.COLOR_BGR2RGB))
        plt.title('Sau khi loại bỏ nền đỏ')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(contrast_enhanced, cmap='gray')
        plt.title('Sau tăng cường tương phản')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(denoised, cmap='gray')
        plt.title('Sau giảm nhiễu')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(sharpened, cmap='gray')
        plt.title('Sau làm sắc nét')
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.imshow(binary, cmap='gray')
        plt.title('Kết quả nhị phân cuối cùng')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Lưu kết quả
        if output_path:
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            # Lưu ảnh màu đã xử lý
            color_output = output_path.replace('.jpg', '_color_enhanced.jpg')
            cv2.imwrite(color_output, no_red_bg)
            
            # Lưu ảnh nhị phân
            binary_output = output_path.replace('.jpg', '_binary.jpg')
            cv2.imwrite(binary_output, binary)
            
            print(f"Đã lưu kết quả:")
            print(f"- Ảnh màu: {color_output}")
            print(f"- Ảnh nhị phân: {binary_output}")
        
        plt.show()
        
        return no_red_bg, binary

def main():
    parser = argparse.ArgumentParser(description='Xử lý nâng cao chất lượng ảnh tài liệu có nền đỏ')
    parser.add_argument('input', help='Đường dẫn đến file ảnh đầu vào')
    parser.add_argument('-o', '--output', help='Đường dẫn đến file đầu ra', default='enhanced_output.jpg')
    
    args = parser.parse_args()
    
    enhancer = AdvancedDocumentEnhancer()
    
    try:
        print("Bắt đầu xử lý ảnh tài liệu...")
        final_color, binary = enhancer.complete_enhancement_pipeline(args.input, args.output)
        print("Xử lý hoàn tất!")
        
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()