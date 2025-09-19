# pdf_to_jpg.py
import os
import fitz  # PyMuPDF
from PIL import Image
import io
import argparse
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFToJPGConverter:
    def __init__(self, dpi: int = 300, output_quality: int = 95):
        """
        Khởi tạo converter PDF to JPG
        
        Args:
            dpi (int): Độ phân giải đầu ra (dots per inch)
            output_quality (int): Chất lượng ảnh JPG (1-100)
        """
        self.dpi = dpi
        self.output_quality = output_quality
        self.supported_formats = ['.pdf']
    
    def is_pdf_file(self, file_path: str) -> bool:
        """Kiểm tra xem file có phải là PDF không"""
        return os.path.splitext(file_path.lower())[1] == '.pdf'
    
    def create_output_directory(self, pdf_path: str, output_dir: str = None) -> str:
        """
        Tạo thư mục đầu ra
        
        Args:
            pdf_path: Đường dẫn file PDF
            output_dir: Thư mục đầu ra (nếu None sẽ tạo tự động)
        
        Returns:
            str: Đường dẫn thư mục đầu ra
        """
        if output_dir is None:
            # Tạo thư mục cùng tên với file PDF
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_dir = os.path.join(os.path.dirname(pdf_path), f"{base_name}_images")
        
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def convert_pdf_to_images(self, pdf_path: str, output_dir: str = None) -> List[str]:
        """
        Chuyển đổi PDF thành các file JPG
        
        Args:
            pdf_path: Đường dẫn file PDF
            output_dir: Thư mục đầu ra (optional)
        
        Returns:
            List[str]: Danh sách đường dẫn các file JPG đã tạo
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File PDF không tồn tại: {pdf_path}")
        
        if not self.is_pdf_file(pdf_path):
            raise ValueError(f"File không phải định dạng PDF: {pdf_path}")
        
        # Tạo thư mục đầu ra
        output_dir = self.create_output_directory(pdf_path, output_dir)
        output_files = []
        
        try:
            # Mở file PDF
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            
            logger.info(f"Đang chuyển đổi PDF: {os.path.basename(pdf_path)}")
            logger.info(f"Số trang: {total_pages}")
            logger.info(f"Độ phân giải: {self.dpi} DPI")
            logger.info(f"Thư mục đầu ra: {output_dir}")
            
            for page_num in range(total_pages):
                # Lấy trang
                page = pdf_document.load_page(page_num)
                
                # Render trang thành ảnh với độ phân giải cao
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)  # 72 DPI là DPI mặc định của PDF
                pix = page.get_pixmap(matrix=mat)
                
                # Chuyển đổi thành PIL Image
                img_data = pix.tobytes("ppm")
                pil_image = Image.open(io.BytesIO(img_data))
                
                # Chuyển đổi sang RGB (đảm bảo định dạng phù hợp cho JPG)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Tạo tên file đầu ra
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                output_filename = f"{base_name}_page_{page_num + 1:03d}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                # Lưu ảnh với chất lượng cao
                pil_image.save(output_path, 'JPEG', quality=self.output_quality, optimize=True)
                output_files.append(output_path)
                
                logger.info(f"Đã chuyển đổi trang {page_num + 1}/{total_pages} -> {output_filename}")
            
            pdf_document.close()
            logger.info(f"Hoàn thành! Đã tạo {len(output_files)} file JPG.")
            
        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi PDF: {e}")
            raise
        
        return output_files
    
    def convert_multiple_pdfs(self, pdf_paths: List[str], output_dirs: List[str] = None) -> dict:
        """
        Chuyển đổi nhiều file PDF cùng lúc
        
        Args:
            pdf_paths: Danh sách đường dẫn file PDF
            output_dirs: Danh sách thư mục đầu ra tương ứng (optional)
        
        Returns:
            dict: Dictionary chứa kết quả chuyển đổi
        """
        results = {}
        
        for i, pdf_path in enumerate(pdf_paths):
            try:
                output_dir = output_dirs[i] if output_dirs and i < len(output_dirs) else None
                jpg_files = self.convert_pdf_to_images(pdf_path, output_dir)
                results[pdf_path] = {
                    'success': True,
                    'output_files': jpg_files,
                    'output_dir': os.path.dirname(jpg_files[0]) if jpg_files else None
                }
            except Exception as e:
                results[pdf_path] = {
                    'success': False,
                    'error': str(e),
                    'output_files': []
                }
        
        return results

def main():
    """Hàm main để chạy từ command line"""
    parser = argparse.ArgumentParser(description='Chuyển đổi PDF thành JPG')
    parser.add_argument('input', help='Đường dẫn file PDF hoặc thư mục chứa PDF')
    parser.add_argument('-o', '--output', help='Thư mục đầu ra (optional)')
    parser.add_argument('-d', '--dpi', type=int, default=300, help='Độ phân giải đầu ra (default: 300)')
    parser.add_argument('-q', '--quality', type=int, default=95, help='Chất lượng JPG (1-100, default: 95)')
    parser.add_argument('-r', '--recursive', action='store_true', help='Quét đệ quy thư mục con')
    
    args = parser.parse_args()
    
    converter = PDFToJPGConverter(dpi=args.dpi, output_quality=args.quality)
    pdf_files = []
    
    # Xử lý đầu vào
    if os.path.isfile(args.input):
        if converter.is_pdf_file(args.input):
            pdf_files = [args.input]
        else:
            logger.error("File đầu vào không phải PDF")
            return
    
    elif os.path.isdir(args.input):
        # Tìm tất cả file PDF trong thư mục
        for root, dirs, files in os.walk(args.input if args.recursive else args.input):
            for file in files:
                if converter.is_pdf_file(file):
                    pdf_files.append(os.path.join(root, file))
        
        if not pdf_files:
            logger.warning("Không tìm thấy file PDF nào trong thư mục")
            return
    
    else:
        logger.error("Đường dẫn không tồn tại")
        return
    
    logger.info(f"Tìm thấy {len(pdf_files)} file PDF để chuyển đổi")
    
    # Chuyển đổi
    results = converter.convert_multiple_pdfs(pdf_files)
    
    # Hiển thị kết quả
    successful = sum(1 for result in results.values() if result['success'])
    logger.info(f"Hoàn thành! Thành công: {successful}/{len(pdf_files)}")

if __name__ == "__main__":
    print("Bắt đầu chuyển đổi PDF sang JPG...")
    main()