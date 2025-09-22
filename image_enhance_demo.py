import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

# --- Các hàm tự động hóa ---
def auto_clahe_params(image):
    """
    Tự động xác định tham số cho CLAHE.
    """
    # Chuyển đổi sang không gian màu LAB và trích xuất kênh L
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Tính toán độ lệch chuẩn của kênh L để đánh giá độ tương phản
    std_dev = np.std(l_channel)

    # Dựa trên độ lệch chuẩn, xác định clipLimit
    if std_dev < 50:
        clip_limit = 3.0
    elif std_dev < 75:
        clip_limit = 2.0
    else:
        clip_limit = 1.0
    
    # Dựa trên mật độ cạnh, xác định tileGridSize
    edges = cv2.Canny(l_channel, 100, 200)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    
    if edge_density > 0.1:
        tile_size = 8
    else:
        tile_size = 16
        
    return clip_limit, tile_size

def auto_sharpen_params(gray_image):
    """
    Tự động xác định tham số làm sắc nét dựa trên phân tích độ mờ của ảnh.
    """
    # Tính toán Laplacian của ảnh
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    
    # Tính phương sai của Laplacian để đánh giá độ mờ
    laplacian_variance = laplacian.var()
    
    # Đặt ngưỡng để xác định mức độ mờ
    blur_threshold = 100 
    
    # Xác định alpha và beta dựa trên phương sai
    if laplacian_variance < blur_threshold:
        alpha = 2.0
    else:
        alpha = 1.2
        
    beta = 1.0 - alpha
    
    return alpha, beta

def auto_bilateral_params(gray_image):
    """
    Tự động xác định tham số cho Bilateral Filter dựa trên phân tích ảnh.
    """
    # Tính toán Laplacian của ảnh để đánh giá độ nhiễu
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    
    # Phương sai của Laplacian là thước đo mức độ nhiễu và chi tiết
    laplacian_variance = laplacian.var()
    
    # Đặt ngưỡng để phân loại ảnh
    noise_threshold = 100 
    d = 9

    if laplacian_variance > noise_threshold:
        sigma_color = 100
        sigma_space = 100
    else:
        sigma_color = 50
        sigma_space = 50
        
    return d, sigma_color, sigma_space

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ứng dụng xử lý ảnh")
        self.geometry("1400x800")
        
        self.image = None
        self.processed_image = None
        self.image_path = None
        
        # Tạo khung chính
        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Khung điều khiển bên trái
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Điều khiển", padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Khung hiển thị ảnh
        self.display_frame = ttk.Frame(self.main_frame)
        self.display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Khung hiển thị Histogram
        self.histogram_frame = ttk.LabelFrame(self.main_frame, text="Biểu đồ Histogram", padding="10")
        self.histogram_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.file_name_label = ttk.Label(self.control_frame, text="Tên file: Chưa chọn")
        self.file_name_label.pack(pady=5)
        
        # Tạo các nút điều khiển
        self.open_button = ttk.Button(self.control_frame, text="Chọn ảnh", command=self.open_image)
        self.open_button.pack(pady=5)
        
        self.save_button = ttk.Button(self.control_frame, text="Lưu ảnh đã xử lý", command=self.save_image)
        self.save_button.pack(pady=5)
        
        # Tạo các tab để hiển thị ảnh
        self.tab_control = ttk.Notebook(self.display_frame)
        self.tab_control.pack(fill=tk.BOTH, expand=True)

        self.original_tab = ttk.Frame(self.tab_control)
        self.processed_tab = ttk.Frame(self.tab_control)

        self.tab_control.add(self.original_tab, text='Ảnh gốc')
        self.tab_control.add(self.processed_tab, text='Ảnh đã xử lý')

        self.original_canvas = tk.Canvas(self.original_tab, bg="white")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        self.processed_canvas = tk.Canvas(self.processed_tab, bg="white")
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)
        
        # --- Các thanh trượt cho CLAHE ---
        clahe_frame = ttk.LabelFrame(self.control_frame, text="1. Tăng tương phản (CLAHE)", padding="10")
        clahe_frame.pack(pady=10, fill=tk.X)

        ttk.Label(clahe_frame, text="Clip Limit").pack(pady=5)
        self.clahe_clip_limit_slider = ttk.Scale(clahe_frame, from_=0.0, to=10.0, orient=tk.HORIZONTAL, command=self.apply_filters)
        self.clahe_clip_limit_slider.set(2.0)
        self.clahe_clip_limit_slider.pack(fill=tk.X)
        self.clahe_clip_limit_value_label = ttk.Label(clahe_frame, text="Giá trị: 2.0")
        self.clahe_clip_limit_value_label.pack(pady=2)
        
        ttk.Label(clahe_frame, text="Tile Grid Size").pack(pady=5)
        self.clahe_tile_grid_size_slider = ttk.Scale(clahe_frame, from_=2, to=32, orient=tk.HORIZONTAL, command=self.apply_filters)
        self.clahe_tile_grid_size_slider.set(8)
        self.clahe_tile_grid_size_slider.pack(fill=tk.X)
        self.clahe_tile_grid_size_value_label = ttk.Label(clahe_frame, text="Giá trị: 8")
        self.clahe_tile_grid_size_value_label.pack(pady=2)
        
        # --- Các thanh trượt cho làm sắc nét ---
        sharpen_frame = ttk.LabelFrame(self.control_frame, text="2. Làm sắc nét chữ", padding="10")
        sharpen_frame.pack(pady=10, fill=tk.X)
        
        ttk.Label(sharpen_frame, text="Trọng số Alpha").pack(pady=5)
        self.sharpen_alpha_slider = ttk.Scale(sharpen_frame, from_=0.0, to=5.0, orient=tk.HORIZONTAL, command=self.apply_filters)
        self.sharpen_alpha_slider.set(1.5)
        self.sharpen_alpha_slider.pack(fill=tk.X)
        self.sharpen_alpha_value_label = ttk.Label(sharpen_frame, text="Giá trị: 1.5")
        self.sharpen_alpha_value_label.pack(pady=2)

        ttk.Label(sharpen_frame, text="Trọng số Beta").pack(pady=5)
        self.sharpen_beta_slider = ttk.Scale(sharpen_frame, from_=-5.0, to=0.0, orient=tk.HORIZONTAL, command=self.apply_filters)
        self.sharpen_beta_slider.set(-0.5)
        self.sharpen_beta_slider.pack(fill=tk.X)
        self.sharpen_beta_value_label = ttk.Label(sharpen_frame, text="Giá trị: -0.5")
        self.sharpen_beta_value_label.pack(pady=2)
        
        # --- Các thanh trượt cho làm mịn ---
        smooth_frame = ttk.LabelFrame(self.control_frame, text="3. Làm mịn (Bilateral Filter)", padding="10")
        smooth_frame.pack(pady=10, fill=tk.X)
        
        ttk.Label(smooth_frame, text="Sigma Color").pack(pady=5)
        self.bilateral_sigma_color_slider = ttk.Scale(smooth_frame, from_=10, to=200, orient=tk.HORIZONTAL, command=self.apply_filters)
        self.bilateral_sigma_color_slider.set(75)
        self.bilateral_sigma_color_slider.pack(fill=tk.X)
        self.bilateral_sigma_color_value_label = ttk.Label(smooth_frame, text="Giá trị: 75")
        self.bilateral_sigma_color_value_label.pack(pady=2)
        
        ttk.Label(smooth_frame, text="Sigma Space").pack(pady=5)
        self.bilateral_sigma_space_slider = ttk.Scale(smooth_frame, from_=10, to=200, orient=tk.HORIZONTAL, command=self.apply_filters)
        self.bilateral_sigma_space_slider.set(75)
        self.bilateral_sigma_space_slider.pack(fill=tk.X)
        self.bilateral_sigma_space_value_label = ttk.Label(smooth_frame, text="Giá trị: 75")
        self.bilateral_sigma_space_value_label.pack(pady=2)

        # Canvas cho Histogram gốc
        ttk.Label(self.histogram_frame, text="Ảnh gốc").pack(pady=5)
        self.original_histogram_canvas = tk.Canvas(self.histogram_frame, bg="white", width=256, height=150)
        self.original_histogram_canvas.pack(fill=tk.X)

        # Canvas cho Histogram đã xử lý
        ttk.Label(self.histogram_frame, text="Ảnh đã xử lý").pack(pady=5)
        self.processed_histogram_canvas = tk.Canvas(self.histogram_frame, bg="white", width=256, height=150)
        self.processed_histogram_canvas.pack(fill=tk.X)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.image = cv2.imread(self.image_path)
            if self.image is not None:
                self.file_name_label.config(text=f"Tên file: {os.path.basename(self.image_path)}")
                
                # Tính toán các tham số tự động
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                clip_limit, tile_size = auto_clahe_params(self.image)
                alpha, beta = auto_sharpen_params(gray_image)
                d, sigma_c, sigma_s = auto_bilateral_params(gray_image)

                # Cập nhật giá trị lên các thanh trượt và nhãn
                self.clahe_clip_limit_slider.set(clip_limit)
                self.clahe_clip_limit_value_label.config(text=f"Giá trị: {clip_limit:.2f}")

                self.clahe_tile_grid_size_slider.set(tile_size)
                self.clahe_tile_grid_size_value_label.config(text=f"Giá trị: {tile_size}")

                self.sharpen_alpha_slider.set(alpha)
                self.sharpen_alpha_value_label.config(text=f"Giá trị: {alpha:.2f}")

                self.sharpen_beta_slider.set(beta)
                self.sharpen_beta_value_label.config(text=f"Giá trị: {beta:.2f}")

                self.bilateral_sigma_color_slider.set(sigma_c)
                self.bilateral_sigma_color_value_label.config(text=f"Giá trị: {sigma_c}")

                self.bilateral_sigma_space_slider.set(sigma_s)
                self.bilateral_sigma_space_value_label.config(text=f"Giá trị: {sigma_s}")

                # Hiển thị ảnh và histogram gốc
                self.display_image(self.image, self.original_canvas)
                self.display_histogram(self.image, self.original_histogram_canvas)
                self.apply_filters()
            else:
                print("Lỗi: Không thể đọc file ảnh.")
    
    def apply_filters(self, event=None):
        if self.image is None:
            return

        # Lấy giá trị từ thanh trượt
        clip_limit = self.clahe_clip_limit_slider.get()
        tile_size = int(self.clahe_tile_grid_size_slider.get())
        alpha = self.sharpen_alpha_slider.get()
        beta = self.sharpen_beta_slider.get()
        sigma_color = int(self.bilateral_sigma_color_slider.get())
        sigma_space = int(self.bilateral_sigma_space_slider.get())
        d = 9 # Kích thước kernel d được giữ cố định

        # Cập nhật giá trị trên nhãn khi thanh trượt thay đổi
        self.clahe_clip_limit_value_label.config(text=f"Giá trị: {clip_limit:.2f}")
        self.clahe_tile_grid_size_value_label.config(text=f"Giá trị: {tile_size}")
        self.sharpen_alpha_value_label.config(text=f"Giá trị: {alpha:.2f}")
        self.sharpen_beta_value_label.config(text=f"Giá trị: {beta:.2f}")
        self.bilateral_sigma_color_value_label.config(text=f"Giá trị: {sigma_color}")
        self.bilateral_sigma_space_value_label.config(text=f"Giá trị: {sigma_space}")

        # Chuyển đổi sang không gian màu LAB để xử lý ánh sáng và màu sắc riêng biệt
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Áp dụng CLAHE cho kênh L
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))

        # Chuyển đổi ngược lại không gian màu
        enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Chuyển đổi không gian màu sang thang độ xám
        gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        
        # Áp dụng làm sắc nét (Unsharp Masking)
        gaussian = cv2.GaussianBlur(gray_image, (0, 0), 3.0)
        sharpened_image = cv2.addWeighted(gray_image, alpha, gaussian, beta, 0)
        
        # Áp dụng bộ lọc làm mịn
        denoised_image = cv2.bilateralFilter(sharpened_image, d, sigma_color, sigma_space)
        
        self.processed_image = denoised_image
        
        # Hiển thị ảnh và histogram đã xử lý
        self.display_image(self.processed_image, self.processed_canvas)
        self.display_histogram(self.processed_image, self.processed_histogram_canvas)

    def display_image(self, image, canvas):
        # Lấy kích thước canvas và đảm bảo chúng hợp lệ
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 20 or canvas_height <= 20:
            self.after(100, lambda: self.display_image(image, canvas))
            return
            
        # Chuyển đổi ảnh OpenCV sang định dạng Tkinter
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Giảm kích thước ảnh để hiển thị vừa khung canvas
        max_width = canvas_width - 20
        max_height = canvas_height - 20
        pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Cập nhật canvas
        canvas.delete("all")
        canvas.config(width=pil_image.width, height=pil_image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.image = tk_image
        
    def display_histogram(self, image, canvas):
        # Biểu đồ Histogram
        # Đảm bảo ảnh là ảnh grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 0 or canvas_height <= 0:
            return

        cv2.normalize(hist, hist, 0, canvas_height, cv2.NORM_MINMAX)
        
        canvas.delete("all")
        canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="white", outline="gray")
        
        bin_width = canvas_width / 256
        
        for i in range(256):
            canvas.create_line(
                i,
                canvas_height,
                i,
                canvas_height - hist[i][0],
                fill="black"
            )

    def save_image(self):
        if self.processed_image is not None and self.image_path:
            original_dir, original_file = os.path.split(self.image_path)
            file_name, file_ext = os.path.splitext(original_file)
            new_file_name = f"processed_{file_name}{file_ext}"
            save_path = os.path.join(original_dir, new_file_name)
            
            # Lưu ảnh đã xử lý
            cv2.imwrite(save_path, self.processed_image)
            msg =f"Ảnh đã được lưu tại: {save_path}"
            print(msg)
            tk.messagebox.showinfo("Lưu ảnh", msg)

if __name__ == "__main__":
    app = App()
    app.mainloop()
