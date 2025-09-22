# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng xử lý ảnh")
        self.root.geometry("1200x800")

        self.original_image = None
        self.processed_image = None
        self.file_path = None
        self.tk_image_original = None  # Giữ tham chiếu ảnh gốc để tránh bị thu hồi
        self.tk_image_processed = None # Giữ tham chiếu ảnh đã xử lý

        # Chia layout chính thành hai khung: khung điều khiển và khung hiển thị ảnh
        self.control_frame = tk.Frame(root, width=300, bg="#e0e0e0", padx=10, pady=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.image_frame = tk.Frame(root, bg="#f0f0f0")
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tạo widget cho khung điều khiển
        self.create_controls()

        # Tạo hệ thống tab
        self.notebook = ttk.Notebook(self.image_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.original_tab = tk.Frame(self.notebook)
        self.processed_tab = tk.Frame(self.notebook)

        self.notebook.add(self.original_tab, text="Ảnh Gốc")
        self.notebook.add(self.processed_tab, text="Ảnh đã xử lý")

        # Tạo canvas để hiển thị ảnh gốc và ảnh đã xử lý
        self.canvas_original = tk.Canvas(self.original_tab, bg="#ffffff")
        self.canvas_original.pack(fill=tk.BOTH, expand=True)
        self.canvas_original.bind("<Configure>", self.on_canvas_resize)

        self.canvas_processed = tk.Canvas(self.processed_tab, bg="#ffffff")
        self.canvas_processed.pack(fill=tk.BOTH, expand=True)
        self.canvas_processed.bind("<Configure>", self.on_canvas_resize)

    def create_controls(self):
        """Tạo các nút và thanh trượt trong khung điều khiển."""
        tk.Label(self.control_frame, text="Điều khiển xử lý ảnh", font=("Arial", 14, "bold"), bg="#e0e0e0").pack(pady=(0, 10))

        # Hiển thị tên file ảnh
        self.file_name_label = tk.Label(self.control_frame, text="Chưa có ảnh nào được chọn", bg="#e0e0e0", wraplength=280)
        self.file_name_label.pack(pady=(0, 10), fill=tk.X)

        # Nút Tải ảnh
        load_button = tk.Button(self.control_frame, text="Tải ảnh", command=self.load_image, font=("Arial", 12))
        load_button.pack(pady=5, fill=tk.X)
        
        # Nút Lưu ảnh
        save_button = tk.Button(self.control_frame, text="Lưu ảnh", command=self.save_image, font=("Arial", 12))
        save_button.pack(pady=5, fill=tk.X)

        # --- CÁC THANH TRƯỢT ĐIỀU CHỈNH THÔNG SỐ ---
        
        # Bước 1: CLAHE (Tăng cường độ tương phản)
        self.clahe_label = tk.Label(self.control_frame, text="1. Tăng cường tương phản (CLAHE)", font=("Arial", 12), bg="#e0e0e0")
        self.clahe_label.pack(anchor="w", pady=(10, 0))
        
        tk.Label(self.control_frame, text="Clip Limit:", bg="#e0e0e0").pack(anchor="w")
        self.clahe_clip_limit = tk.Scale(self.control_frame, from_=1.0, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_image)
        self.clahe_clip_limit.set(3.0)
        self.clahe_clip_limit.pack(fill=tk.X)
        
        tk.Label(self.control_frame, text="Kích thước ô lưới (Grid Size):", bg="#e0e0e0").pack(anchor="w")
        self.clahe_grid_size = tk.Scale(self.control_frame, from_=1, to=200, resolution=1, orient=tk.HORIZONTAL, command=self.update_image)
        self.clahe_grid_size.set(100)
        self.clahe_grid_size.pack(fill=tk.X)

        # Bước 2: Sharpening (Làm sắc nét)
        tk.Label(self.control_frame, text="2. Làm sắc nét chữ viết", font=("Arial", 12), bg="#e0e0e0").pack(anchor="w", pady=(10, 0))
        
        tk.Label(self.control_frame, text="Độ sắc nét:", bg="#e0e0e0").pack(anchor="w")
        self.sharpen_amount = tk.Scale(self.control_frame, from_=0.0, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_image)
        self.sharpen_amount.set(0.5)
        self.sharpen_amount.pack(fill=tk.X)

        # Bước 3: Smoothing (Làm mịn)
        tk.Label(self.control_frame, text="3. Làm mịn ảnh", font=("Arial", 12), bg="#e0e0e0").pack(anchor="w", pady=(10, 0))
        
        self.smooth_type = tk.StringVar(value="Gaussian")
        tk.Radiobutton(self.control_frame, text="Gaussian Blur", variable=self.smooth_type, value="Gaussian", command=self.update_smooth_controls, bg="#e0e0e0").pack(anchor="w")
        tk.Radiobutton(self.control_frame, text="Bilateral Filter", variable=self.smooth_type, value="Bilateral", command=self.update_smooth_controls, bg="#e0e0e0").pack(anchor="w")
        
        # Gaussian Blur Controls
        self.gaussian_frame = tk.Frame(self.control_frame, bg="#e0e0e0")
        self.gaussian_frame.pack(fill=tk.X)
        tk.Label(self.gaussian_frame, text="Kích thước kernel:", bg="#e0e0e0").pack(anchor="w")
        self.gaussian_size = tk.Scale(self.gaussian_frame, from_=1, to=31, resolution=2, orient=tk.HORIZONTAL, command=self.update_image)
        self.gaussian_size.set(3)
        self.gaussian_size.pack(fill=tk.X)
        
        # Bilateral Filter Controls
        self.bilateral_frame = tk.Frame(self.control_frame, bg="#e0e0e0")
        
        tk.Label(self.bilateral_frame, text="Kích thước kernel (d):", bg="#e0e0e0").pack(anchor="w")
        self.bilateral_d = tk.Scale(self.bilateral_frame, from_=1, to=20, resolution=1, orient=tk.HORIZONTAL, command=self.update_image)
        self.bilateral_d.set(9)
        self.bilateral_d.pack(fill=tk.X)
        
        tk.Label(self.bilateral_frame, text="Sigma Color:", bg="#e0e0e0").pack(anchor="w")
        self.bilateral_sigma_color = tk.Scale(self.bilateral_frame, from_=1, to=200, resolution=1, orient=tk.HORIZONTAL, command=self.update_image)
        self.bilateral_sigma_color.set(75)
        self.bilateral_sigma_color.pack(fill=tk.X)
        
        tk.Label(self.bilateral_frame, text="Sigma Space:", bg="#e0e0e0").pack(anchor="w")
        self.bilateral_sigma_space = tk.Scale(self.bilateral_frame, from_=1, to=200, resolution=1, orient=tk.HORIZONTAL, command=self.update_image)
        self.bilateral_sigma_space.set(75)
        self.bilateral_sigma_space.pack(fill=tk.X)
        
        self.update_smooth_controls()

    def update_smooth_controls(self):
        """Ẩn/hiện các thanh trượt làm mịn tùy thuộc vào lựa chọn của người dùng."""
        if self.smooth_type.get() == "Gaussian":
            self.gaussian_frame.pack(fill=tk.X)
            self.bilateral_frame.pack_forget()
        else:
            self.gaussian_frame.pack_forget()
            self.bilateral_frame.pack(fill=tk.X)
        
        self.update_image()

    def load_image(self):
        """Tải ảnh từ hệ thống tệp và hiển thị."""
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if not filepath:
            return

        # Lưu đường dẫn và tên file
        self.file_path = filepath
        self.file_name_label.config(text=f"Tên file: {os.path.basename(filepath)}")

        # Đọc ảnh bằng OpenCV và chuyển đổi sang không gian màu RGB
        self.original_image = cv2.imread(filepath)
        if self.original_image is None:
            tk.messagebox.showerror("Lỗi", "Không thể tải ảnh. Vui lòng thử lại.")
            return

        # Chuyển đổi BGR sang RGB để hiển thị đúng
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Hiển thị ảnh gốc
        self.display_image(self.original_image, self.canvas_original, "original")
        
        # Bắt đầu xử lý ảnh
        self.update_image()

    def save_image(self):
        """Lưu ảnh đã xử lý vào hệ thống tệp."""
        if self.processed_image is None:
            tk.messagebox.showwarning("Cảnh báo", "Không có ảnh nào đã xử lý để lưu.")
            return

        if self.file_path:
            # Tạo tên file mới
            directory, file_name = os.path.split(self.file_path)
            base_name, ext = os.path.splitext(file_name)
            new_file_name = f"processed_{base_name}{ext}"
            new_file_path = os.path.join(directory, new_file_name)
            
            # Chuyển ảnh từ RGB sang BGR để lưu bằng OpenCV
            processed_bgr = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(new_file_path, processed_bgr)
            tk.messagebox.showinfo("Thông báo", f"Ảnh đã được lưu thành công tại:\n{new_file_path}")
        else:
            tk.messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh lên trước khi lưu.")

    def update_image(self, event=None):
        """
        Áp dụng toàn bộ pipeline xử lý ảnh và cập nhật hình ảnh hiển thị.
        Hàm này được gọi mỗi khi có thanh trượt nào đó thay đổi.
        """
        if self.original_image is None:
            return

        current_image = self.original_image.copy()
        
        # Bước 1: CLAHE (Tăng cường độ tương phản)
        lab_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab_image)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit.get(),
            tileGridSize=(self.clahe_grid_size.get(), self.clahe_grid_size.get())
        )
        l_clahe = clahe.apply(l)
        limg = cv2.merge((l_clahe, a, b))
        current_image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        # Chuyển đổi sang ảnh xám để làm sắc nét
        gray_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)

        # Bước 2: Sharpening (Làm sắc nét chữ viết)
        gaussian_blur = cv2.GaussianBlur(gray_image, (0, 0), 3.0)
        sharpened_image = cv2.addWeighted(gray_image, 2, gaussian_blur, -self.sharpen_amount.get(),  0)

        # Bước 3: Smoothing (Làm mịn ảnh)
        if self.smooth_type.get() == "Gaussian":
            k_size = int(self.gaussian_size.get())
            if k_size % 2 == 0: k_size += 1
            if k_size > 1:
                denoised_image = cv2.GaussianBlur(sharpened_image, (k_size, k_size), 0)
            else:
                denoised_image = sharpened_image
        else: # Bilateral Filter
            d = int(self.bilateral_d.get())
            sigma_color = int(self.bilateral_sigma_color.get())
            sigma_space = int(self.bilateral_sigma_space.get())
            # Bilateral filter hoạt động tốt hơn trên ảnh màu
            sharpened_color = cv2.cvtColor(sharpened_image, cv2.COLOR_GRAY2RGB)
            denoised_image = cv2.bilateralFilter(sharpened_color, d, sigma_color, sigma_space)
            
        # Cập nhật ảnh đã xử lý
        if len(denoised_image.shape) == 2:
             denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_GRAY2RGB)
             
        self.processed_image = denoised_image
        self.display_image(self.processed_image, self.canvas_processed, "processed")
    
    def on_canvas_resize(self, event):
        """Hàm này được gọi khi kích thước canvas thay đổi."""
        if self.original_image is not None:
            self.display_image(self.original_image, self.canvas_original, "original")
        if self.processed_image is not None:
            self.display_image(self.processed_image, self.canvas_processed, "processed")

    def display_image(self, img_data, canvas, img_type):
        """Hiển thị ảnh trên canvas, tự động điều chỉnh kích thước."""
        if img_data is None:
            return

        canvas.delete("all")
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        h, w = img_data.shape[:2]
        ratio = min(canvas_width / w, canvas_height / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)

        pil_image = Image.fromarray(img_data)
        pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
        
        # Cập nhật tham chiếu ảnh
        if img_type == "original":
            self.tk_image_original = ImageTk.PhotoImage(pil_image)
            canvas.create_image(
                canvas_width / 2,
                canvas_height / 2,
                image=self.tk_image_original,
                anchor=tk.CENTER
            )
        else:
            self.tk_image_processed = ImageTk.PhotoImage(pil_image)
            canvas.create_image(
                canvas_width / 2,
                canvas_height / 2,
                image=self.tk_image_processed,
                anchor=tk.CENTER
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
