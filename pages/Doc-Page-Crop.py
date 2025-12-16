from kukalib.doc_page_crop import *

import streamlit as st
import cv2
import numpy as np
import datetime
import tempfile
import os

def main_loop():
    versionInfo = getVersionInfo()

    st.set_page_config(page_title="Document Page Cropping Tool", layout="wide")
    st.title("Demo Tool: Document Page Detection & Cropping")
    st.text("Version: " + versionInfo["version"] + " - Release date:" + versionInfo["date"].strftime("%Y-%m-%d"))

    st.subheader("Phát hiện và Crop vùng tài liệu từ ảnh")
    
    # Add description
    st.markdown("""
    **Tính năng:**
    - Tự động phát hiện vùng tài liệu trong ảnh
    - Loại bỏ background thừa
    - Chỉnh góc nghiêng và perspective
    - Hỗ trợ cả file PDF (tự động xử lý từng trang)
    
    **Phương pháp:**
    - **Auto**: Tự động chọn phương pháp tốt nhất (ưu tiên U2-Net)
    - **U2-Net**: AI-based detection (KHUYẾN NGHỊ - chính xác nhất, model 4.4MB)
    - **Content**: Content-based detection (nhanh, tốt cho tài liệu có margin rõ)
    - **DeepLabV3**: Semantic segmentation CNN (cần cài thêm torch)
    - **EdgeLinking**: Edge-based detection (tốt cho document có edge rõ)
    - **Morphology**: Advanced morphological operations (tốt cho high-contrast)
    - **OpenCV**: Traditional method (nhanh nhất, background đơn giản)
    
    **Lưu ý**: U2-Net sẽ tự động tải model (4.4MB) vào thư mục kukalib/models/ lần đầu sử dụng.
    """)
    
    # Method selection and debug option
    col_method, col_debug, col_empty = st.columns([1, 1, 1])
    with col_method:
        detection_method = st.selectbox(
            "Chọn phương pháp phát hiện:",
            ["auto", "u2net", "content", "deeplabv3", "edgelinking", "morphology", "opencv"],
            index=0,
            help="Auto: Ưu tiên U2-Net (chính xác nhất). U2-Net: AI detection với model ONNX 4.4MB."
        )
    with col_debug:
        show_debug = st.checkbox(
            "Debug mode",
            value=False,
            help="Hiển thị thông tin debug chi tiết trong console"
        )
    
    # Show example images
    st.subheader("📸 Ví dụ kết quả")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1:
        if os.path.exists('docs/samples/input/sample1_simple_document.jpg'):
            st.image('docs/samples/input/sample1_simple_document.jpg', caption='Ảnh gốc', use_column_width=True)
    with col_ex2:
        if os.path.exists('docs/samples/output/sample1_simple_document_debug.jpg'):
            st.image('docs/samples/output/sample1_simple_document_debug.jpg', caption='Phát hiện', use_column_width=True)
    with col_ex3:
        if os.path.exists('docs/samples/output/sample1_simple_document_cropped.jpg'):
            st.image('docs/samples/output/sample1_simple_document_cropped.jpg', caption='Kết quả', use_column_width=True)
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Upload ảnh hoặc PDF",
        type=['jpg', 'png', 'jpeg', 'tif', 'tiff', 'pdf'],
        help="Hỗ trợ các định dạng: JPG, PNG, TIFF, PDF"
    )
    
    if not uploaded_file:
        st.info("👆 Vui lòng upload file để bắt đầu xử lý")
        return None
    
    # Check if PDF or Image
    file_type = uploaded_file.type
    
    if 'pdf' in file_type:
        # Handle PDF
        st.subheader("📄 Xử lý file PDF")
        
        # Save PDF to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_pdf_path = tmp_file.name
        
        try:
            # Convert PDF to images
            with st.spinner('Đang convert PDF...'):
                images = convertPdfToImages(tmp_pdf_path)
            
            if not images:
                st.error("❌ Không thể đọc file PDF. Vui lòng kiểm tra lại file.")
                return None
            
            st.success(f"✅ Đã tải {len(images)} trang từ PDF")
            
            # Process each page
            for page_num, img in enumerate(images):
                st.markdown(f"### Trang {page_num + 1}/{len(images)}")
                
                with st.spinner(f'Đang xử lý trang {page_num + 1}...'):
                    cropped, debug, corners, method_used = detectAndCropDocumentPage(
                        img, 
                        method=detection_method, 
                        debug=show_debug
                    )
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Ảnh gốc**")
                    st.image(img, channels='BGR', use_column_width=True)
                    # Download button for original image
                    is_success_orig, buffer_orig = cv2.imencode(".jpg", img)
                    if is_success_orig:
                        st.download_button(
                            label=f"⬇️ Tải ảnh gốc",
                            data=buffer_orig.tobytes(),
                            file_name=f"page_{page_num + 1}_original.jpg",
                            mime="image/jpeg",
                            key=f"dl_orig_{page_num}"
                        )
                
                with col2:
                    st.markdown(f"**Phát hiện** ({method_used})")
                    st.image(debug, channels='BGR', use_column_width=True)
                
                with col3:
                    st.markdown("**Kết quả**")
                    st.image(cropped, channels='BGR', use_column_width=True)
                    
                    # Download button for cropped image
                    is_success, buffer = cv2.imencode(".jpg", cropped)
                    if is_success:
                        st.download_button(
                            label=f"⬇️ Tải kết quả",
                            data=buffer.tobytes(),
                            file_name=f"page_{page_num + 1}_cropped.jpg",
                            mime="image/jpeg",
                            key=f"dl_crop_{page_num}"
                        )
                
                if page_num < len(images) - 1:
                    st.markdown("---")
        
        finally:
            # Clean up temp file
            if os.path.exists(tmp_pdf_path):
                os.unlink(tmp_pdf_path)
    
    else:
        # Handle Image
        st.subheader("🖼️ Xử lý ảnh")
        
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        src = cv2.imdecode(file_bytes, 1)
        
        if src is None:
            st.error("❌ Không thể đọc file ảnh. Vui lòng kiểm tra lại file.")
            return None
        
        # Process image
        with st.spinner('Đang phát hiện và crop tài liệu...'):
            cropped, debug, corners, method_used = detectAndCropDocumentPage(
                src, 
                method=detection_method, 
                debug=show_debug
            )
        
        # Show results
        st.markdown("### Kết quả xử lý")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Ảnh gốc**")
            st.image(src, channels='BGR', use_column_width=True)
            st.caption(f"Kích thước: {src.shape[1]}x{src.shape[0]}")
        
        with col2:
            st.markdown(f"**Phát hiện**")
            st.image(debug, channels='BGR', use_column_width=True)
            st.caption(f"Phương pháp: {method_used}")
        
        with col3:
            st.markdown("**Kết quả Crop**")
            st.image(cropped, channels='BGR', use_column_width=True)
            st.caption(f"Kích thước: {cropped.shape[1]}x{cropped.shape[0]}")
        
        # Download button
        st.markdown("---")
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl2:
            is_success, buffer = cv2.imencode(".jpg", cropped)
            if is_success:
                st.download_button(
                    label="⬇️ Tải về ảnh đã crop",
                    data=buffer.tobytes(),
                    file_name="document_cropped.jpg",
                    mime="image/jpeg"
                )
        
        with col_dl3:
            is_success, buffer = cv2.imencode(".jpg", debug)
            if is_success:
                st.download_button(
                    label="⬇️ Tải về ảnh debug",
                    data=buffer.tobytes(),
                    file_name="document_debug.jpg",
                    mime="image/jpeg"
                )
    
    # Add footer with info
    st.markdown("---")
    st.markdown("""
    **💡 Tips:**
    - Ảnh rõ nét, tài liệu không bị che khuất sẽ cho kết quả tốt nhất
    - Nền đơn giản, tương phản cao giúp phát hiện chính xác hơn
    - Với background phức tạp, hãy thử phương pháp U2-Net (cần cài đặt `rembg`)
    - PDF sẽ được tự động chuyển đổi và xử lý từng trang
    
    **📚 Tài liệu kỹ thuật:** [docs/doc_page_detection_solution.md](docs/doc_page_detection_solution.md)
    """)

if __name__ == '__main__':
    main_loop()
