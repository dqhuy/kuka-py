from cardcrop.cardcrop import *

import streamlit as st

import cv2
import streamlit as st
import numpy as np

def main_loop():
    st.set_page_config(page_title="Card Croping Tool")
    st.title("Demo Tool: Card Croping")
    st.subheader("Crop card từ ảnh chụp và chỉnh nghiêng ") 
    st.image('pages/card-crop-sample.jpg',caption='Ảnh mẫu và kết quả xử lý')

    updloaded_file = st.file_uploader("Upload ảnh", type=['jpg', 'png', 'jpeg','tif','tiff'])
    if not updloaded_file:
        return None
    
    file_bytes = np.asarray(bytearray(updloaded_file.read()), dtype=np.uint8)
    src = cv2.imdecode(file_bytes, 1)
    cropedImg,debugImg=cardCrop(src)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Ảnh gốc")
        st.image(src,channels='BGR')
    with col2:
        st.subheader("Định vị")
        st.image(debugImg,channels='BGR')
    with col3:
        st.subheader("Kết quả")
        st.image(cropedImg,channels='BGR')

    #save uploaded image
    cv2.imwrite("./upload/"+updloaded_file.name,src)

if __name__ == '__main__':
    main_loop()