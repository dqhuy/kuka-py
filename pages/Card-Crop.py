from kukalib.cardcrop import *

import streamlit as st

import cv2
import streamlit as st
import numpy as np
import datetime

def main_loop():
    versionInfo = getVersionInfo()

    st.set_page_config(page_title="Card Croping Tool")
    st.title("Demo Tool: Card Croping")
    st.text("Version: " + versionInfo["version"] + " - Release date:" + versionInfo["date"].strftime("%Y-%m-%d"))

    st.subheader("Crop card từ ảnh chụp và chỉnh nghiêng ") 
    st.image('pages/card-crop-sample.jpg',caption='Ảnh mẫu và kết quả xử lý')

    
    updloaded_file = st.file_uploader("Upload ảnh", type=['jpg', 'png', 'jpeg','tif','tiff'])
    if not updloaded_file:
        return None
    
    file_bytes = np.asarray(bytearray(updloaded_file.read()), dtype=np.uint8)
    src = cv2.imdecode(file_bytes, 1)
    cropedImg,debugImg,(tl,tr,br,bl)=cardCrop(src)
    cropedImg2,debugImg2,(tl2,tr2,br2,bl2)=cardCrop2(src)
    
 
    st.subheader("Ảnh gốc")
    st.image(src,channels='BGR')

    containerv1 = st.container()
    with containerv1:
        col2, col3 = st.columns(2)
        with col2:
            st.subheader("Định vị v1")
            st.image(debugImg,channels='BGR')
        with col3:
            st.subheader("Kết quả v1")
            st.image(cropedImg,channels='BGR')

    containerv2 = st.container()
    with containerv2:
        col4,col5 = st.columns(2)
        with col4:
            st.subheader("Định vị v2")
            st.image(debugImg2,channels='BGR')
        with col5:
            st.subheader("Kết quả v2")
            st.image(cropedImg2,channels='BGR')

    #save uploaded image

if __name__ == '__main__':
    main_loop()