from kukalib.cardcrop import *
from kukalib.docdeskew import *

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
    cropedImg2,debugImg2,(tl2,tr2,br2,bl2),hasCropped=cropDocument(src)
    if(not hasCropped):
        cropedImg2,angle,debugImg2=deskew(src)
 
    st.subheader("Ảnh gốc")
    st.image(src,channels='BGR')

   
    containerv2 = st.container()
    with containerv2:
        col4,col5 = st.columns(2)
        with col4:
            st.subheader("Định vị ")
            st.image(debugImg2,channels='BGR')
        with col5:
            st.subheader("Kết quả xử lý")
            st.image(cropedImg2,channels='BGR')

    #save uploaded image

if __name__ == '__main__':
    main_loop()