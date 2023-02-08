from cardcrop.cardcrop import *

import streamlit as st

import cv2
import streamlit as st
import numpy as np

def main_loop():
    st.title("Demo Tool: Crop ảnh thẻ")
    st.text("Input: Ảnh chụp thẻ (ID Card/Drive license/Bank car/Student card..) bị nghiêng / xiên") 
    st.text("Output: Ảnh crop thẻ và nắn thẳng")


    updloaded_file = st.subheader.file_uploader("Upload ảnh", type=['jpg', 'png', 'jpeg'])
    if not updloaded_file:
        return None
    
    file_bytes = np.asarray(bytearray(updloaded_file.read()), dtype=np.uint8)
    src = cv2.imdecode(file_bytes, 1)
    cropedImg,debugImg=cardCrop(src)
    
    st.text("Ảnh gốc:")
    st.image(src,channels='BGR')
    st.text("Ảnh định vị")
    st.image(debugImg,channels='BGR')
    st.text("Kết quả xử lý")
    st.image(cropedImg,channels='BGR')

    #save uploaded image
    cv2.imwrite("./upload/"+updloaded_file.name,src)

if __name__ == '__main__':
    main_loop()