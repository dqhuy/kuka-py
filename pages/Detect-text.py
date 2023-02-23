from kukalib.detect import *

import streamlit as st

import cv2
import streamlit as st
import numpy as np

from google.oauth2 import service_account
from google.cloud import storage

from tempfile import TemporaryFile

import os

def saveToStorage(filename,imagedata):
    # Create API client.
    credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"])
    
    client = storage.Client(credentials=credentials)

    bucketName="kuka-py-app"
    
    bucket=client.bucket(bucket_name=bucketName)

    with TemporaryFile() as tempFile:
        tempFilePath = "".join([str(tempFile.name),".jpg"])
        tempFolder,tempName=os.path.split(tempFilePath)
        cv2.imwrite(tempFilePath,imagedata)
        blob = bucket.blob("error-deskew/"+tempName)
        blob.upload_from_filename(filename=tempFilePath,content_type="image/jpeg")
        blob.make_public()

        url=blob.public_url
        print("update success: " + blob.public_url)
        #clear tem file
        os.remove(tempFilePath)
        return url
    
def main_loop():
    st.set_page_config(page_title="Document deskew tool")
    st.title("Demo Tool: Detect text")
    st.subheader("Khoanh vùng có text") 
    updloaded_file = st.file_uploader("Upload ảnh", type=['jpg', 'png', 'jpeg','tif','tiff'])
    if not updloaded_file:
        return None
    
    file_bytes = np.asarray(bytearray(updloaded_file.read()), dtype=np.uint8)
    src = cv2.imdecode(file_bytes, 1)
    x,y,w,h=getTextPosition(src)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh gốc")
        st.image(src,channels='BGR')
    with col2:
        result=src.copy()
        if(w>0):
            result=cv2.rectangle(src,(x,y),(x+w,y+h),(9,0,255),1)
        st.subheader("Kết quả")
        st.image(result,channels='BGR')
        if(w==0):
            st.text("NO TEXT")

    #save uploaded image
    # with st.form("my_form"):
    #     st.write("Bạn vui lòng bấm nút [Báo lỗi] nếu thấy kết quả xử lý không tốt")

    #     # Every form must have a submit button.
    #     submitted = st.form_submit_button("Báo lỗi",)
    #     if submitted:
    #         #cv2.imwrite("./error-deskew/"+updloaded_file.name,src)
    #         url=saveToStorage(updloaded_file.name,src)
    #         st.text("Cảm ơn bạn đã báo lỗi!")
    #         st.write("File đã lưu: ["+url+"](" + url + ")")

if __name__ == '__main__':
    main_loop()