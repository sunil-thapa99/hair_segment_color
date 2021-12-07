import streamlit as st
from PIL import Image
import numpy as np
import cv2

from skimage import io
import matplotlib.pyplot as plt
from skimage.transform import resize, rotate
from keras.models import load_model

from dnn_face_detection import detect_face

model = load_model('model.h5')

st.title("Hair Color Segmentation")

def predict(image):
    return model.predict(np.asarray([image]) ).reshape((224,224))

def display(img):
    plt.imshow(img,cmap='gray')
    plt.show()

data = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if data is not None:
    img = np.asarray(bytearray(data.read()), dtype=np.uint8)
    img = cv2.imdecode(img, 1)
    
    bounding_box = detect_face(img)

    for box in bounding_box:
        (x,y,w,h) = box.astype("int")

        crop_face = img[0:h, 0:img.shape[1]]

    res_img = cv2.resize(crop_face, (224, 224))
    img = cv2.cvtColor(crop_face, cv2.COLOR_BGR2GRAY)

    pred_image = resize(img,(224,224)).reshape((224,224,1))
    pred = predict(pred_image)
    rgb_img = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)

    title_container = st.container()
    col1, col2 = st.columns([10, 1])

    with title_container:
        with col1:
            slide = st.slider("Color Fade Value", 0.0, 1.0, step=0.1)
        with col2:
            color = st.color_picker('Pick A Color', '#00f900')
    hex_strip = color.lstrip('#')
    rgb = tuple(int(hex_strip[i:i+2], 16) for i in (0, 2, 4))
    
    # rgb_img[pred>0.5] = (105, 105, 105)
    rgb_img[pred>0.5] = rgb
    norm_image = cv2.normalize(rgb_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    
    new_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    
    combined = cv2.addWeighted(new_img, 1, norm_image, slide, 0)
    
    title_container = st.container()
    col1, col2 = st.columns(2)
    with title_container:
        with col1:
            col1.header("Original")
            col1.image(new_img, use_column_width=True)
        with col2:
            col2.header("Colored")
            col2.image(combined, use_column_width=True)

st.markdown("""
    <hr>
""", unsafe_allow_html=True)

def resize_img(img_path):
    img = cv2.imread(img_path)
    res_img = cv2.resize(img, (224, 224))
    new_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    return new_img

st.subheader("Samples")
title_container = st.container()
col1, col2 = st.columns(2)
with title_container:
    with col1:
        col1.image(resize_img('sample_results/1.jpg'), use_column_width=True)
        col1.image(resize_img('sample_results/2.jpg'), use_column_width=True)
        col1.image(resize_img('sample_results/3.jpg'), use_column_width=True)
        col1.image(resize_img('sample_results/5.jpg'), use_column_width=True)
    with col2:
        col2.image(resize_img('sample_results/1_result.jpeg'), use_column_width=True)
        col2.image(resize_img('sample_results/2_result.jpeg'), use_column_width=True)
        col2.image(resize_img('sample_results/3_result.jpeg'), use_column_width=True)
        col2.image(resize_img('sample_results/5_result.jpeg'), use_column_width=True)