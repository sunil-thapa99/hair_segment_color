import streamlit as st
from PIL import Image

data = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if data is not None:
    st.image(data, use_column_width=True)
    img = Image.open(data)
    

