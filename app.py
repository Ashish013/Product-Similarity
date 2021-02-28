from PIL import Image
import streamlit as st
import time
from model import calculate_similarity

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(layout="wide")
img_list = ["jpg","jpeg","png","jfif"]
st.header("Predicting similarity score of products")
st.write("")

# https://wallpapercave.com/wp/wp3536889.png
page_bg_img = '''<style>body {background-image: url("https://wallpapercave.com/wp/wp5612914.jpg");background-size: cover;}</style>'''
st.markdown(page_bg_img, unsafe_allow_html=True)

col1,col2 = st.beta_columns(2)

img1 = col1.file_uploader("Upload Image-1", type = img_list)
if img1 is not None:
    image1 = Image.open(img1)
    col1.image(image1, caption='Image-1', use_column_width=True)

img2 = col2.file_uploader("Upload Image-2", type = img_list)
if img2 is not None:
    image2 = Image.open(img2)
    col2.image(image2, caption='Image-2', use_column_width=True)


t = st.sidebar.slider("Similarity Threshold:",0,100,16)
click = st.button("Run similarity model")
if click:
    if (img1 is not None) and (img2 is not None):
        text = st.empty()
        st.write("")
        text.write("Processing the images....")
        time.sleep(1.5)
        text.write("Predicting the similarity score....")
        results = calculate_similarity(image1, image2, t)

        # print out the similarity score
        text.write(f"The two products are {results[0]} with a similarity score of {results[1]}")
    else:
        text = st.empty()
        text.write("Upload images to run a similarity model !")
        time.sleep(2)
        text.write("")

