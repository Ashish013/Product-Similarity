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

loss_dict = {"Contrastive Loss" : 1,"Binary Cross-Entropy" : 2}
loss_func = st.selectbox("Select a loss function: ",["Binary Cross-Entropy","Contrastive Loss"],1)

if loss_func == "Binary Cross-Entropy":
    thresh = st.slider("Confidence Interval: ",0.0,1.0,0.5,0.1)
else:
    #thresh = st.slider("Inverse-Confidence Interval: ",10,25,17,1)
    thresh = 17

st.write("")
click = st.button("Run similarity model")

if click:
    if (img1 is not None) and (img2 is not None):
        text = st.empty()
        text.write("### Processing the images....")
        time.sleep(1.5)
        text.write("### Predicting the similarity score....")
        prediction = calculate_similarity(image1, image2, loss_dict[loss_func],text)

        if loss_func == "Contrastive Loss":
            # Contrastive Loss
            if prediction > thresh:
                result = "dissimilar"
            else:
                result = "similar"
            text.write(f"### Output: The two products are **{result}** and can be grouped in to a similar class.")

        else:
            # Binary Cross-Entropy loss
            if prediction > thresh:
                result = "similar"
            else:
                result = "dissimilar"
            text.write(f"### Output: The two products are **{result}**  with a similarity score of {prediction}.")

    else:
        text = st.empty()
        text.write("Upload images to run a similarity model !")
        time.sleep(2)
        text.write("")

