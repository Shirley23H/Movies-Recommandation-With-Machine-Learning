import streamlit as st
import requests
from PIL import Image


# Streamlit setup
st.set_page_config(
    page_title="Pop Creuse",
    layout="centered",
    page_icon=':popcorn:'
)

# Background
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://raw.githubusercontent.com/MaximeBarski/streamlit_popcreuse/main/cartoon_sidebar_background.png");
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-image: url('https://raw.githubusercontent.com/MaximeBarski/streamlit_popcreuse/main/cartoon_sidebar_background.png');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("<h2 style='text-align: center; color: black;'>Bienvenue sur l'application Pop Creuse !</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Creuse et découvre ton prochain film préféré...</h3>", unsafe_allow_html=True)

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image('https://raw.githubusercontent.com/MaximeBarski/streamlit_popcreuse/main/sized_600x600.png')