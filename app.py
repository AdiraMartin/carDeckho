import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    file_id = "1i3V927wxBWcgjqj5JlI6YQ2ffT0uZQER"
    url = f"https://drive.google.com/uc?id={file_id}"
    return pd.read_csv(url)

df = load_data()

st.dataframe(df)
