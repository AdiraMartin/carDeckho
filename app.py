import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    file_id = "1i3V927wxBWcgjqj5JlI6YQ2ffT0uZQER"
    url = f"https://drive.google.com/uc?id={file_id}"
    return pd.read_csv(url)

df = load_data()

st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio("Go to", ["Inside Data", "Where to Sell?", "Price Prediction"])

# Konten berdasarkan tab yang dipilih
if selected_tab == "Inside Data":
    st.title("🔍 Inside Data")
    # Row for
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Total Data", value=len(df))

    with col2:
        avg_km = df["km"].mean()
        st.metric(label="Average KM", value=f"{avg_km:,.0f} KM")

    st.markdown("---")  


elif selected_tab == "Where to Sell?":
    st.title("📍 Where to Sell?")


elif selected_tab == "Price Prediction":
    st.title("💰 Price Prediction")

