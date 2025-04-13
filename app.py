import streamlit as st
import pandas as pd
import altair as alt

@st.cache_data
def load_data():
    file_id = "1i3V927wxBWcgjqj5JlI6YQ2ffT0uZQER"
    url = f"https://drive.google.com/uc?id={file_id}"
    return pd.read_csv(url)

df = load_data()

# Sidebar
st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio("Go to", ["Inside Data", "Where to Sell?", "Price Prediction"])

# Inside Data
if selected_tab == "Inside Data":
    st.title("🔍 Inside Data")

    # Row 1 - 3 Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Total Data", value=len(df))

    with col2:
        avg_km = df["km"].mean()
        st.metric(label="Average KM", value=f"{avg_km:,.0f} KM")

    with col3:
        avg_age = df["age"].mean()
        st.metric(label="Average Age", value=f"{avg_age:.1f} yrs")

    # Row 2 - 2 Metrics
    col4, col5 = st.columns(2)

    with col4:
        avg_price = df["pu"].mean()
        st.metric(label="Average Price", value=f"₹{avg_price:,.0f}")

    with col5:
        avg_discount = df["discountValue"].mean()
        st.metric(label="Average Discount", value=f"₹{avg_discount:,.0f}")

    st.markdown("---")

    # Bar chart - Mobil per lokasi
    st.subheader("📊 Jumlah Mobil per Lokasi")

    location_counts = df["location_categories"].value_counts().reset_index()
    location_counts.columns = ["location", "count"]

    chart = alt.Chart(location_counts).mark_bar().encode(
        x=alt.X("location", sort='-y', title="Location"),
        y=alt.Y("count", title="Jumlah Mobil"),
        tooltip=["location", "count"]
    ).properties(
        width=700,
        height=400
    )

    st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    st.subheader("Full Dataset")


# Where to Sell
elif selected_tab == "Where to Sell?":
    st.title("📍 Where to Sell?")
    st.write("Coming soon...")

# Price Prediction
elif selected_tab == "Price Prediction":
    st.title("💰 Price Prediction")
    st.write("Coming soon...")
