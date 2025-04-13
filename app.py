import streamlit as st
import pandas as pd
import altair as alt

# Set layout to wide
st.set_page_config(page_title="Car Dashboard", layout="wide")

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

    # Filters in columns
    st.header("Filters")

    # Create two columns for filters
    col_left, col_right = st.columns(2)

    # Left column filters
    with col_left:
        state_filter = st.selectbox("Select State", ["All"] + list(df["state"].unique()))
        location_filter = st.selectbox("Select Location Category", ["All"] + list(df["location_categories"].unique()))
        price_filter = st.slider("Price Range", int(df["pu"].min()), int(df["pu"].max()), (int(df["pu"].min()), int(df["pu"].max())))

    # Right column filters
    with col_right:
        km_filter = st.slider("KM Driven", int(df["km"].min()), int(df["km"].max()), (int(df["km"].min()), int(df["km"].max())))
        age_filter = st.slider("Age", int(df["age"].min()), int(df["age"].max()), (int(df["age"].min()), int(df["age"].max())))
        bt_filter = st.selectbox("Select Body Type", ["All"] + list(df["bt"].unique()))

    # Apply filters to the dataset
    filtered_df = df[
        ((df["state"] == state_filter) | (state_filter == "All")) &
        ((df["location_categories"] == location_filter) | (location_filter == "All")) &
        (df["pu"] >= price_filter[0]) & (df["pu"] <= price_filter[1]) &
        (df["km"] >= km_filter[0]) & (df["km"] <= km_filter[1]) &
        (df["age"] >= age_filter[0]) & (df["age"] <= age_filter[1]) &
        ((df["bt"] == bt_filter) | (bt_filter == "All"))
    ]

    # 1 Row - 5 Metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(label="Total Data", value=len(filtered_df))

    with col2:
        avg_km = filtered_df["km"].mean()
        st.metric(label="Average KM", value=f"{avg_km:,.0f} KM")

    with col3:
        avg_age = filtered_df["age"].mean()
        st.metric(label="Average Age", value=f"{avg_age:.1f} years")

    with col4:
        avg_price = filtered_df["pu"].mean()
        st.metric(label="Average Price", value=f"₹{avg_price:,.0f}")

    with col5:
        avg_discount = filtered_df["discountValue"].mean()
        st.metric(label="Average Discount", value=f"₹{avg_discount:,.0f}")

    st.markdown("---")

    # Top row with views and popularity score
    col_top1, col_top2 = st.columns(2)

    with col_top1:
        total_views = filtered_df["views"].sum()
        st.metric(label="Total Views", value=f"{total_views:,.0f}")

    with col_top2:
        popularity_score = total_views / len(filtered_df)
        st.metric(label="Popularity Score", value=f"{popularity_score:.2f}")

    st.markdown("---")

    # 3 columns side-by-side for charts
    col_chart1, col_chart2, col_chart3 = st.columns(3)

    with col_chart1:
        st.subheader("📊 Number of Cars per Location")
        location_counts = filtered_df["location_categories"].value_counts().reset_index()
        location_counts.columns = ["location", "count"]

        chart_loc = alt.Chart(location_counts).mark_bar().encode(
            x=alt.X("location", sort='-y', title="Location"),
            y=alt.Y("count", title="Number of Cars"),
            tooltip=["location", "count"]
        ).properties(
            width=350,
            height=400
        )

        st.altair_chart(chart_loc, use_container_width=True)

    with col_chart2:
        st.subheader("📈 Price Segments")
        
        # Custom Price Segmentation
        bins = [0, 2, 5, 8, 10, float('inf')]
        labels = ["0 - 2", "2 - 5", "5 - 8", "8 - 10", "10+"]
        filtered_df["price_segment"] = pd.cut(filtered_df["pu"], bins=bins, labels=labels, right=False)

        price_seg = filtered_df.groupby("price_segment").agg(
            count=("price_segment", "count"),
            avg_discount=("discountValue", "mean")
        ).reset_index()

        chart_seg = alt.Chart(price_seg).mark_bar().encode(
            x=alt.X("price_segment", sort='-y', title="Price Segment"),
            y=alt.Y("count", title="Number of Cars"),
            tooltip=["price_segment", "count", alt.Tooltip("avg_discount", format=".0f", title="Avg Discount")]
        ).properties(
            width=350,
            height=400
        )

        st.altair_chart(chart_seg, use_container_width=True)

    with col_chart3:
        st.subheader("💸 Average Discount per Price Segment")
        price_discount = filtered_df.groupby("price_segment").agg(
            avg_discount_value=("discountValue", "mean")
        ).reset_index()

        chart_discount = alt.Chart(price_discount).mark_bar().encode(
            x=alt.X("price_segment", sort='-y', title="Price Segment"),
            y=alt.Y("avg_discount_value", title="Average Discount (₹)"),
            tooltip=["price_segment", "avg_discount_value"]
        ).properties(
            width=350,
            height=400
        )

        st.altair_chart(chart_discount, use_container_width=True)

    st.markdown("---")

    # New Chart - Top 30 Models by Views
    st.subheader("🚗 Top 30 Models by Views")
    
    # Calculate count of models, sort by views, and display top 30
    model_counts = filtered_df.groupby("model").agg(
        total_views=("views", "sum")
    ).reset_index()
    
    model_counts = model_counts.sort_values(by="total_views", ascending=False).head(30)

    # Highlight the top 5 models with a different color
    top_5_models = model_counts.head(5)
    model_counts["color"] = model_counts["model"].apply(
        lambda x: "Top 5" if x in top_5_models["model"].values else "Other"
    )

    chart_model = alt.Chart(model_counts).mark_bar().encode(
        x=alt.X("model", sort='-y', title="Model"),
        y=alt.Y("total_views", title="Total Views"),
        color=alt.Color("color", scale=alt.Scale(domain=["Top 5", "Other"], range=["#FF5733", "#1F77B4"])),
        tooltip=["model", "total_views"]
    ).properties(
        width=800,  # Wider chart
        height=400
    )

    st.altair_chart(chart_model, use_container_width=True)

    st.markdown("---")
    st.subheader("Full Dataset")
    st.dataframe(filtered_df)

# Where to Sell
elif selected_tab == "Where to Sell?":
    st.title("📍 Where to Sell?")
    st.write("Coming soon...")

# Price Prediction
elif selected_tab == "Price Prediction":
    st.title("💰 Price Prediction")
    st.write("Coming soon...")
