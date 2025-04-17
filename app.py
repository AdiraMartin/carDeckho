import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import joblib
import requests
import io

# Set layout to wide
st.set_page_config(page_title="Car Dashboard", layout="wide")

@st.cache_data
def load_data():
    file_id = "1X-Id3JZELUMNaqPKqwEY1GNcfbCr6FP9"
    url = f"https://drive.google.com/uc?id={file_id}"
    return pd.read_csv(url)

df = load_data()

# Sidebar
st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio("Go to", ["Inside Data", "Where to Sell?", "Price Prediction"])

if selected_tab == "Inside Data":
    st.title("🔍 Inside Data")


if selected_tab == "Inside Data":
    st.title("🔍 Inside Data")

    # ==== Predicted Year Filter ====
    predicted_year = st.selectbox("Select Predicted Year", ["2023", "2024"])

    if predicted_year == "2023":
        df["predicted_price"] = df["predicted_price_2023"]
        df["lower_bound"] = df["lower_bound_2023"]
        df["upper_bound"] = df["upper_bound_2023"]
        df["age"] = df["age_2023"]
    else:
        df["predicted_price"] = df["predicted_price_2024"]
        df["lower_bound"] = df["lower_bound_2024"]
        df["upper_bound"] = df["upper_bound_2024"]
        df["age"] = df["age_2024"]

    # ==== Filters ====
    st.header("Filters")
    col_left, col_right = st.columns(2)

    with col_left:
        location_filter = st.selectbox("Select Location Category", ["All"] + list(df["location_categories"].unique()))
        state_filter = st.selectbox("Select State", ["All"] + list(df["state"].unique()))
        model_filter = st.selectbox("Select Model", ["All"] + list(df["model_name"].unique()))
        bt_filter = st.selectbox("Select Body Type", ["All"] + list(df["bt"].unique()))

    with col_right:
        price_filter = st.slider("Price", 0, int(df["price in rupias"].max()), (0, int(df["price in rupias"].max())))
        discount_filter = st.slider("Discount Range", 0, int(df["discountValue"].max()), (0, int(df["discountValue"].max())))
        km_filter = st.slider("KM Driven", int(df["km"].min()), int(df["km"].max()), (int(df["km"].min()), int(df["km"].max())))

    submit_button = st.button(label="Submit Filters")

    if submit_button:
        filtered_df = df[
            ((df["location_categories"] == location_filter) | (location_filter == "All")) &
            ((df["state"] == state_filter) | (state_filter == "All")) &
            ((df["model_name"] == model_filter) | (model_filter == "All")) &
            ((df["bt"] == bt_filter) | (bt_filter == "All")) &
            (df["price in rupias"] >= price_filter[0] * 100000) & (df["price in rupias"] <= price_filter[1] * 100000) &
            (df["discountValue"] >= discount_filter[0]) & (df["discountValue"] <= discount_filter[1]) &
            (df["km"] >= km_filter[0]) & (df["km"] <= km_filter[1])
        ]

        # === Metrics ===
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.metric(label="Total Data", value=len(filtered_df))
        with col2: st.metric(label="Average KM", value=f"{filtered_df['km'].mean():,.0f} KM")
        with col3: st.metric(label="Average Age", value=f"{filtered_df['age'].mean():.1f} years")
        with col4: st.metric(label="Average Price", value=f"₹{filtered_df['price in rupias'].mean():,.0f}")
        with col5: st.metric(label="Average Discount", value=f"₹{filtered_df['discountValue'].mean():,.0f}")

        st.markdown("---")

        col_top1, col_top2 = st.columns(2)
        with col_top1:
            total_views = filtered_df["views"].sum()
            st.metric(label="Total Views", value=f"{total_views:,.0f}")
        with col_top2:
            popularity_score = total_views / len(filtered_df)
            st.metric(label="Popularity Score", value=f"{popularity_score:.2f}")

        st.markdown("---")

        # === Charts: Location, Segment, Discount ===
        col_chart1, col_chart2, col_chart3 = st.columns(3)

        with col_chart1:
            st.subheader("📊 Number of Cars per Location")
            location_counts = filtered_df["location_categories"].value_counts().reset_index()
            location_counts.columns = ["location", "count"]
            chart_loc = alt.Chart(location_counts).mark_bar().encode(
                x=alt.X("location", sort='-y', title="Location"),
                y=alt.Y("count", title="Number of Cars"),
                tooltip=["location", "count"]
            ).properties(width=350, height=400)
            st.altair_chart(chart_loc, use_container_width=True)

        with col_chart2:
            st.subheader("📈 Price Segments")
            price_seg = filtered_df.groupby("price_segment").agg(
                count=("price_segment", "count"),
                avg_discount=("discountValue", "mean")
            ).reset_index()
            price_order = ["0lakh-2lakh", "2lakh-5lakh", "5lakh-8lakh", "8lakh-10lakh", "10+lakh"]
            price_seg["price_segment"] = pd.Categorical(price_seg["price_segment"], categories=price_order, ordered=True)
            price_seg = price_seg.sort_values("price_segment")
            chart_seg = alt.Chart(price_seg).mark_bar().encode(
                x=alt.X("price_segment", title="Price Segment"),
                y=alt.Y("count", title="Number of Cars"),
                tooltip=["price_segment", "count", alt.Tooltip("avg_discount", format=".0f")]
            ).properties(width=350, height=400)
            st.altair_chart(chart_seg, use_container_width=True)

        with col_chart3:
            st.subheader("💸 Average Discount per Price Segment")
            price_discount = filtered_df.groupby("price_segment").agg(
                avg_discount_value=("discountValue", "mean")
            ).reset_index()
            price_discount["price_segment"] = pd.Categorical(price_discount["price_segment"], categories=price_order, ordered=True)
            price_discount = price_discount.sort_values("price_segment")
            chart_discount = alt.Chart(price_discount).mark_bar().encode(
                x=alt.X("price_segment", title="Price Segment"),
                y=alt.Y("avg_discount_value", title="Average Discount (₹)"),
                tooltip=["price_segment", alt.Tooltip("avg_discount_value", format=".0f")]
            ).properties(width=350, height=400)
            st.altair_chart(chart_discount, use_container_width=True)

        # === Chart: Top 30 Models by Views ===
        st.subheader("🚗 Top 30 Models by Views")
        model_counts = filtered_df.groupby("model_name").agg(total_views=("views", "sum")).reset_index()
        model_counts = model_counts.sort_values(by="total_views", ascending=False).head(30)
        top_5_models = model_counts.head(5)
        model_counts["color"] = model_counts["model_name"].apply(
            lambda x: "Top 5" if x in top_5_models["model_name"].values else "Other"
        )
        chart_model = alt.Chart(model_counts).mark_bar().encode(
            x=alt.X("model_name", sort='-y', title="Model"),
            y=alt.Y("total_views", title="Total Views"),
            color=alt.Color("color", scale=alt.Scale(domain=["Top 5", "Other"], range=["#FF5733", "#1F77B4"])),
            tooltip=["model_name", "total_views"]
        ).properties(width=800, height=400)
        st.altair_chart(chart_model, use_container_width=True)

        # === New Chart: Predicted Price + Confidence Bounds ===
        st.markdown("---")
        st.subheader(f"📉 Predicted Price & Confidence Band ({predicted_year})")

        plot_data = filtered_df[["model_name", "predicted_price", "lower_bound", "upper_bound"]].copy()
        plot_data = plot_data.groupby("model_name").agg({
            "predicted_price": "mean",
            "lower_bound": "mean",
            "upper_bound": "mean"
        }).reset_index()
        plot_data = plot_data.sort_values("predicted_price", ascending=False).head(30)
        plot_melted = plot_data.melt(id_vars="model_name",
                                     value_vars=["predicted_price", "lower_bound", "upper_bound"],
                                     var_name="Type", value_name="Price")
        line_chart = alt.Chart(plot_melted).mark_line(point=True).encode(
            x=alt.X("model_name:N", sort="-y", title="Model"),
            y=alt.Y("Price:Q", title="Predicted Price (₹)"),
            color=alt.Color("Type:N", scale=alt.Scale(
                domain=["predicted_price", "lower_bound", "upper_bound"],
                range=["#1f77b4", "#2ca02c", "#d62728"]),
                title="Price Type"),
            tooltip=["model_name", "Type", alt.Tooltip("Price", format=",.0f")]
        ).properties(width=800, height=400)

        st.altair_chart(line_chart, use_container_width=True)

        # === Data Table ===
        st.markdown("---")
        st.subheader("Full Dataset")
        st.dataframe(filtered_df)

# Inside Where to Sell tab
elif selected_tab == "Where to Sell?":
    st.title("📍 Where to Sell?")

    st.header("Most Popular Locations to Sell Cars (Based on State)")

    # Filter for price segment
    price_segment_filter = st.selectbox(
        "Select Price Segment",
        ["All", "0lakh-2lakh", "2lakh-5lakh", "5lakh-8lakh", "8lakh-10lakh", "10+lakh"]
    )

    # Get the most common states and the average price for each state
    state_counts = df.groupby("state").agg(
        number_of_cars=("state", "count"),
        average_price=("price in rupias", "mean"),
        average_discount=("discountValue", "mean")  # Calculate average discount per state
    ).reset_index()
    state_counts.columns = ["State", "Number of Cars", "Average Price", "Average Discount"]

    # Filter by price segment if the user selects a specific segment
    if price_segment_filter != "All":
        df_filtered = df[df["price_segment"] == price_segment_filter]
        state_counts_filtered = df_filtered.groupby("state").agg(
            number_of_cars=("state", "count"),
            average_price=("price in rupias", "mean"),
            average_discount=("discountValue", "mean")
        ).reset_index()
        state_counts_filtered.columns = ["State", "Number of Cars", "Average Price", "Average Discount"]
    else:
        state_counts_filtered = state_counts  # If "All" is selected, show all states

    # Input for margin and target sales percentage
    margin_percentage = st.number_input("Input Margin (%)", min_value=0, max_value=100, value=5) / 100  # Convert to decimal
    target_sales_percentage = st.number_input("Input Target Sales (%)", min_value=0, max_value=100, value=10) / 100  # Convert to decimal

    # Input for profit thresholds
    high_profit_threshold = st.number_input("High Profit Threshold (₹)", min_value=0, value=100000)
    moderate_profit_threshold = st.number_input("Moderate Profit Threshold (₹)", min_value=0, value=50000)

    # Calculate profit per state considering the discount
    state_counts_filtered["Profit per Car"] = (state_counts_filtered["Average Price"] - state_counts_filtered["Average Discount"]) * margin_percentage
    state_counts_filtered["Profit for Target Sales"] = state_counts_filtered["Number of Cars"] * target_sales_percentage * state_counts_filtered["Profit per Car"]

    # Display a bar chart for the most popular states
    state_chart = alt.Chart(state_counts_filtered).mark_bar().encode(
        x=alt.X("State", sort='-y', title="State"),
        y=alt.Y("Profit for Target Sales", title="Profit for Target Sales (₹)"),
        tooltip=["State", "Profit for Target Sales"]
    ).properties(
        width=700,
        height=400
    )

    st.altair_chart(state_chart, use_container_width=True)

    st.markdown("---")

    st.header("Profit Calculation and Analysis")

    st.write(
        "Based on the data, we calculate the expected profit for each state based on the following assumptions:\n"
        "- **Margin**: The profit margin for each car sold. Adjust the margin to see the impact.\n"
        "- **Target Sales**: The percentage of cars expected to be sold (e.g., 10% means we expect to sell 10% of the cars listed in that state).\n"
        "- **Discount**: The discount applied to the cars, which will reduce the total price used for calculating profit.\n"
        "The chart above shows the profit for selling the target percentage of cars in each state based on the margin, target sales, and discount you entered."
    )

    # Display state profit data
    st.write(state_counts_filtered[["State", "Number of Cars", "Average Price", "Average Discount", "Profit for Target Sales"]])

    st.markdown("---")

    # Selling recommendations based on profit analysis
    st.header("Selling Recommendations")

    st.write(
        "Based on the calculated profit, here are some recommendations:"
    )

    for _, row in state_counts_filtered.iterrows():
        if row["Profit for Target Sales"] > high_profit_threshold:  # Dynamic threshold for high profit
            st.write(f"✔️ **{row['State']}**: High potential for profit with ₹{row['Profit for Target Sales']:,.0f} expected for target sales.")
        elif row["Profit for Target Sales"] > moderate_profit_threshold:
            st.write(f"⚠️ **{row['State']}**: Moderate profit potential with ₹{row['Profit for Target Sales']:,.0f} expected for target sales.")
        else:
            st.write(f"❌ **{row['State']}**: Low profit potential with ₹{row['Profit for Target Sales']:,.0f} expected for target sales.")

    st.markdown("---")
    
    st.write("You can adjust the assumptions (e.g., margin, target sales) to analyze different scenarios.")

elif selected_tab == "Price Prediction":
    st.title("🚗 Car Price Prediction")

    # --- Load Hugging Face Resources ---
    @st.cache_resource
    def load_from_huggingface(url):
        response = requests.get(url)
        return joblib.load(io.BytesIO(response.content))

    base_url = "https://huggingface.co/AdiraMartin/cardekho-price-model/resolve/main/"

    # Select year
    selected_year = st.selectbox("Select Prediction Year", [2024, 2023])

    if selected_year == 2024:
        rf_model = load_from_huggingface(base_url + "rf_model_2024.pkl")
        scaler = load_from_huggingface(base_url + "scaler_2024.pkl")
        feature_columns = load_from_huggingface(base_url + "feature_columns_2024.pkl")
    else:
        rf_model = load_from_huggingface(base_url + "rf_model_2023.pkl")
        scaler = load_from_huggingface(base_url + "scaler_2023.pkl")
        feature_columns = load_from_huggingface(base_url + "feature_columns_2023.pkl")

    encoders = load_from_huggingface(base_url + "encoders.pkl")
    mappings = load_from_huggingface(base_url + "mappings.pkl")

    # --- Helper function for encoding ---
    def get_encoded_input(label, encoder):
        class_list = list(encoder.classes_)
        selected_label = st.selectbox(label, class_list)
        encoded_value = encoder.transform([selected_label])[0]
        return encoded_value

    # --- User Input ---
    st.subheader("Enter Car Details")

    state = get_encoded_input("State", encoders['state'])
    brand = get_encoded_input("Brand", encoders['brand_name'])
    model_name = get_encoded_input("Model Name", encoders['model_name'])
    variant_name = get_encoded_input("Variant Name", encoders['variant_name'])
    fuel_type = get_encoded_input("Fuel Type", encoders['ft'])
    body_type = get_encoded_input("Body Type", encoders['bt'])

    transmission = st.radio("Transmission", list(mappings['tt'].keys()))
    user_type = st.radio("User Type", list(mappings['utype'].keys()))
    km_driven = st.number_input("Kilometers Driven", value=30000)
    seating = st.selectbox("Seating Capacity", [2, 4, 5, 6, 7])
    myear = st.number_input("Manufacturing Year", min_value=2000, max_value=2025, value=2019)

    # --- Predict Button ---
    if st.button("🔮 Predict Price"):
        age = selected_year - myear
        df_input = pd.DataFrame([{
            'state': state,
            'brand_name': brand,
            'model_name': model_name,
            'variant_name': variant_name,
            'ft': fuel_type,
            'bt': body_type,
            'tt': mappings['tt'][transmission],
            'utype': mappings['utype'][user_type],
            'log_km': np.log1p(km_driven),
            'seating_capacity_new': seating,
            'myear': myear,
            'top_features_count': 0,  # Default if not used
            'has_acceleration': 1,    # Assume yes if not known
            f'age_{selected_year}': age
        }])

        # Pastikan semua kolom match
        df_input = df_input.reindex(columns=feature_columns, fill_value=0)

        try:
            X_scaled = scaler.transform(df_input)
            predicted_price = rf_model.predict(X_scaled)[0]
            lower = predicted_price * 0.9
            upper = predicted_price * 1.1

            st.success(f"💰 Estimated car price: ₹ {int(predicted_price):,}")
            st.write(f"📉 Lower Bound: ₹ {int(lower):,}")
            st.write(f"📈 Upper Bound: ₹ {int(upper):,}")

        except Exception as e:
            st.error(f"Prediction failed. Please check your input. Error: {e}")
