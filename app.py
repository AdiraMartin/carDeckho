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
        location_filter = st.selectbox("Select Location Category", ["All"] + list(df["location_categories"].unique()))
        state_filter = st.selectbox("Select State", ["All"] + list(df["state"].unique()))
        bt_filter = st.selectbox("Select Body Type", ["All"] + list(df["bt"].unique()))

    # Right column filters
    with col_right:
        # Price range slider (pu column)
        price_filter = st.slider("Price", 0, int(df["pu"].max()), (0, int(df["pu"].max())))
        discount_filter = st.slider("Discount Range", 0, int(df["discountValue"].max()), (0, int(df["discountValue"].max())))
        km_filter = st.slider("KM Driven", int(df["km"].min()), int(df["km"].max()), (int(df["km"].min()), int(df["km"].max())))

    # Add a submit button to apply the filters
    submit_button = st.button(label="Submit Filters")

    # Only apply the filter when the submit button is clicked
    if submit_button:
        # Apply filters to the dataset with dependencies
        filtered_df = df[
            ((df["location_categories"] == location_filter) | (location_filter == "All")) &
            ((df["state"] == state_filter) | (state_filter == "All")) &
            ((df["bt"] == bt_filter) | (bt_filter == "All")) &
            (df["pu"] >= price_filter[0] * 100000) & (df["pu"] <= price_filter[1] * 100000) &  # Convert to Lakh
            (df["discountValue"] >= discount_filter[0]) & (df["discountValue"] <= discount_filter[1]) &
            (df["km"] >= km_filter[0]) & (df["km"] <= km_filter[1])
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
            
            # Grouping & ordering for Price Segments
            price_seg = filtered_df.groupby("price_segment").agg(
                count=("price_segment", "count"),
                avg_discount=("discountValue", "mean")
            ).reset_index()
        
            # Set manual order
            price_order = ["0lakh-2lakh", "2lakh-5lakh", "5lakh-8lakh", "8lakh-10lakh", "10+lakh"]
            price_seg["price_segment"] = pd.Categorical(price_seg["price_segment"], 
                                                        categories=price_order, 
                                                        ordered=True)
            price_seg = price_seg.sort_values("price_segment")
        
            # Chart for number of cars per price segment
            chart_seg = alt.Chart(price_seg).mark_bar().encode(
                x=alt.X("price_segment", title="Price Segment"),
                y=alt.Y("count", title="Number of Cars"),
                tooltip=["price_segment", "count", alt.Tooltip("avg_discount", format=".0f", title="Avg Discount")]
            ).properties(
                width=350,
                height=400
            )
        
            st.altair_chart(chart_seg, use_container_width=True)
        
        with col_chart3:
            st.subheader("💸 Average Discount per Price Segment")
        
            # Grouping & ordering for Average Discount
            price_discount = filtered_df.groupby("price_segment").agg(
                avg_discount_value=("discountValue", "mean")
            ).reset_index()
        
            price_discount["price_segment"] = pd.Categorical(price_discount["price_segment"], 
                                                             categories=price_order, 
                                                             ordered=True)
            price_discount = price_discount.sort_values("price_segment")
        
            # Chart for average discount
            chart_discount = alt.Chart(price_discount).mark_bar().encode(
                x=alt.X("price_segment", title="Price Segment"),
                y=alt.Y("avg_discount_value", title="Average Discount (₹)"),
                tooltip=["price_segment", alt.Tooltip("avg_discount_value", format=".0f", title="Avg Discount")]
            ).properties(
                width=350,
                height=400
            )
        
            st.altair_chart(chart_discount, use_container_width=True)


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
        average_price=("pu", "mean"),
        average_discount=("discountValue", "mean")  # Calculate average discount per state
    ).reset_index()
    state_counts.columns = ["State", "Number of Cars", "Average Price", "Average Discount"]

    # Filter by price segment if the user selects a specific segment
    if price_segment_filter != "All":
        df_filtered = df[df["price_segment"] == price_segment_filter]
        state_counts_filtered = df_filtered.groupby("state").agg(
            number_of_cars=("state", "count"),
            average_price=("pu", "mean"),
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
    @st.cache_resource
    def load_from_huggingface(url):
        response = requests.get(url)
        return joblib.load(io.BytesIO(response.content))
    
    # Load model dan encoder dari HuggingFace
    base_url = "https://huggingface.co/AdiraMartin/cardekho-price-model/resolve/main/"
    rf_model = load_from_huggingface(base_url + "rf_model.pkl")
    scaler = load_from_huggingface(base_url + "scaler.pkl")
    encoders = load_from_huggingface(base_url + "encoders.pkl")
    mappings = load_from_huggingface(base_url + "mappings.pkl")
    
    def get_encoded_input(label, encoder):
        label_list = list(encoder.classes_)  # List of labels to show in the dropdown
        selected_label = st.selectbox(label, label_list)  # Dropdown of labels
        encoded_value = encoder.transform([selected_label])[0]  # Encoded numeric value for the model
        return encoded_value, selected_label

    # === Input Section ===
    st.subheader("Masukkan Detail Mobil")
    
    state, _ = get_encoded_input("State", encoders['state'])
    brand, _ = get_encoded_input("Brand", encoders['brand_name'])
    model_name, _ = get_encoded_input("Model Name", encoders['model_name'])
    variant_name, _ = get_encoded_input("Variant Name", encoders['variant_name'])
    fuel_type, _ = get_encoded_input("Fuel Type", encoders['ft'])
    body_type, _ = get_encoded_input("Body Type", encoders['bt'])
    st.write("State classes:", encoders['state'].classes_)
    
    tt = st.radio("Transmission Type", list(mappings['tt'].keys()))
    utype = st.radio("User Type", list(mappings['utype'].keys()))
    
    km = st.number_input("Kilometer Driven", value=30000)
    discount = st.slider("Discount (Rp)", 0, 50000000, 0, step=100000)
    seating = st.selectbox("Seating Capacity", [2, 4, 5, 6, 7])
    
    # === Prediction Section ===
    if st.button("Prediksi Harga"):
        # Prepare input for prediction
        df_input = pd.DataFrame([{
            'state': state,
            'brand_name': brand,
            'model_name': model_name,
            'variant_name': variant_name,
            'ft': fuel_type,
            'bt': body_type,
            'tt': mappings['tt'][tt],
            'utype': mappings['utype'][utype],
            'log_km': np.log1p(km),
            'discountValue': discount,
            'seating_capacity_new': seating
        }])
    
        # Scale the input data
        X_scaled = scaler.transform(df_input)
    
        # Make the prediction
        pred_price = rf_model.predict(X_scaled)[0]
    
        # Show the result
        st.success(f"💰 Perkiraan harga mobil: Rp {int(pred_price):,}")
