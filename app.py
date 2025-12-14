import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="YouTube Revenue Predictor",
    layout="wide",
    page_icon="🎥"
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    le_category = joblib.load("le_category.pkl")
    le_device = joblib.load("le_device.pkl")
    le_country = joblib.load("le_country.pkl")
    results_df = pd.read_pickle("results.pkl")
    return model, scaler, features, le_category, le_device, le_country, results_df

model, scaler, features, le_category, le_device, le_country, results_df = load_artifacts()


st.title("🎥 YouTube Ad Revenue Predictor")
st.markdown("### உங்கள் video metrics ஐ enter செய்து ad revenue predict செய்யுங்கள்")


col1, col2, col3 = st.columns(3)
with col1:
    views = st.number_input("👀 Views", min_value=0, value=10000, step=1000)
with col2:
    likes = st.number_input("👍 Likes", min_value=0, value=500, step=50)
with col3:
    comments = st.number_input("💬 Comments", min_value=0, value=50, step=10)

col1, col2, col3 = st.columns(3)
with col1:
    watch_time = st.number_input("⏱️ Watch Time (minutes)", min_value=0.0, value=300.0)
with col2:
    video_length = st.number_input("📏 Video Length (minutes)", min_value=0.0, value=10.0)
with col3:
    subscribers = st.number_input("📈 Subscribers", min_value=0, value=10000)


col1, col2, col3 = st.columns(3)
with col1:
    category = st.selectbox(
        "📂 Category",
        le_category.classes_.tolist()
    )
with col2:
    device = st.selectbox(
        "📱 Device",
        le_device.classes_.tolist()
    )
with col3:
    country = st.selectbox(
        "🌍 Country",
        le_country.classes_.tolist()
    )


if st.button("🔮 வருமானம் Predict செய்யவும்", type="primary", use_container_width=True):

    input_data = pd.DataFrame({
        "views": [views],
        "likes": [likes],
        "comments": [comments],
        "watch_time_minutes": [watch_time],
        "video_length_minutes": [video_length],
        "subscribers": [subscribers],
        "engagement_rate": [(likes + comments) / (views + 1)],
        "revenue_per_view": [(likes + comments) / (views + 1) * 0.001],
        "category_encoded": [le_category.transform([category])[0]],
        "device_encoded": [le_device.transform([device])[0]],
        "country_encoded": [le_country.transform([country])[0]],
    })[features]

    
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
    except:
        prediction = model.predict(input_data)[0]

    st.success(f"🎉 **Predicted Ad Revenue: ${prediction:.2f} USD**")
    st.balloons()


st.subheader("📊 Model Performance Comparison")
st.dataframe(
    results_df.style.highlight_max(axis=0, color="lightgreen"),
    use_container_width=True
)

st.markdown("---")
st.caption("Content Monetization Modeler | Streamlit Deployment")
