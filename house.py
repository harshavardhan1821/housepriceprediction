import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }

    .title {
        text-align: center;
        color: #1f2937;
        font-size: 42px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        color: gray;
        margin-bottom: 30px;
    }

    .prediction-box {
        background-color: #2563eb;
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    }

    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
        border: none;
    }

    .stButton>button:hover {
        background-color: #1e40af;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SAMPLE DATASET
# ---------------------------------------------------

data = {
    "Area": [1000, 1200, 1500, 1800, 2000, 2200, 2500],
    "Bedrooms": [2, 2, 3, 3, 4, 4, 5],
    "Age": [10, 8, 6, 5, 4, 3, 2],
    "Price": [200000, 250000, 320000, 380000, 450000, 500000, 600000]
}

df = pd.DataFrame(data)

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------

X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown("<div class='title'>🏠 House Price Prediction</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>Predict house prices using Machine Learning & Linear Regression</div>",
    unsafe_allow_html=True
)

# ---------------------------------------------------
# LAYOUT
# ---------------------------------------------------

col1, col2 = st.columns([1, 1])

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

with col1:

    st.subheader("Enter House Details")

    area = st.number_input(
        "Area (Square Feet)",
        min_value=500,
        max_value=5000,
        value=1500,
        step=100
    )

    bedrooms = st.slider(
        "Number of Bedrooms",
        1,
        10,
        3
    )

    age = st.slider(
        "House Age (Years)",
        0,
        50,
        5
    )

    predict_btn = st.button("Predict House Price")

# ---------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------

with col2:

    st.subheader("Prediction Result")

    if predict_btn:

        input_data = pd.DataFrame({
            "Area": [area],
            "Bedrooms": [bedrooms],
            "Age": [age]
        })

        prediction = model.predict(input_data)[0]

        st.markdown(
            f"""
            <div class='prediction-box'>
                Estimated Price<br><br>
                ${prediction:,.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.info("Enter details and click Predict")

# ---------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------

st.markdown("---")

st.subheader("📊 Training Dataset")

st.dataframe(df, use_container_width=True)

# ---------------------------------------------------
# MODEL INFO
# ---------------------------------------------------

st.markdown("---")

st.subheader("ℹ️ Model Information")

st.write("Algorithm Used: Linear Regression")

st.write("Features Used:")
st.write("- Area")
st.write("- Bedrooms")
st.write("- House Age")