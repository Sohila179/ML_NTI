
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# 1. Page Configuration
# =========================

st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🚗",
    layout="wide"
)

# =========================
# 2. Custom CSS
# =========================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #10152f 0%, #20244b 50%, #35245c 100%);
    color: white;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #171b3b, #292052);
}

h1, h2, h3 {
    color: white !important;
}

p, label {
    color: #d6d9f0 !important;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #353b70, #59438c);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #7163bd;
}

.stButton > button {
    background: linear-gradient(90deg, #7458e8, #bd5be8);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 25px;
    font-weight: bold;
}

.stButton > button:hover {
    border: 1px solid white;
    color: white;
}

[data-testid="stHeader"] {
    background: transparent;
}

</style>
""", unsafe_allow_html=True)

# =========================
# 3. Load Files
# =========================

@st.cache_resource
def load_model():
    return joblib.load("accident_risk_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

artifacts = load_model()
train = load_data()

model = artifacts["model"]
onehot = artifacts["onehot"]
scaler = artifacts["scaler"]

feature_columns = artifacts["feature_columns"]

categorical_columns = [
    "road_type",
    "lighting",
    "weather",
    "time_of_day"
]

# =========================
# 4. Header
# =========================

st.sidebar.title("🚗 RoadGuard AI")
st.sidebar.caption("ROAD SAFETY ANALYTICS")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Risk Prediction",
        "Model Performance",
        "Data Insights",
        "Prediction History",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.success("AI Model Loaded Successfully")

# =========================
# 5. Dashboard
# =========================

if page == "Dashboard":

    st.title("🚗 RoadGuard AI")
    st.subheader("Road Accident Risk Analytics")
    st.write(
        "An intelligent machine learning system "
        "for predicting road accident risk."
    )

    st.divider()

    avg_risk = train["accident_risk"].mean()
    max_risk = train["accident_risk"].max()
    total_records = len(train)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        f"{total_records:,}"
    )

    col2.metric(
        "Average Accident Risk",
        f"{avg_risk:.4f}"
    )

    col3.metric(
        "Maximum Risk",
        f"{max_risk:.4f}"
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Accident Risk Distribution")

        risk_counts = train["accident_risk"].value_counts(
            bins=10
        ).sort_index()

        risk_counts.index = risk_counts.index.astype(str)

        st.bar_chart(risk_counts)

    with right:
        st.subheader("Average Risk by Road Type")

        road_risk = train.groupby(
            "road_type"
        )["accident_risk"].mean()

        st.bar_chart(road_risk)

    st.info(
        "Use the sidebar to explore predictions, "
        "model performance and dataset insights."
    )

# =========================
# 6. Risk Prediction
# =========================

elif page == "Risk Prediction":

    st.title("🎯 Accident Risk Prediction")

    st.write(
        "Enter road and environmental conditions "
        "to estimate accident risk."
    )

    st.divider()

    raw_input = {}

    st.subheader("Road Conditions")

    col1, col2 = st.columns(2)

    for i, col in enumerate(categorical_columns):

        options = list(
            onehot.categories_[i]
        )

        with (col1 if i % 2 == 0 else col2):

            raw_input[col] = st.selectbox(
                col.replace("_", " ").title(),
                options
            )

    st.divider()

    st.subheader("Other Information")

    base_columns = [
        col for col in train.columns
        if col not in ["id", "accident_risk"]
        and col not in categorical_columns
    ]

    col1, col2 = st.columns(2)

    for i, col in enumerate(base_columns):

        with (col1 if i % 2 == 0 else col2):

            if pd.api.types.is_bool_dtype(train[col]):

                options = list(train[col].dropna().unique())

                raw_input[col] = st.selectbox(
                    col.replace("_", " ").title(),
                    options,
                    format_func=lambda x: "Yes" if x else "No"
                )

            elif pd.api.types.is_numeric_dtype(train[col]):

                minimum = float(train[col].min())
                maximum = float(train[col].max())
                default = float(train[col].median())

                if minimum < maximum:

                    raw_value = st.number_input(
                        col.replace("_", " ").title(),
                        min_value=minimum,
                        max_value=maximum,
                        value=default
                    )

                else:
                    raw_value = st.number_input(
                        col.replace("_", " ").title(),
                        value=default
                    )

                if pd.api.types.is_integer_dtype(train[col]):
                    raw_input[col] = int(raw_value)
                else:
                    raw_input[col] = raw_value

            else:

                options = list(train[col].dropna().unique())

                raw_input[col] = st.selectbox(
                    col.replace("_", " ").title(),
                    options
                )

    st.divider()

    if st.button("🚀 Predict Accident Risk"):

        input_df = pd.DataFrame([raw_input])

        encoded_df = onehot.transform(
            input_df[categorical_columns]
        )

        if not isinstance(encoded_df, pd.DataFrame):

            encoded_df = pd.DataFrame(
                encoded_df,
                columns=onehot.get_feature_names_out()
            )

        processed = pd.concat(
            [
                input_df.drop(columns=categorical_columns),
                encoded_df
            ],
            axis=1
        )

        processed = processed.reindex(
            columns=feature_columns
        )

        scaled_input = scaler.transform(processed)

        prediction = float(
            model.predict(scaled_input)[0]
        )

        st.session_state.setdefault(
            "prediction_history", []
        )

        st.session_state["prediction_history"].append({
            "Prediction": prediction
        })

        st.success("Prediction Completed Successfully!")

        st.metric(
            "Predicted Accident Risk",
            f"{prediction:.4f}"
        )

        st.progress(
            float(np.clip(prediction, 0, 1))
        )

        st.caption(
            "The result is an estimated risk score, "
            "not a guarantee that an accident will occur."
        )

# =========================
# 7. Model Performance
# =========================

elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "Evaluation results from the model experiments."
    )

    st.divider()

    col1, col2 = st.columns(2)

    col1.metric(
        "Selected Model",
        "Bagging Regressor"
    )

    col2.metric(
        "Validation RMSE",
        "0.056346"
    )

    st.subheader("Model Information")

    performance_data = pd.DataFrame({
        "Metric": [
            "Validation RMSE"
        ],
        "Value": [
            0.056346
        ]
    })

    st.dataframe(
        performance_data,
        use_container_width=True
    )

    st.info(
        "Bagging Regressor was selected based on "
        "the validation results of the experiments."
    )

# =========================
# 8. Data Insights
# =========================

elif page == "Data Insights":

    st.title("📈 Data Insights")

    st.write(
        "Explore patterns and relationships in the training dataset."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Road Type Distribution")

        st.bar_chart(
            train["road_type"].value_counts()
        )

    with col2:

        st.subheader("Average Risk by Weather")

        weather_risk = train.groupby(
            "weather"
        )["accident_risk"].mean()

        st.bar_chart(weather_risk)

    st.divider()

    st.subheader("Average Risk by Lighting")

    lighting_risk = train.groupby(
        "lighting"
    )["accident_risk"].mean()

    st.bar_chart(lighting_risk)

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        train.head(10),
        use_container_width=True
    )

# =========================
# 9. Prediction History
# =========================

elif page == "Prediction History":

    st.title("🕘 Prediction History")

    history = st.session_state.get(
        "prediction_history", []
    )

    if history:

        history_df = pd.DataFrame(history)

        history_df.index = range(
            1, len(history_df) + 1
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        if st.button("Clear History"):

            st.session_state["prediction_history"] = []

            st.rerun()

    else:

        st.info(
            "No predictions yet. Go to Risk Prediction "
            "to make your first prediction."
        )

# =========================
# 10. About Project
# =========================

elif page == "About Project":

    st.title("ℹ️ About RoadGuard AI")

    st.write(
        """
        RoadGuard AI is a machine learning project
        designed to estimate road accident risk.

        The project uses historical road and environmental
        conditions to predict accident risk scores.
        """
    )

    st.divider()

    st.subheader("Machine Learning Model")

    st.write("Selected Model: Bagging Regressor")

    st.write(
        "Base Estimator: Decision Tree Regressor"
    )

    st.write(
        "Number of Estimators: 50"
    )

    st.write(
        "Maximum Tree Depth: 10"
    )

    st.divider()

    st.subheader("Technologies Used")

    st.write(
        "- Python\n"
        "- Pandas and NumPy\n"
        "- Scikit-learn\n"
        "- Streamlit\n"
        "- Machine Learning"
    )

    st.success(
        "RoadGuard AI - Making Roads Safer with Data."
    )
