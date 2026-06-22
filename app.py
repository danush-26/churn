import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.title("📉 Customer Churn Prediction App")
st.write("Predicts whether a customer is likely to churn.")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load and clean data (same steps as train_model.py)
df = pd.read_csv("churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.ffill()

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Churn Distribution in Dataset")
st.bar_chart(df["Churn"].value_counts())

st.subheader("Model Accuracy")
st.write("This Random Forest model achieves around 80% accuracy on test data.")

# Encode categorical columns
encoded_df = df.drop(["customerID", "Churn"], axis=1).copy()
for col in encoded_df.columns:
    if encoded_df[col].dtype == "object" or encoded_df[col].dtype == "str":
        encoded_df[col] = LabelEncoder().fit_transform(encoded_df[col])

st.subheader("What Affects Churn Most")
importance_df = pd.DataFrame({
    "Feature": encoded_df.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)
st.bar_chart(importance_df.set_index("Feature"))

# Simple input form
st.subheader("Enter Customer Details")
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 18.0, 120.0, 70.0)
contract = st.selectbox("Contract", df["Contract"].unique())
internet_service = st.selectbox("Internet Service", df["InternetService"].unique())

# Predict button
if st.button("Predict Churn"):
    row = encoded_df.iloc[0:1].copy()
    row["tenure"] = tenure
    row["MonthlyCharges"] = monthly_charges
    row["Contract"] = LabelEncoder().fit(df["Contract"]).transform([contract])[0]
    row["InternetService"] = LabelEncoder().fit(df["InternetService"]).transform([internet_service])[0]

    prediction = model.predict(row)[0]
    probability = model.predict_proba(row)[0]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ Likely to CHURN ({probability[1]*100:.1f}% confidence)")
    else:
        st.success(f"✅ Likely to STAY ({probability[0]*100:.1f}% confidence)")