import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================================================
# LOAD RESOURCES
# ======================================================
clf_model = joblib.load("models/best_classifier.joblib")
reg_model = joblib.load("models/best_regressor.joblib")
FEATURE_COLUMNS = joblib.load("models/feature_list.joblib")
scaler = joblib.load("models/scaler.joblib")

NUMERIC_FEATURES = list(scaler.feature_names_in_)

LABEL_MAP = {
    0: "Eligible",
    1: "High_Risk",
    2: "Not_Eligible"
}

# ======================================================
# PREPROCESSING PIPELINE
# ======================================================
def preprocess_input(raw):

    df = pd.DataFrame([raw])

    # ---------- Derived Features ----------
    df["total_expenses"] = (
        df["monthly_rent"]
        + df["school_fees"]
        + df["college_fees"]
        + df["travel_expenses"]
        + df["groceries_utilities"]
        + df["other_monthly_expenses"]
    )

    df["debt_to_income"] = df["current_emi_amount"] / df["monthly_salary"].replace(0, np.nan)
    df["debt_to_income"] = df["debt_to_income"].replace([np.inf, -np.inf], np.nan).fillna(0)

    df["expense_to_income"] = df["total_expenses"] / df["monthly_salary"].replace(0, np.nan)
    df["expense_to_income"] = df["expense_to_income"].replace([np.inf, -np.inf], np.nan).fillna(0)

    df["loan_affordability_ratio"] = df["requested_amount"] / df["monthly_salary"].replace(0, np.nan)
    df["loan_affordability_ratio"] = df["loan_affordability_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)

    df["employment_stability"] = (df["years_of_employment"] > 5).astype(int)

    df["income_credit_interaction"] = df["monthly_salary"] * df["credit_score"]

    df["loan_balance_ratio"] = df["requested_amount"] / df["bank_balance"].replace(0, np.nan)
    df["loan_balance_ratio"] = df["loan_balance_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # ---------- SCALE NUMERIC FEATURES EXACTLY AS TRAINED ----------
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0

    df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])

    # ---------- Credit Score Category ----------
    bins = [0, 579, 669, 739, 799, np.inf]
    labels = ["Very Poor", "Fair", "Good", "Very Good", "Excellent"]
    df["credit_score_category"] = pd.cut(df["credit_score"], bins=bins, labels=labels)

    # ---------- Dummy Encoding ----------
    CAT_COLS = [
        'gender', 'marital_status', 'education', 'employment_type',
        'company_type', 'house_type', 'existing_loans', 'emi_scenario',
        'credit_score_category'
    ]

    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)

    # ---------- Add missing columns ----------
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # ---------- Final column alignment ----------
    df = df[FEATURE_COLUMNS]

    return df


# ======================================================
# STREAMLIT UI
# ======================================================

st.set_page_config(page_title="EMI Eligibility System", layout="wide")

# Generic Card Styles (auto adjusts to Light/Dark)
CARD_STYLE = """
    <div style="
        padding:18px 25px;
        border-radius:12px;
        border:1px solid rgba(255,255,255,0.15);
        background:rgba(0,0,0,0.25);
        margin-bottom:20px;">
"""

CARD_END = "</div>"

TITLE_STYLE = """
    <h2 style='text-align:center;'>
        💰 EMI Eligibility & Maximum EMI Prediction System
    </h2>
    <p style='text-align:center; font-size:16px; opacity:0.85;'>
        Enter customer details below to check EMI eligibility & max EMI capacity.
    </p>
    <br>
"""

st.markdown(TITLE_STYLE, unsafe_allow_html=True)


# ------------------------------------------------------
# Card helper
# ------------------------------------------------------
def card(title):
    st.markdown(CARD_STYLE.replace("\n", "") + f"<h4>{title}</h4>", unsafe_allow_html=True)

def end_card():
    st.markdown(CARD_END, unsafe_allow_html=True)


# ------------------------------------------------------
# Personal + Employment Information
# ------------------------------------------------------

col1, col2 = st.columns([1, 1])

with col1:
    card("🧍 Personal Information")
    age = st.number_input("Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["male", "female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    education = st.selectbox(
        "Education", 
        ["High School", "Graduate", "Post Graduate", "Professional"]
    )
    end_card()

    card("💼 Employment Details")
    employment_type = st.selectbox(
        "Employment Type",
        ["Private", "Self-employed", "Government", "Freelancer"]
    )
    years_of_employment = st.number_input("Years of Employment", 0.0, 40.0, 3.0)
    company_type = st.selectbox("Company Type", ["MNC", "Mid-size", "Small", "Startup"])
    end_card()

with col2:
    card("🏠 Living & Family")
    house_type = st.selectbox("House Type", ["Own", "Rented"])
    monthly_rent = st.number_input("Monthly Rent (₹)", 0.0, 100000.0, 10000.0)
    family_size = st.number_input("Family Size", 1, 15, 3)
    dependents = st.number_input("Dependents", 0, 10, 1)
    end_card()

    card("💳 Loan & EMI Details")
    existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
    emi_scenario = st.selectbox(
        "EMI Scenario",
        ["Education EMI", "Home Appliances EMI", "Personal Loan EMI", "Vehicle EMI"]
    )
    current_emi_amount = st.number_input("Current EMI (₹)", 0.0)
    credit_score = st.number_input("Credit Score", 300, 900, 700)
    end_card()


# ------------------------------------------------------
# Financial Details
# ------------------------------------------------------

st.markdown("<h3>📊 Financial Details</h3>", unsafe_allow_html=True)

card("📉 Monthly Financial Breakdown")
col3, col4 = st.columns(2)

with col3:
    school_fees = st.number_input("School Fees (₹)", 0.0)
    college_fees = st.number_input("College Fees (₹)", 0.0)
    travel_expenses = st.number_input("Travel Expenses (₹)", 0.0)
    groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0.0)

with col4:
    other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", 0.0)
    bank_balance = st.number_input("Bank Balance (₹)", 0.0)
    emergency_fund = st.number_input("Emergency Fund (₹)", 0.0)
    monthly_salary = st.number_input("Monthly Salary (₹)", 0.0, 1000000.0, 50000.0)

end_card()


# ------------------------------------------------------
# Loan Request
# ------------------------------------------------------

card("📑 Loan Request Information")
col5, col6 = st.columns(2)

with col5:
    requested_amount = st.number_input("Requested Loan Amount (₹)", 0.0)

with col6:
    requested_tenure = st.number_input("Requested Tenure (months)", 1, 240, 24)

end_card()


# ------------------------------------------------------
# Predict Button
# ------------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

center = st.columns([3, 1, 3])[1]
with center:
    predict_btn = st.button("🔍 Predict Eligibility", use_container_width=True)


# ------------------------------------------------------
# Prediction Result Card
# ------------------------------------------------------

if predict_btn:

    raw = {
        "age": age, "gender": gender, "marital_status": marital_status, "education": education,
        "monthly_salary": monthly_salary, "employment_type": employment_type,
        "years_of_employment": years_of_employment, "company_type": company_type,
        "house_type": house_type, "monthly_rent": monthly_rent, "family_size": family_size,
        "dependents": dependents, "school_fees": school_fees, "college_fees": college_fees,
        "travel_expenses": travel_expenses, "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses, "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount, "credit_score": credit_score,
        "bank_balance": bank_balance, "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario, "requested_amount": requested_amount,
        "requested_tenure": requested_tenure
    }

    processed = preprocess_input(raw)
    pred_class = int(clf_model.predict(processed)[0])
    pred_emi = float(reg_model.predict(processed)[0])
  
    st.subheader("📘 Prediction Summary")
    st.success(f"🟢 Eligibility Result: **{LABEL_MAP[pred_class]}**")
    st.info(f"💰 Predicted Maximum EMI: **₹{pred_emi:,.2f}**")
    st.caption("⚠️ Actual EMI offer may vary based on bank policies.")

    st.markdown("</div>", unsafe_allow_html=True)
