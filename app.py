import streamlit as st
import random
import time
import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

try:
    from stock_analysis import StockAnalysisSystem
    STOCK_SYSTEM_AVAILABLE = True
except ImportError:
    STOCK_SYSTEM_AVAILABLE = False


st.set_page_config(page_title="AI Credit Scoring (Thin-File)", page_icon="💳", layout="wide")


# Load trained ML models
@st.cache_resource
def load_ml_model():
    """Load the trained RandomForest model for loan eligibility prediction."""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model_info.pkl', 'rb') as f:
            info = pickle.load(f)
        return model, info, None
    except FileNotFoundError:
        return None, None, (
            "Model files not found. Run `python train_model.py` to generate `model.pkl` and `model_info.pkl`."
        )
    except ModuleNotFoundError as e:
        return None, None, (
            f"Missing dependency while loading model: {e}. "
            "Install required packages in the active environment, then restart Streamlit."
        )
    except Exception as e:
        return None, None, f"Failed to load model artifacts: {e}"


@st.cache_resource
def load_stock_system():
    """Load stock analysis system with ML model."""
    if not STOCK_SYSTEM_AVAILABLE:
        return None
    try:
        if os.path.exists('stock_data.csv'):
            return StockAnalysisSystem('stock_data.csv')
    except Exception as e:
        st.error(f"Stock system error: {str(e)}")
    return None


ml_model, ml_info, ml_model_error = load_ml_model()
stock_system = load_stock_system()

if ml_model is None:
    st.warning("Loan predictor model is unavailable.")
    if ml_model_error:
        st.error(ml_model_error)


def make_ml_prediction(
    model,
    num_cols,
    cat_cols,
    no_of_dependents: int,
    education: str,
    self_employed: str,
    income_annum: float,
    loan_amount: float,
    loan_term: int,
    residential_assets: float,
    commercial_assets: float,
    luxury_assets: float,
    bank_assets: float,
):
    """Make loan eligibility prediction using trained ML model."""
    if model is None:
        return None, None, "Model not available"

    try:
        total_assets = residential_assets + commercial_assets + luxury_assets + bank_assets
        loan_income_ratio = safe_ratio(loan_amount, income_annum)
        asset_coverage_ratio = safe_ratio(total_assets, loan_amount)

        # Create feature input dataframe
        input_data = pd.DataFrame({
            ' no_of_dependents': [no_of_dependents],
            ' education': [education],
            ' self_employed': [self_employed],
            ' income_annum': [income_annum],
            ' loan_amount': [loan_amount],
            ' loan_term': [loan_term],
            ' residential_assets_value': [residential_assets],
            ' commercial_assets_value': [commercial_assets],
            ' luxury_assets_value': [luxury_assets],
            ' bank_asset_value': [bank_assets],
            'loan_income_ratio': [loan_income_ratio],
            'total_assets': [total_assets],
            'asset_coverage_ratio': [asset_coverage_ratio],
        })

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        pred_label = "Approved" if prediction == 1 else "Rejected"
        pred_prob = probability[1] if prediction == 1 else probability[0]

        return pred_label, pred_prob, None
    except Exception as e:
        return None, None, str(e)


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return 0.0 if denominator is 0 to avoid division errors."""
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def apply_underwriting_rules(
    income_annum: float,
    loan_amount: float,
    loan_term: int,
    total_assets: float,
):
    """Policy-level guardrails applied on top of ML output."""
    reasons = []

    loan_income_ratio = safe_ratio(loan_amount, income_annum)
    asset_coverage_ratio = safe_ratio(total_assets, loan_amount)
    monthly_income = income_annum / 12.0
    monthly_rate = 0.11 / 12.0
    n = max(1, int(loan_term))
    emi = loan_amount * monthly_rate * ((1 + monthly_rate) ** n) / (((1 + monthly_rate) ** n) - 1)
    emi_income_ratio = safe_ratio(emi, monthly_income)

    hard_reject = False

    if loan_income_ratio > 12:
        hard_reject = True
        reasons.append("Loan amount is more than 12x annual income")
    if emi_income_ratio > 0.65:
        hard_reject = True
        reasons.append("Estimated EMI exceeds 65% of monthly income")
    caution_flags = []
    if loan_income_ratio > 5:
        caution_flags.append("High loan-to-income ratio")
    if emi_income_ratio > 0.45:
        caution_flags.append("EMI burden is elevated")
    if asset_coverage_ratio < 0.35:
        caution_flags.append("Asset coverage is weak")

    return {
        "hard_reject": hard_reject,
        "reasons": reasons,
        "caution_flags": caution_flags,
        "loan_income_ratio": loan_income_ratio,
        "asset_coverage_ratio": asset_coverage_ratio,
        "emi_income_ratio": emi_income_ratio,
    }


def compute_score_and_breakdown(
    income: float,
    expenses: float,
    savings: float,
    discretionary_pct: float,
    investment: float,
):
    """Compute score, feature ratios, and explainable point breakdown."""
    savings_ratio = safe_ratio(savings, income)
    expense_ratio = safe_ratio(expenses, income)
    discretionary_ratio = discretionary_pct / 100.0
    investment_ratio = safe_ratio(investment, income)

    base_score = 600
    breakdown = []

    # Positive contributions
    savings_points = max(0, min(120, int(savings_ratio * 200)))
    investment_points = max(0, min(80, int(investment_ratio * 150)))

    # Negative contributions
    high_expense_points = max(0, min(120, int(max(0.0, expense_ratio - 0.5) * 240)))
    high_discretionary_points = max(0, min(100, int(max(0.0, discretionary_ratio - 0.3) * 220)))

    score_raw = (
        base_score
        + savings_points
        + investment_points
        - high_expense_points
        - high_discretionary_points
    )
    final_score = max(300, min(850, int(score_raw)))

    breakdown.append(("Base score", f"+{base_score}"))
    breakdown.append(("Savings habit", f"+{savings_points}"))
    breakdown.append(("Investment behavior", f"+{investment_points}"))
    breakdown.append(("High expenses", f"-{high_expense_points}"))
    breakdown.append(("High discretionary spending", f"-{high_discretionary_points}"))

    features = {
        "savings_ratio": savings_ratio,
        "expense_ratio": expense_ratio,
        "discretionary_ratio": discretionary_ratio,
        "investment_ratio": investment_ratio,
    }

    return final_score, features, breakdown


def generate_suggestions(features: dict, investment: float):
    suggestions = []

    if features["savings_ratio"] < 0.2:
        suggestions.append("Increase savings rate to at least 20% of monthly income.")

    if features["discretionary_ratio"] > 0.4:
        suggestions.append("Reduce unnecessary discretionary spending (target <= 40%).")

    if investment <= 0 or features["investment_ratio"] < 0.1:
        suggestions.append("Start a SIP/investment habit to improve financial stability.")

    if not suggestions:
        suggestions.append("Great profile. Keep maintaining disciplined spending and savings behavior.")

    return suggestions


def analyze_investment_profile(income: float, expenses: float, savings: float, discretionary_pct: float) -> dict:
    """Analyze financial profile to recommend investment capacity and behavior."""
    
    # Calculate key ratios
    monthly_income = income
    monthly_expenses = expenses
    monthly_savings = savings
    discretionary_ratio = discretionary_pct / 100.0
    
    expense_ratio = safe_ratio(expenses, income)
    savings_ratio = safe_ratio(savings, income)
    available_for_investment = monthly_income - monthly_expenses - monthly_savings
    
    # Investment capacity based on profile
    investment_capacity = 0
    investment_recommendation = ""
    investment_level = "Low"
    
    # Strong profile: high savings, low expenses
    if savings_ratio >= 0.25 and expense_ratio <= 0.5:
        investment_capacity = max(0, available_for_investment * 0.7)  # 70% of leftover
        investment_recommendation = "Excellent financial position. You can comfortably invest 70% of your available surplus."
        investment_level = "High"
    
    # Moderate profile: medium savings, medium expenses
    elif savings_ratio >= 0.15 and expense_ratio <= 0.65:
        investment_capacity = max(0, available_for_investment * 0.5)  # 50% of leftover
        investment_recommendation = "Good financial health. Consider investing 50% of your monthly surplus for wealth building."
        investment_level = "Medium"
    
    # Lower profile: need to improve before heavy investing
    else:
        investment_capacity = max(0, available_for_investment * 0.25)  # 25% of leftover
        investment_recommendation = "Focus on increasing savings first. Once savings ratio reaches 20%, increase investments to 5-10% of income."
        investment_level = "Low"
    
    return {
        "capacity": investment_capacity,
        "recommendation": investment_recommendation,
        "level": investment_level,
        "savings_ratio": savings_ratio,
        "expense_ratio": expense_ratio,
        "available": available_for_investment,
    }


def answer_query(
    query: str,
    score: int,
    savings_ratio: float,
    expense_ratio: float,
    discretionary_ratio: float,
    investment: float,
    income: float,
    expenses: float,
    savings: float,
    discretionary_pct: float,
) -> str:
    """Rule-based assistant that responds only from current financial profile data."""
    if not query or not query.strip():
        return "Please ask a question about your credit profile."

    q = query.strip().lower()

    if "why" in q and "score" in q and "low" in q:
        reasons = []
        if savings_ratio < 0.2:
            reasons.append("your savings ratio is below the healthy 20% level")
        if expense_ratio > 0.6:
            reasons.append("your expense ratio is high")
        if discretionary_ratio > 0.4:
            reasons.append("your discretionary spending is elevated")

        if reasons:
            return f"Your score is lower mainly because {', '.join(reasons)}."
        return "Your score is not low due to major red flags; smaller spending/savings changes can still improve it."

    if "how" in q and "improve" in q and "score" in q:
        actions = [
            "Increase monthly savings toward at least 20% of income",
            "Reduce discretionary spending below 40%",
            "Start or increase SIP/investment contributions",
        ]
        return "To improve your score: " + "; ".join(actions) + "."

    if "what if" in q and "reduce" in q and "expense" in q:
        new_expenses = expenses * 0.8
        new_score, _, _ = compute_score_and_breakdown(
            income=income,
            expenses=new_expenses,
            savings=savings,
            discretionary_pct=discretionary_pct,
            investment=investment,
        )
        return f"If you reduce expenses by 20%, your estimated score becomes {new_score} (from {score})."

    if "what if" in q and "increase" in q and "saving" in q:
        new_savings = savings * 1.2
        new_score, _, _ = compute_score_and_breakdown(
            income=income,
            expenses=expenses,
            savings=new_savings,
            discretionary_pct=discretionary_pct,
            investment=investment,
        )
        return f"If you increase savings by 20%, your estimated score becomes {new_score} (from {score})."

    if "risk" in q:
        if score < 500:
            risk = "High Risk"
        elif score <= 700:
            risk = "Moderate"
        else:
            risk = "Healthy"
        return f"Based on your current score of {score}, your risk level is: {risk}."

    return "I can only answer questions about your credit profile."


st.title("AI-Based Credit Scoring for Thin-File Users")
st.caption("We help you improve your credit — not just measure it.")

st.markdown(
    """
    <style>
    .section-card {
        background: #ffffff;
        border: 1px solid #e8edf7;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }
    .score-card {
        border-radius: 18px;
        padding: 22px;
        text-align: center;
        color: #ffffff;
        background: linear-gradient(135deg, #0f1f4b 0%, #1f4ba8 55%, #2e7dd7 100%);
        box-shadow: 0 10px 28px rgba(15, 31, 75, 0.28);
        margin: 6px 0 12px 0;
    }
    .score-label {
        font-size: 16px;
        opacity: 0.9;
        margin-bottom: 8px;
    }
    .score-value {
        font-size: 64px;
        font-weight: 800;
        line-height: 1;
        margin: 8px 0;
    }
    .score-status {
        font-size: 14px;
        opacity: 0.95;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        background: linear-gradient(90deg, #0f4aa0 0%, #1d72d8 100%);
        color: white;
        font-weight: 700;
        padding: 0.65rem 0.9rem;
    }
    .stButton > button:hover {
        filter: brightness(1.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["💳 Credit Score Analyzer", "🏦 Loan Eligibility Predictor", "📈 Stock Price Prediction"])

# ============================================================================
# TAB 1: CREDIT SCORE ANALYZER (Original)
# ============================================================================
with tab1:
    left_col, right_col = st.columns([1, 1.3], gap="large")

    with left_col:
        with st.container():
            st.subheader("User Inputs")
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            monthly_income = st.number_input("Monthly Income", min_value=0.0, value=30000.0, step=1000.0)
            monthly_expenses = st.number_input("Monthly Expenses", min_value=0.0, value=15000.0, step=1000.0)
            savings = st.number_input("Savings", min_value=0.0, value=5000.0, step=500.0)
            discretionary_pct = st.slider("Discretionary Spending (%)", min_value=0, max_value=100, value=30)
            investment_amount = st.number_input(
                "Investment Amount (Optional)",
                min_value=0.0,
                value=0.0,
                step=500.0,
                help="Leave as 0 if not investing yet.",
            )
            enable_live_monitoring = st.toggle("Enable Live Monitoring", value=False)
            st.markdown("</div>", unsafe_allow_html=True)

    if "connected_investment" not in st.session_state:
        st.session_state.connected_investment = False
    if "investment_profile" not in st.session_state:
        st.session_state.investment_profile = None

    if "live_expenses" not in st.session_state:
        st.session_state.live_expenses = monthly_expenses
    if "live_savings" not in st.session_state:
        st.session_state.live_savings = savings
    if "prev_live_expenses" not in st.session_state:
        st.session_state.prev_live_expenses = monthly_expenses
    if "prev_live_savings" not in st.session_state:
        st.session_state.prev_live_savings = savings

    with left_col:
        with st.container():
            if st.button("📊 Analyze Investment Profile"):
                # Analyze based on current financial profile
                investment_profile = analyze_investment_profile(
                    income=monthly_income,
                    expenses=monthly_expenses,
                    savings=savings,
                    discretionary_pct=discretionary_pct,
                )
                st.session_state.investment_profile = investment_profile
                st.session_state.connected_investment = True
                st.success("✅ Investment profile analyzed from your financial data")

    # Use investment capacity from profile if connected, otherwise use manual input
    simulated_investment = investment_amount
    investment_analysis_available = False
    
    if st.session_state.connected_investment and st.session_state.investment_profile:
        profile = st.session_state.investment_profile
        # If user has no manual investment, use recommended capacity
        if investment_amount == 0:
            simulated_investment = profile['capacity']
        investment_analysis_available = True

    live_alerts = []
    active_expenses = monthly_expenses
    active_savings = savings

    if enable_live_monitoring:
        st.session_state.prev_live_expenses = st.session_state.live_expenses
        st.session_state.prev_live_savings = st.session_state.live_savings

        expense_jitter = random.uniform(-0.03, 0.05)
        savings_jitter = random.uniform(-0.02, 0.06)

        st.session_state.live_expenses = max(0.0, monthly_expenses * (1 + expense_jitter))
        st.session_state.live_savings = max(0.0, savings * (1 + savings_jitter))

        if st.session_state.live_expenses > st.session_state.prev_live_expenses:
            live_alerts.append("Spending increasing")
        if st.session_state.live_savings > st.session_state.prev_live_savings:
            live_alerts.append("Savings improving")

        active_expenses = st.session_state.live_expenses
        active_savings = st.session_state.live_savings
    else:
        st.session_state.live_expenses = monthly_expenses
        st.session_state.live_savings = savings
        st.session_state.prev_live_expenses = monthly_expenses
        st.session_state.prev_live_savings = savings

    score, features, breakdown = compute_score_and_breakdown(
        income=monthly_income,
        expenses=active_expenses,
        savings=active_savings,
        discretionary_pct=float(discretionary_pct),
        investment=simulated_investment,
    )

    with right_col:
        with st.container():
            if score < 500:
                score_color = "#ff6b6b"
                score_band = "Needs Attention"
            elif score <= 700:
                score_color = "#ffd166"
                score_band = "Building"
            else:
                score_color = "#7be495"
                score_band = "Strong"

            st.markdown(
                f"""
                <div class='score-card'>
                    <div class='score-label'>Estimated Credit Score</div>
                    <div class='score-value' style='color:{score_color};'>{score}</div>
                    <div class='score-status'>Band: {score_band} | Range: 300 - 850</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            st.subheader("Smart Q&A Assistant")
            user_query = st.text_input("Ask about your credit profile")
            if user_query is not None:
                qa_response = answer_query(
                    query=user_query,
                    score=score,
                    savings_ratio=features["savings_ratio"],
                    expense_ratio=features["expense_ratio"],
                    discretionary_ratio=features["discretionary_ratio"],
                    investment=simulated_investment,
                    income=monthly_income,
                    expenses=active_expenses,
                    savings=active_savings,
                    discretionary_pct=float(discretionary_pct),
                )
                if user_query.strip():
                    st.info(qa_response)

            if investment_analysis_available:
                st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
                st.subheader("💡 Investment Profile Analysis")
                profile = st.session_state.investment_profile
                
                # Color coding for investment level
                if profile['level'] == "High":
                    level_color = "#7be495"
                    level_emoji = "🟢"
                elif profile['level'] == "Medium":
                    level_color = "#ffd166"
                    level_emoji = "🟡"
                else:
                    level_color = "#ff9999"
                    level_emoji = "🟠"
                
                st.markdown(
                    f"""
                    <div style='border-radius: 12px; padding: 14px; 
                                background: linear-gradient(135deg, {level_color}22 0%, {level_color}11 100%);
                                border-left: 4px solid {level_color};'>
                        <div style='font-size: 14px; font-weight: 700; color: #333;'>
                            {level_emoji} Investment Level: {profile['level']}
                        </div>
                        <div style='font-size: 13px; color: #555; margin-top: 8px;'>
                            {profile['recommendation']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                col_inv1, col_inv2 = st.columns(2)
                with col_inv1:
                    st.metric(
                        "Recommended Monthly Investment",
                        f"₹{profile['capacity']:,.0f}",
                        delta=f"{(profile['capacity'] / monthly_income * 100):.1f}% of income"
                    )
                with col_inv2:
                    st.metric(
                        "Available After Savings",
                        f"₹{profile['available']:,.0f}",
                    )

            if enable_live_monitoring:
                if score < 500:
                    st.error("Financial Health: High Risk")
                elif score <= 700:
                    st.warning("Financial Health: Moderate")
                else:
                    st.success("Financial Health: Healthy")

                le_col1, le_col2 = st.columns(2)
                with le_col1:
                    st.metric(
                        "Live Expenses",
                        f"{active_expenses:,.0f}",
                        delta=f"{active_expenses - st.session_state.prev_live_expenses:,.0f}",
                    )
                with le_col2:
                    st.metric(
                        "Live Savings",
                        f"{active_savings:,.0f}",
                        delta=f"{active_savings - st.session_state.prev_live_savings:,.0f}",
                    )

                for alert in live_alerts:
                    if alert == "Spending increasing":
                        st.error(alert)
                    elif alert == "Savings improving":
                        st.success(alert)

        with st.container():
            st.subheader("📊 Financial Insights")
            for reason, points in breakdown:
                value = int(points.replace("+", "")) if points.startswith("+") else -int(points.replace("-", ""))
                message = f"{reason}: {points}"
                if reason == "Base score":
                    st.warning(message)
                elif value >= 0:
                    st.success(message)
                else:
                    st.error(message)

        with st.container():
            st.subheader("🚀 Improvement Plan")
            for item in generate_suggestions(features, simulated_investment):
                st.info(item)

        with st.container():
            st.subheader("⚡ What-If Simulator")
            st.caption("Adjust spending and savings assumptions to see potential score improvement.")

            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                expense_reduction_pct = st.slider("Expenses Reduction (%)", 0, 50, 10)
            with sim_col2:
                savings_increase_pct = st.slider("Savings Increase (%)", 0, 100, 20)

            sim_expenses = monthly_expenses * (1 - expense_reduction_pct / 100.0)
            sim_savings = savings * (1 + savings_increase_pct / 100.0)

            sim_score, sim_features, _ = compute_score_and_breakdown(
                income=monthly_income,
                expenses=sim_expenses,
                savings=sim_savings,
                discretionary_pct=float(discretionary_pct),
                investment=simulated_investment,
            )

            score_delta = sim_score - score
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Current Score", score)
            with m2:
                st.metric("Improved Score", sim_score, delta=score_delta)

            if score_delta > 0:
                st.success(f"Potential Improvement: +{score_delta} points")
            elif score_delta < 0:
                st.warning(f"Potential Change: {score_delta} points")
            else:
                st.info("No score change under current simulation inputs.")

    if enable_live_monitoring:
        time.sleep(3)
        st.rerun()

# ============================================================================
# TAB 2: LOAN ELIGIBILITY PREDICTOR (ML Model)
# ============================================================================
with tab2:
    st.subheader("🤖 AI-Powered Loan Eligibility Prediction")
    st.caption("Our ML model analyzes your financial profile and predicts loan approval likelihood.")

    if ml_model is None:
        st.error("⚠️ ML model not available. Please run `python train_model.py` first to train the model.")
    else:
        pred_left, pred_right = st.columns([1, 1], gap="large")

        # ---- LEFT COLUMN: Input Features ----
        with pred_left:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("📋 Your Financial Profile")

            col1, col2 = st.columns(2)
            with col1:
                no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
                education = st.selectbox("Education Level", ["Graduate", "Not Graduate"], index=0)
                self_employed = st.selectbox("Self Employed?", ["No", "Yes"], index=0)

            with col2:
                income_annum = st.number_input("Annual Income (₹)", min_value=50000.0, max_value=500000000.0, value=360000.0, step=10000.0)
                loan_amount = st.number_input("Loan Amount (₹)", min_value=10000.0, max_value=1000000000.0, value=150000.0, step=10000.0)
                loan_term = st.number_input("Loan Term (months)", min_value=6, max_value=480, value=36, step=6)

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            st.subheader("💰 Credit & Assets")

            col3, col4 = st.columns(2)
            with col3:
                residential_assets = st.number_input("Residential Assets (₹)", min_value=0.0, max_value=2000000000.0, value=150000.0, step=10000.0)
                commercial_assets = st.number_input("Commercial Assets (₹)", min_value=0.0, max_value=2000000000.0, value=0.0, step=10000.0)

            with col4:
                luxury_assets = st.number_input("Luxury Assets (₹)", min_value=0.0, max_value=2000000000.0, value=0.0, step=5000.0)
                bank_assets = st.number_input("Bank Assets (₹)", min_value=0.0, max_value=2000000000.0, value=50000.0, step=5000.0)

            st.markdown("</div>", unsafe_allow_html=True)

        # ---- RIGHT COLUMN: Prediction & Explanation ----
        with pred_right:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("🎯 Prediction Result")

            # Make prediction
            pred_label, pred_prob, error = make_ml_prediction(
                ml_model,
                ml_info['num_cols'],
                ml_info['cat_cols'],
                no_of_dependents=no_of_dependents,
                education=education,
                self_employed=self_employed,
                income_annum=income_annum,
                loan_amount=loan_amount,
                loan_term=loan_term,
                residential_assets=residential_assets,
                commercial_assets=commercial_assets,
                luxury_assets=luxury_assets,
                bank_assets=bank_assets,
            )

            total_assets = residential_assets + commercial_assets + luxury_assets + bank_assets
            policy = apply_underwriting_rules(
                income_annum=income_annum,
                loan_amount=loan_amount,
                loan_term=loan_term,
                total_assets=total_assets,
            )

            if error:
                st.error(f"Prediction error: {error}")
            elif pred_label:
                if policy["hard_reject"]:
                    pred_label = "Rejected"
                    pred_prob = max(pred_prob, 0.80)
                elif pred_label == "Approved" and len(policy["caution_flags"]) >= 2:
                    pred_prob = min(pred_prob, 0.62)

                # Display prediction card
                if pred_label == "Approved":
                    pred_color = "#7be495"
                    pred_icon = "✅"
                else:
                    pred_color = "#ff6b6b"
                    pred_icon = "❌"

                st.markdown(
                    f"""
                    <div style='border-radius: 14px; padding: 20px; text-align: center; 
                                background: linear-gradient(135deg, {pred_color}22 0%, {pred_color}11 100%);
                                border: 2px solid {pred_color};'>
                        <div style='font-size: 14px; color: #555;'>Model Prediction</div>
                        <div style='font-size: 48px; margin: 10px 0;'>{pred_icon}</div>
                        <div style='font-size: 32px; font-weight: 700; color: {pred_color};'>{pred_label}</div>
                        <div style='font-size: 14px; color: #777; margin-top: 8px;'>
                            Confidence: {pred_prob * 100:.1f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

                # Show explainability
                st.subheader("📊 Key Factors")
                
                debt_to_income = safe_ratio(loan_amount / 12, income_annum / 12)
                if policy["hard_reject"]:
                    for reason in policy["reasons"]:
                        st.error(f"⚠️ Policy reject: {reason}")
                elif policy["caution_flags"]:
                    for flag in policy["caution_flags"]:
                        st.warning(f"⚠️ Risk flag: {flag}")
                
                if pred_label == "Approved":
                    st.info(f"💼 Solid annual income (₹{income_annum:,.0f}) supports loan approval")
                    if debt_to_income < 0.3:
                        st.success(f"✅ Low debt-to-income ratio ({debt_to_income:.2%}) - Excellent!")
                else:
                    if debt_to_income > 0.5:
                        st.error(f"⚠️ High debt-to-income ratio ({debt_to_income:.2%}) - needs improvement")

                st.markdown("</div>", unsafe_allow_html=True)

                # Suggestions
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                st.subheader("🚀 Recommendations to Improve Chances")

                suggestions = []
                if debt_to_income > 0.4:
                    suggestions.append("Reduce loan amount relative to income")
                if no_of_dependents > 5:
                    suggestions.append("Consider future income stability with multiple dependents")
                if income_annum < 2000000:
                    suggestions.append("Increase annual income or explore co-applicant options")

                if not suggestions:
                    st.success("Your profile looks strong! Your loan application has good chances.")
                else:
                    for i, sugg in enumerate(suggestions, 1):
                        st.info(f"{i}. {sugg}")

                st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# TAB 3: STOCK PRICE PREDICTION & ANALYSIS
# ============================================================================
with tab3:
    if not STOCK_SYSTEM_AVAILABLE or stock_system is None:
        st.error("📉 Stock analysis system not available. Please ensure stock_data.csv exists.")
    else:
        st.subheader("📈 AI-Powered Stock Analysis & Price Prediction")
        st.markdown("Enter ANY stock ticker to get intelligent investment recommendations using live market data")
        
        # Input section
        col1, col2, col3 = st.columns(3, gap="medium")
        
        with col1:
            stock_ticker = st.text_input(
                "Stock Ticker Symbol",
                value="AAPL",
                placeholder="Enter ticker (e.g., AAPL, GOOGL, MSFT, TCS.NS)",
                help="Enter stock ticker symbol. Examples: AAPL, GOOGL, MSFT, TSLA, TCS.NS, INFY.NS"
            ).upper().strip()
        
        with col2:
            investment_horizon = st.slider(
                "Investment Horizon (days)",
                min_value=30,
                max_value=180,
                value=60,
                step=15,
                help="How long you plan to hold the stock"
            )
        
        with col3:
            risk_level = st.radio(
                "Risk Tolerance",
                ["Low", "Medium", "High"],
                horizontal=True,
                help="Your investment risk appetite"
            ).lower()
        
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        # Analysis section
        if st.button("🔍 Analyze Stock", use_container_width=True):
            if not stock_ticker:
                st.error("Please enter a stock ticker symbol")
            else:
                with st.spinner(f"Analyzing {stock_ticker}..."):
                    try:
                        # Get prediction using live data
                        recommendation = stock_system.generate_recommendation_live(
                            stock_ticker, 
                            investment_horizon=investment_horizon, 
                            risk_level=risk_level
                        )
                        
                        if recommendation.get('status') == 'FAILED':
                            st.error(f"❌ Analysis failed: {recommendation.get('error')}")
                        else:
                            # Display recommendation cards
                            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
                            
                            pred_col1, pred_col2, pred_col3 = st.columns(3, gap="medium")
                            
                            with pred_col1:
                                st.markdown(
                                    f"""
                                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                                padding: 20px; border-radius: 12px; color: white;'>
                                        <div style='font-size: 12px; opacity: 0.9;'>Current Stock Price</div>
                                        <div style='font-size: 28px; font-weight: 700; margin: 10px 0;'>{recommendation.get('current_price', 'N/A')}</div>
                                        <div style='font-size: 11px; opacity: 0.85;'>Latest: {recommendation.get('ticker', stock_ticker)}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            
                            with pred_col2:
                                trend = recommendation.get('trend', 'Sideways')
                                trend_icon = "📈" if trend == "Uptrend" else "📉" if trend == "Downtrend" else "➡️"
                                trend_color = "#10b981" if trend == "Uptrend" else "#ef4444" if trend == "Downtrend" else "#f59e0b"
                                st.markdown(
                                    f"""
                                    <div style='background: linear-gradient(135deg, {trend_color}20 0%, {trend_color}10 100%); 
                                                border: 2px solid {trend_color}; padding: 20px; border-radius: 12px;'>
                                        <div style='font-size: 12px; color: #666;'>Current Trend</div>
                                        <div style='font-size: 28px; margin: 10px 0;'>{trend_icon}</div>
                                        <div style='font-size: 18px; font-weight: 700; color: {trend_color};'>{trend}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            
                            with pred_col3:
                                future_forecast = recommendation.get('future_forecast', 'Unknown')
                                forecast_strength = recommendation.get('forecast_strength', '0%')
                                future_icon = "📈" if "Uptrend" in future_forecast else "📉" if "Downtrend" in future_forecast else "↔️"
                                future_color = "#10b981" if "Uptrend" in future_forecast else "#ef4444" if "Downtrend" in future_forecast else "#f59e0b"
                                st.markdown(
                                    f"""
                                    <div style='background: linear-gradient(135deg, {future_color}20 0%, {future_color}10 100%); 
                                                border: 2px solid {future_color}; padding: 20px; border-radius: 12px;'>
                                        <div style='font-size: 12px; color: #666;'>{investment_horizon}-Day Outlook</div>
                                        <div style='font-size: 28px; margin: 10px 0;'>{future_icon}</div>
                                        <div style='font-size: 12px; font-weight: 700; color: {future_color};'>{forecast_strength}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            
                            # Investment recommendation
                            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                            
                            rec_label = recommendation.get('recommendation', 'HOLD')
                            rec_reason = recommendation.get('reason', '')
                            
                            # Extract decision from label
                            if "BUY" in rec_label:
                                rec_color = "#10b981"
                            elif "AVOID" in rec_label or "HIGH RISK" in rec_label:
                                rec_color = "#ef4444"
                            else:
                                rec_color = "#f59e0b"
                            
                            st.markdown(
                                f"""
                                <div style='background: linear-gradient(135deg, {rec_color}25 0%, {rec_color}10 100%); 
                                            border-left: 5px solid {rec_color}; padding: 20px; border-radius: 8px;'>
                                    <div style='font-size: 14px; color: #666; margin-bottom: 8px;'>AI Investment Recommendation</div>
                                    <div style='font-size: 26px; font-weight: 700; color: {rec_color}; margin-bottom: 12px;'>{rec_label}</div>
                                    <div style='font-size: 13px; color: #555; line-height: 1.6;'>
                                        <strong>Analysis:</strong> {rec_reason}<br>
                                        <strong>Ticker:</strong> {recommendation.get('ticker', stock_ticker)}<br>
                                        <strong>Volatility:</strong> {recommendation.get('volatility', 'Unknown')}<br>
                                        <strong>Momentum:</strong> {recommendation.get('momentum', 'Unknown')}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Show additional details
                            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                            st.subheader("📊 Detailed Analysis")
                            
                            detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4, gap="medium")
                            
                            with detail_col1:
                                st.metric("Predicted Range", recommendation.get('predicted_range', 'N/A'))
                            
                            with detail_col2:
                                st.metric("Price Margin", recommendation.get('price_margin', 'N/A'))
                            
                            with detail_col3:
                                st.metric("Trend Confidence", recommendation.get('trend_confidence', 'N/A'))
                            
                            with detail_col4:
                                st.metric("Forecast Strength", recommendation.get('forecast_strength', 'N/A'))
                    
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        import traceback
                        st.write(traceback.format_exc())

        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        st.subheader("💡 Real-Time 'Best to Buy' Suggestions")
        st.markdown("We monitor top market movers and find the best buy currently available using real-time internet data.")
        
        if st.button("🚀 Find Best Stocks to Buy Now", use_container_width=True):
            with st.spinner("Scraping live internet data for top trending stocks... This skips our internal model completely."):
                try:
                    import urllib.request
                    import json
                    import yfinance as yf
                    
                    # 1. Fetch real-time trending tickers from Yahoo Finance API
                    req = urllib.request.Request('https://query2.finance.yahoo.com/v1/finance/trending/US?count=15', headers={'User-Agent': 'Mozilla/5.0'})
                    res = urllib.request.urlopen(req)
                    data = json.loads(res.read())
                    trending_tickers = [quote['symbol'] for quote in data['finance']['result'][0]['quotes']]
                    
                    st.info(f"🔍 Found {len(trending_tickers)} trending tickers online right now: {', '.join(trending_tickers)}")
                    
                    # 2. Check actual Wall Street analyst consensus for each via yfinance
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, t in enumerate(trending_tickers):
                        try:
                            # Skip complex tickers or get info
                            info = yf.Ticker(t).info
                            rec = info.get('recommendationKey', 'none').replace('_', ' ').title()
                            
                            if rec.lower() in ['buy', 'strong buy']:
                                current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
                                target_price = info.get('targetMeanPrice', 'N/A')
                                
                                results.append({
                                    'ticker': t,
                                    'rec': rec,
                                    'price': current_price,
                                    'target': target_price,
                                    'name': info.get('shortName', t)
                                })
                        except Exception:
                            pass
                        
                        progress_bar.progress((i + 1) / len(trending_tickers))
                        
                    if results:
                        st.success(f"Found {len(results)} 'Buy' consensus stocks out of the current internet trends!")
                        for s in results[:5]:  # show top 5
                            upside_str = ""
                            if s['price'] != 'N/A' and s['target'] != 'N/A':
                                try:
                                    upside = ((float(s['target']) - float(s['price'])) / float(s['price'])) * 100
                                    upside_str = f" | <strong>Projected Upside:</strong> {upside:.1f}%"
                                except:
                                    pass
                                    
                            st.markdown(
                                f"""
                                <div style='background: linear-gradient(135deg, #10b98125 0%, #10b98110 100%); 
                                            border-left: 5px solid #10b981; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                                    <div style='font-size: 22px; font-weight: 700; color: #10b981;'>{s['ticker']} - {s['name']}</div>
                                    <div style='font-size: 16px; font-weight: 700; color: #333;'>Wall Street Consensus: {s['rec']}</div>
                                    <div style='font-size: 14px; color: #444; margin-top: 8px;'>
                                        <strong>Current Price:</strong> ${s['price']} <br>
                                        <strong>Analyst Mean Target:</strong> ${s['target']} {upside_str}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    else:
                        st.warning("Currently, none of the top trending stocks online have a clear 'Buy' consensus from analysts.")
                        
                except Exception as e:
                    st.error(f"❌ Failed to fetch online data: {str(e)}")

st.markdown("---")
st.caption("This is an educational AI model for predictive assessment and not an official financial institution decision.")

if enable_live_monitoring:
    time.sleep(3)
    st.rerun()
