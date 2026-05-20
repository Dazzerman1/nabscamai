import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="NAB ScamShield AI", layout="wide")

st.title("NAB ScamShield AI Prototype")
st.caption("AI-Powered Payment Scam and Fraud Detection System")

# Historical transaction data used to train the ML anomaly detection model
historical_data = pd.DataFrame({
    "amount": [20, 35, 50, 80, 120, 150, 200, 75, 60, 95, 110, 250, 300, 45, 90, 130, 180, 210],
    "new_payee": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
    "international": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    "late_night": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    "new_device": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
})

model = IsolationForest(contamination=0.15, random_state=42)
model.fit(historical_data)

st.info(
    "This prototype demonstrates genuine AI functionality by using an Isolation Forest "
    "machine learning model trained on sample transaction data to detect anomalous payment behaviour."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Transaction Input",
    "2. AI Processing",
    "3. Fraud Result",
    "4. Analyst Dashboard"
])

with tab1:
    st.header("Transaction Input: Customer Payment App")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("Payment amount ($)", min_value=1, value=6000)
        merchant = st.text_input("Merchant / Payee", value="Crypto Exchange Ltd")
        payment_type = st.selectbox(
            "Payment type",
            ["Retail purchase", "Bank transfer", "Online marketplace", "Crypto transfer"]
        )

    with col2:
        new_payee = st.selectbox("New payee?", ["No", "Yes"])
        international = st.selectbox("International payment?", ["No", "Yes"])
        late_night = st.selectbox("Late night transaction?", ["No", "Yes"])
        new_device = st.selectbox("New device used?", ["No", "Yes"])

    st.write("### Customer Action")
    st.write("Customer submits a payment through NAB digital banking.")

    st.write("### Current Transaction")
    st.dataframe(pd.DataFrame({
        "Feature": ["Amount", "Merchant", "Payment Type", "New Payee", "International", "Late Night", "New Device"],
        "Value": [amount, merchant, payment_type, new_payee, international, late_night, new_device]
    }), use_container_width=True)

new_payee_num = 1 if new_payee == "Yes" else 0
international_num = 1 if international == "Yes" else 0
late_night_num = 1 if late_night == "Yes" else 0
new_device_num = 1 if new_device == "Yes" else 0

input_data = pd.DataFrame({
    "amount": [amount],
    "new_payee": [new_payee_num],
    "international": [international_num],
    "late_night": [late_night_num],
    "new_device": [new_device_num]
})

ml_prediction = model.predict(input_data)[0]
ml_result = "Anomaly detected" if ml_prediction == -1 else "Normal transaction pattern"

risk_points = 0
reasons = []

if amount > 1000:
    risk_points += 25
    reasons.append("Transaction amount is significantly higher than normal customer behaviour.")

if new_payee_num == 1:
    risk_points += 20
    reasons.append("Payment is being sent to a new or unverified payee.")

if international_num == 1:
    risk_points += 20
    reasons.append("Transaction is international, increasing scam and recovery risk.")

if late_night_num == 1:
    risk_points += 10
    reasons.append("Transaction occurred outside typical customer activity hours.")

if new_device_num == 1:
    risk_points += 15
    reasons.append("Payment was initiated from a new device.")

if payment_type == "Crypto transfer":
    risk_points += 20
    reasons.append("Crypto transfers are high risk because funds are difficult to recover.")

if ml_prediction == -1:
    risk_points += 20
    reasons.append("The machine learning model identified this payment as anomalous.")

risk_score = min(risk_points, 100)

if risk_score >= 70:
    risk_level = "HIGH RISK"
    action = "Transaction blocked. Customer verification and fraud team review required."
    business_outcome = "Payment blocked, customer notified, fraud team alerted."
elif risk_score >= 40:
    risk_level = "MEDIUM RISK"
    action = "Customer warning shown. Extra confirmation required before payment proceeds."
    business_outcome = "Scam warning displayed and transaction logged for monitoring."
else:
    risk_level = "LOW RISK"
    action = "Payment approved."
    business_outcome = "Payment proceeds and transaction is stored as normal behaviour."

confidence = min(99, 75 + int(risk_score / 4))

with tab2:
    st.header("AI Processing / Analysis")

    st.subheader("AI Processing Pipeline")

    processing_steps = pd.DataFrame({
        "Step": [
            "1. Transaction input received",
            "2. Feature extraction",
            "3. ML anomaly detection",
            "4. Risk score generation",
            "5. Decision logic",
            "6. Automated business response"
        ],
        "Processing Logic": [
            "Customer payment details are collected from the payment form.",
            "The payment is converted into AI model features: amount, new payee, international, late night, and new device.",
            "An Isolation Forest machine learning model compares the payment against historical customer behaviour.",
            "The ML result is combined with scam indicators to calculate a risk score.",
            "The transaction is classified as Low, Medium, or High Risk.",
            "The system approves, warns, blocks, or escalates the payment depending on the risk level."
        ]
    })

    st.dataframe(processing_steps, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Machine Learning Model", "Isolation Forest")
    col2.metric("ML Output", ml_result)
    col3.metric("AI Confidence", f"{confidence}%")

    st.write("### Model Input Features")
    st.dataframe(input_data, use_container_width=True)

    st.write("### Behaviour Comparison")
    average_spend = historical_data["amount"].mean()

    comparison = pd.DataFrame({
        "Category": ["Average Historical Transaction", "Current Transaction"],
        "Amount": [average_spend, amount]
    })

    st.bar_chart(comparison.set_index("Category"))

    st.write("### Workflow Diagram")
    st.code(
        "Customer Payment Input → Feature Extraction → ML Anomaly Detection → Risk Score → Decision Logic → Business Action"
    )

with tab3:
    st.header("AI Fraud Detection Result")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Risk Assessment")

        if risk_level == "HIGH RISK":
            st.error(risk_level)
        elif risk_level == "MEDIUM RISK":
            st.warning(risk_level)
        else:
            st.success(risk_level)

        st.metric("Risk Score", f"{risk_score}%")
        st.progress(risk_score / 100)

        st.write("### Explainable AI Justification")
        if reasons:
            for reason in reasons:
                st.write("- " + reason)
        else:
            st.write("- This payment is consistent with normal customer transaction behaviour.")

    with col2:
        st.subheader("Automated Action / Response")
        st.write(action)

        st.write("### Business Outcome")
        st.write(business_outcome)

        st.write("### Customer Notification")
        if risk_level == "HIGH RISK":
            st.error(
                f"NAB ScamShield AI has blocked your ${amount} payment to {merchant}. "
                "This payment appears unusual and requires verification before it can continue."
            )
        elif risk_level == "MEDIUM RISK":
            st.warning(
                f"This ${amount} payment to {merchant} has unusual features. "
                "Please confirm that you personally authorised this payment."
            )
        else:
            st.success(
                f"Your ${amount} payment to {merchant} appears consistent with normal behaviour."
            )

    st.subheader("Risk Factor Breakdown")

    risk_factors = pd.DataFrame({
        "Risk Factor": [
            "High transaction amount",
            "New payee",
            "International payment",
            "Late night activity",
            "New device",
            "Crypto transfer",
            "ML anomaly detection"
        ],
        "Detected": [
            "Yes" if amount > 1000 else "No",
            new_payee,
            international,
            late_night,
            new_device,
            "Yes" if payment_type == "Crypto transfer" else "No",
            "Yes" if ml_prediction == -1 else "No"
        ],
        "Risk Impact": [
            "High" if amount > 1000 else "Low",
            "Medium" if new_payee == "Yes" else "Low",
            "High" if international == "Yes" else "Low",
            "Medium" if late_night == "Yes" else "Low",
            "Medium" if new_device == "Yes" else "Low",
            "High" if payment_type == "Crypto transfer" else "Low",
            "High" if ml_prediction == -1 else "Low"
        ]
    })

    st.dataframe(risk_factors, use_container_width=True)

with tab4:
    st.header("Fraud Analyst Dashboard: NAB Internal View")

    status = "Flagged" if risk_score >= 70 else "Approved"

    dashboard = pd.DataFrame({
        "Transaction ID": ["TXN00145872", "TXN00145873", "TXN00145874", "TXN00145875", "TXN00145876"],
        "Customer": ["John Smith", "Angela Lee", "Michael Chen", "Sarah Johnson", "David Brown"],
        "Amount": [amount, 1250, 2980, 7600, 530],
        "Risk Score": [f"{risk_score}%", "85%", "82%", "95%", "12%"],
        "Status": [status, "Flagged", "Flagged", "Flagged", "Approved"],
        "Action": [
            "Review" if risk_score >= 70 else "Monitor",
            "Review",
            "Review",
            "Escalate",
            "None"
        ]
    })

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Alerts Today", "127", "+12%")
    col2.metric("High Risk Alerts", "45", "+8%")
    col3.metric("Blocked Transactions", "32", "+15%")
    col4.metric("Estimated Loss Prevented", "$287,450", "+18%")

    st.subheader("High Risk Alert Queue")
    st.dataframe(dashboard, use_container_width=True)

    st.subheader("Automated Fraud Workflow")
    workflow = pd.DataFrame({
        "Workflow Stage": [
            "Payment Submitted",
            "AI Risk Analysis",
            "Transaction Flagged",
            "Customer Notified",
            "Fraud Team Alerted",
            "Case Added to Dashboard"
        ],
        "Status": [
            "Completed",
            "Completed",
            "Completed" if risk_score >= 40 else "Not Required",
            "Completed" if risk_score >= 40 else "Not Required",
            "Completed" if risk_score >= 70 else "Not Required",
            "Completed" if risk_score >= 70 else "Not Required"
        ]
    })

    st.dataframe(workflow, use_container_width=True)

    st.subheader("Transactions by Risk Level")

    risk_summary = pd.DataFrame({
        "Risk Level": ["Low Risk", "Medium Risk", "High Risk"],
        "Transactions": [120, 35, 45]
    })

    fig, ax = plt.subplots()
    ax.bar(risk_summary["Risk Level"], risk_summary["Transactions"])
    ax.set_title("Transactions by Risk Level")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Transactions")
    st.pyplot(fig)

    st.subheader("Prototype Evidence Summary")
    st.write("""
    This prototype demonstrates:
    - User input through a customer payment form
    - AI processing using an Isolation Forest machine learning model
    - AI output through risk score, risk level, and explanation
    - Automated business action through payment approval, warning, blocking, and fraud team escalation
    """)