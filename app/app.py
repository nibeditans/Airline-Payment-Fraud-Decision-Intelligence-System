import pandas as pd
import streamlit as st

from src.prediction import load_model_bundle, predict, transform_features
from src.risk import get_model_decision, get_recommendation, get_risk_category
from src.xai import explain_prediction


MODEL_PATH = "models/airline_fraud_xgboost.joblib"

st.set_page_config(
    page_title="APFDIS",
    page_icon="✈️",
    layout="wide",
)

@st.cache_resource
def load_bundle():
    return load_model_bundle(MODEL_PATH)

model_bundle = load_bundle()

st.title("Airline Payment Fraud Decision Intelligence System")
st.caption(
    "Assess transaction-level payment fraud risk, understand the factors influencing the model prediction, and support risk-based payment decisions."
)

st.divider()
st.subheader("Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input(
        "Amount",
        min_value=0.0,
        value=100.0,
        step=10.0,
    )

    billing_country = st.text_input(
        "Billing Country",
        value="IN",
    )

    route = st.text_input(
        "Route",
        value="HND-NRT",
    )

with col2:
    card_bin = st.text_input(
        "Card BIN",
        value="140442",
    )

    account_age_days = st.number_input(
        "Account Age (days)",
        min_value=0,
        value=30,
        step=1,
    )

    currency = st.text_input(
        "Currency",
        value="USD",
    )

with col3:
    transaction_date = st.date_input(
        "Transaction Date"
    )

    transaction_hour = st.selectbox(
        "Transaction Hour",
        range(24),
        index=12,
        format_func=lambda hour: f"{hour:02d}:00",
    )

st.divider()

st.markdown("**Input Guidance**")

guide_col1, guide_col2 = st.columns(2)

with guide_col1:
    st.caption("**Amount:** Enter the transaction amount.")
    st.caption(
        "**Billing Country:** Use a standard 2-letter country code. E.g. IN, US, GB, AE."
    )
    st.caption(
        "**Route:** Use departure-arrival airport codes. E.g. HND-NRT, DEL-LHR, JFK-LHR."
    )
    st.caption(
        "**Card BIN:** Enter the 6-digit BIN associated with the payment card."
    )

with guide_col2:
    st.caption("**Account Age:** Enter the account age in days.")
    st.caption(
        "**Currency:** Use a standard 3-letter currency code. E.g. USD, INR, EUR, GBP."
    )
    st.caption("**Transaction Date:** Select the transaction date.")
    st.caption("**Transaction Hour:** Select the transaction hour.")


def validate_transaction(
    amount, account_age_days, billing_country, route,
    card_bin, currency, model_bundle,
):
    errors = []
    warnings = []

    billing_country = billing_country.strip().upper()
    route = route.strip().upper()
    card_bin = card_bin.strip()
    currency = currency.strip().upper()

    if amount < 0:
        errors.append("Amount must be greater than or equal to 0.")

    if account_age_days < 0:
        errors.append("Account Age must be greater than or equal to 0 days.")

    if not billing_country:
        errors.append("Billing Country cannot be empty.")
    elif not (
        len(billing_country) == 2
        and billing_country.isalpha()
    ):
        errors.append(
            "Billing Country must be a 2-letter country code. E.g. IN, US, GB, AE."
        )

    if not route:
        errors.append("Route cannot be empty.")
    elif not (
        len(route) == 7
        and route[3] == "-"
        and route[:3].isalpha()
        and route[4:].isalpha()
    ):
        errors.append(
            "Route must use departure-arrival airport codes. E.g. HND-NRT, DEL-LHR, JFK-LHR."
        )

    if not card_bin:
        errors.append("Card BIN cannot be empty.")
    elif not card_bin.isdigit() or len(card_bin) != 6:
        errors.append("Card BIN must be a 6-digit value.")

    if not currency:
        errors.append("Currency cannot be empty.")
    elif not (
        len(currency) == 3
        and currency.isalpha()
    ):
        errors.append(
            "Currency must be a 3-letter currency code. E.g. USD, INR, EUR, GBP."
        )

    if errors:
        return errors, warnings, {
            "billing_country": billing_country,
            "route": route,
            "card_bin": card_bin,
            "currency": currency,
        }

    preprocessor = model_bundle["preprocessor"]
    encoder = preprocessor.named_transformers_["cat"]
    categorical_columns = preprocessor.transformers_[1][2]

    known_categories = dict(
        zip(categorical_columns, encoder.categories_)
    )

    categorical_values = {
        "billing_country": billing_country,
        "route": route,
        "card_bin": card_bin,
        "currency": currency,
    }

    field_labels = {
        "billing_country": "Billing Country",
        "route": "Route",
        "card_bin": "Card BIN",
        "currency": "Currency",
    }

    for feature, value in categorical_values.items():
        if value not in known_categories[feature]:
            warnings.append(
                f'{field_labels[feature]} "{value}" was not represented '
                "in the model's training data. The prediction was generated using the model's learned patterns from the available information."
            )

    return errors, warnings, categorical_values


if st.button("Assess Transaction", type="primary", use_container_width=True):

    errors, warnings, validated_values = validate_transaction(
        amount,
        account_age_days,
        billing_country,
        route,
        card_bin,
        currency,
        model_bundle,
    )

    if errors:
        st.error("Transaction could not be assessed.")

        st.write("Please correct the following:")

        for error in errors:
            st.write(f"- {error}")

    else:
        transaction_datetime = (
            pd.Timestamp(transaction_date)
            + pd.Timedelta(hours=transaction_hour)
        )

        transaction = pd.DataFrame([{
            "transaction_date": transaction_datetime,
            "amount": amount,
            "billing_country": validated_values["billing_country"],
            "route": validated_values["route"],
            "card_bin": validated_values["card_bin"],
            "account_age_days": account_age_days,
            "currency": validated_values["currency"],
        }])

        try:
            probabilities, _ = predict(
                transaction,
                model_bundle,
            )

            fraud_probability = float(probabilities[0])
            threshold = float(model_bundle["threshold"])

            risk_category = get_risk_category(
                fraud_probability,
                threshold,
            )

            model_decision = get_model_decision(
                fraud_probability,
                threshold,
            )

            recommendation = get_recommendation(
                risk_category,
            )

            transformed_features = transform_features(
                transaction,
                model_bundle,
            )

            explanation = explain_prediction(
                transformed_features,
                model_bundle["preprocessor"],
                model_bundle["model"],
            )

            st.subheader("Risk Assessment")

            if warnings:
                st.warning(
                    "Model Coverage Note"
                )

                for warning in warnings:
                    st.write(f"- {warning}")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric(
                    "Fraud Probability",
                    f"{fraud_probability:.2%}",
                )

            with result_col2:
                st.metric(
                    "Risk Category",
                    risk_category.replace("_", " ").title(),
                )

            with result_col3:
                st.metric(
                    "Model Decision",
                    model_decision,
                )

            st.caption(
                f"Decision threshold: {threshold:.2%}"
            )

            st.subheader("Decision Guidance")

            st.write(
                f"**Recommendation:** {recommendation}"
            )

            if risk_category == "low_risk":
                st.success(
                    "The predicted fraud risk is well below the "
                    "decision threshold. Approval is recommended."
                )

            elif risk_category == "high_risk":
                st.error(
                    "The predicted fraud risk is substantially above "
                    "the decision threshold. Further fraud investigation "
                    "or payment controls may be appropriate."
                )

            else:
                st.warning(
                    "The predicted fraud risk is close to the decision threshold. Additional verification or manual review may be appropriate."
                )

            st.subheader("Model Explanation")

            st.caption(
                "Factors influencing this model prediction. Positive SHAP values push the prediction toward fraud, while negative values push it toward legitimate."
            )

            explanation_display = explanation[
                ["feature", "shap_value"]
            ].head(5)

            explanation_display = explanation_display.rename(
                columns={
                    "feature": "Feature",
                    "shap_value": "SHAP Contribution",
                }
            )

            st.dataframe(
                explanation_display,
                hide_index=True,
                use_container_width=True,
            )

        except Exception as error:
            st.error(
                "The transaction could not be assessed. "
                "Please check the input values and try again."
            )
            st.caption(f"Error: {error}")

st.markdown("----")
st.caption("**Developed by Nate!**")
