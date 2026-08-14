"""Business risk decision logic for fraud predictions."""

def get_model_decision(fraud_probability: float, threshold: float) -> str:
    """Return the model decision based on the fraud threshold."""
    return "Flag" if fraud_probability >= threshold else "Approve"

def get_risk_category(fraud_probability: float, threshold: float) -> str:
    """Classify a transaction as low, near-threshold, or high risk."""
    distance = fraud_probability - threshold

    if distance >= 0.05:
        return "high_risk"
    elif distance <= -0.05:
        return "low_risk"
    else:
        return "near_threshold"

def get_recommendation(risk_category: str) -> str:
    """Return the business recommendation for a risk category."""
    recommendations = {
        "low_risk": "Approve",
        "high_risk": "Flag",
        "near_threshold": "Review-sensitive",
    }

    return recommendations[risk_category]
