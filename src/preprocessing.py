import pandas as pd

FINAL_FEATURES = ["amount", "billing_country", "route", "card_bin", "account_age_days", "currency", "transaction_hour", "transaction_dayofweek"]

def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare raw transaction data for model preprocessing.

    Creates the time-based features used by the final model and
    returns the eight final modeling features in the required order.
    """
    data = data.copy()

    data["transaction_hour"] = data["transaction_date"].dt.hour
    data["transaction_dayofweek"] = data["transaction_date"].dt.dayofweek

    return data[FINAL_FEATURES]

