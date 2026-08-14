import pandas as pd
import shap

def explain_prediction(transformed_features, preprocessor, model):
    """
    Generate a local SHAP explanation for one or more transactions.

    Returns SHAP contributions aggregated back to the original
    model features.
    """
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(transformed_features)

    values = shap_values.values

    if values.ndim == 3:
        values = values[:, :, 1]

    shap_df = pd.DataFrame(values, columns=feature_names)

    numeric_columns = preprocessor.transformers_[0][2]
    categorical_columns = preprocessor.transformers_[1][2]

    feature_mapping = {}

    for feature in numeric_columns:
        feature_mapping[f"num__{feature}"] = feature

    for feature in categorical_columns:
        prefix = f"cat__{feature}_"

        for transformed_feature in feature_names:
            if transformed_feature.startswith(prefix):
                feature_mapping[transformed_feature] = feature

    feature_groups = pd.Series(
        feature_mapping,
        name="original_feature"
    )

    grouped = shap_df.T.groupby(feature_groups).sum().T

    if len(grouped) != 1:
        raise ValueError("Expected a single transaction for local explanation.")

    explanation = pd.DataFrame({
        "feature": grouped.columns,
        "shap_value": grouped.iloc[0].values,
    })

    explanation["abs_shap"] = explanation["shap_value"].abs()

    return explanation.sort_values(
        "abs_shap",
        ascending=False
    ).reset_index(drop=True)
