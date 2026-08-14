# Airline Payment Fraud Decision Intelligence System

An end-to-end Data Science project for assessing payment fraud risk in airline transactions and turning model predictions into practical payment decisions.

## Overview

Airlines process a large number of payment transactions every day. A payment system needs to balance two problems:

* Approving fraudulent transactions
* Interrupting legitimate customers unnecessarily

I built this project to estimate the probability that an airline payment transaction is fraudulent and use that risk score to support a practical payment decision.

The project goes beyond a simple fraud classification model. I built a complete workflow:

> **Data → Fraud Risk Prediction → Model Interpretation → Risk Assessment → Payment Decision**

The final system can be used through a Streamlit application where a user can enter transaction details, receive a fraud probability, see the risk level, and understand the factors influencing the prediction.

## Business Problem

A fraud model does not directly answer the operational question:

> **What should we do with this transaction?**

A probability by itself is not enough.

For example, a transaction with a fraud probability of 0.91 should not automatically be treated in the same way as one with a probability of 0.02.

I therefore separated the project into two parts:

1. **Prediction**: estimate the probability of fraud.
2. **Decision support**: use that probability to recommend an appropriate action.

This distinction is one of the main ideas behind the project.

## Dataset

I used the **Airline Fraud Transactions** dataset from Hugging Face.

The original dataset contains approximately 3 million payment transaction records.

I did not include the raw dataset in this repository because it is externally hosted and is not my own dataset.

**Dataset:** `analytical-community/airline_fraud_transactions`

⇰ **[Open the original Hugging Face dataset](https://huggingface.co/datasets/analytical-community/airline_fraud_transactions)**

The original data contains transaction, payment, account, geographic, and session-related information.

Some fields contain identifiers or sensitive-looking information such as email addresses, IP addresses, payment identifiers, and device IDs. I did not use these raw identifiers in the final public modeling dataset.

## Final Modeling Data

After data cleaning, analysis, feature engineering, and feature selection, I used the following features for the final model:

```
amount
billing_country
route
card_bin
account_age_days
currency
transaction_hour
transaction_dayofweek
```

The target variable is: `is_fraud`

The final prepared datasets were split into training and validation data while preserving the time-based nature of the transaction data.

The final training data contains about 2.1 million transactions, with a fraud rate of approximately 1.5%.

## Approach

I followed an end-to-end Data Science workflow rather than treating this as only a machine learning problem.

### 1. Data Understanding

I first inspected the dataset to understand:

* Available fields
* Data types
* Missing values
* Class distribution
* Transaction characteristics
* Potential identifier and leakage concerns

### 2. Data Cleaning and EDA

I cleaned the data and investigated patterns related to:

* Transaction amounts
* Countries
* Routes
* Payment information
* Account age
* Failed attempts
* Transaction timing
* Fraud distribution

The analysis helped determine which variables were useful and which should not be carried into the final model.

### 3. Feature Engineering

I created time-based features from the transaction date, including:

* Transaction hour
* Day of the week

I then prepared the final modeling data for training and validation.

### 4. Model Development

I used **XGBoost** to predict the probability of fraud.

The model produces a probability rather than only returning a fraud/non-fraud label.

This gives the decision layer more information to work with.

### 5. Threshold Selection

I did not use the default classification threshold of `0.50`.

Instead, I evaluated different thresholds and selected: `Decision threshold = 0.91`

This threshold reflects the business trade-off between catching fraudulent transactions and avoiding unnecessary intervention on legitimate transactions.

The important point is that **0.91 is a project-specific decision threshold**, not a universal fraud-detection threshold.

## Decision Layer

The model probability is converted into a practical risk category.

I used three risk categories:

| Risk category    | Meaning                                                        |
| ---------------- | -------------------------------------------------------------- |
| `low_risk`       | Fraud probability is sufficiently below the decision threshold |
| `near_threshold` | Prediction is close to the decision threshold                  |
| `high_risk`      | Fraud probability is sufficiently above the decision threshold |

These categories support three operational recommendations:

| Recommendation       | Meaning                                                                                                  |
| -------------------- | -------------------------------------------------------------------------------------------------------- |
| **Approve**          | The transaction can proceed based on the model's risk assessment                                         |
| **Review-sensitive** | The prediction is close to the threshold, so additional verification or manual review may be appropriate |
| **Flag**             | The predicted fraud risk is high enough to warrant intervention                                          |

This is intentionally a **decision-support system**, not an automated claim that every high-risk transaction is definitely fraudulent.

## Model Explainability

I used **SHAP** to understand which features influenced individual predictions.

For each transaction, the system can provide the main factors influencing the model's fraud prediction.

This helps answer questions such as:

> **Why did the model consider this transaction risky?**

The explanations are designed to support human review and make the model easier to understand.

## Example Predictions

The final decision layer can produce results such as:

```
Fraud probability: 0.999769
Risk category: high_risk
Model decision: Flag
```

```
Fraud probability: 0.017528
Risk category:     low_risk
Model decision:    Approve
```

```text
Fraud probability: 0.908906
Risk category:     near_threshold
Model decision:    Approve
Recommendation:    Review-sensitive
```

The third example is especially important.

A prediction can be just below the decision threshold while still being close enough that additional verification may be worth considering.

This is why I kept **model prediction** and **business recommendation** as separate layers.

## Streamlit Application

I built a Streamlit application around the trained model.

The application allows a user to:

1. Enter transaction information
2. Generate a fraud probability
3. View the risk category
4. See the model decision
5. Review the factors influencing the prediction
6. Understand when additional verification may be appropriate

The application loads the saved model bundle rather than retraining the model every time it runs.

## Project Structure

```
airline-payment-fraud-decision-intelligence-system/
│
├── app/
│   └── app.py
│
├── assets/
│   └── xai_visuals/
│       ├── shap_amount_dependence.png
│       └── shap_global_importance.png
│
├── data/
│   └── processed/
│       ├── airline_fraud_cleaned.parquet
│       └── modeling/
│           ├── train.parquet
│           ├── validation.parquet
│           └── test.parquet
│
├── models/
│   └── airline_fraud_xgboost.joblib
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_development.ipynb
│   └── 05_model_interpretation_xai.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── prediction.py
│   ├── risk.py
│   └── xai.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

The repository separates exploratory work, reusable Python code, trained model artifacts, application code, and project assets.

I intentionally excluded the raw 3-million-row dataset from the repository.

## Technologies

I used the following main tools:

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* XGBoost
* SHAP
* Joblib
* DuckDB
* PyArrow
* Streamlit

I used these tools based on the needs of the project rather than adding technologies simply for the sake of the stack.

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/nibeditans/airline-payment-fraud-decision-intelligence-system.git
cd airline-payment-fraud-decision-intelligence-system
```

### 2. Create a virtual environment

```bash
py -m venv .venv
```

Activate it according to your operating system.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
py -m streamlit run app/app.py
```

The exact application entry point may differ depending on the final repository structure.

## Data Availability

The original dataset is available through Hugging Face:

`analytical-community/airline_fraud_transactions`

I excluded the downloaded Parquet file from GitHub because the dataset is not my own and the raw file is also unnecessarily large for a source repository.

If you wanna reproduce the data preparation workflow, you can obtain the original dataset directly from its source.😉

## Limitations

This project has several important limitations.

### Business costs are not available in the dataset

The dataset does not provide actual financial costs for:

* Fraud losses
* False declines
* Manual review
* Customer friction
* Chargebacks

Therefore, I did not pretend to have exact business costs.

The decision threshold and operational recommendations should be treated as a data-driven project scenario, not as a production airline policy.

### Model predictions are not proof of fraud

A high predicted probability means that the transaction looks risky according to the model.

It does not prove that the transaction is fraudulent.

### Dataset limitations

The model learns patterns from the available dataset. Real-world airline payment systems may contain additional signals that are not available here.

### Production considerations

A real production system would require additional work around:

* Continuous monitoring
* Data drift
* Model performance monitoring
* Threshold recalibration
* Feedback from fraud investigators
* Operational cost measurement
* Security and privacy controls
* Model retraining

These are outside the scope of the initial project.

## What I Built

The final project brings together several parts:

```
                    ┌──────────────────────┐
                    │  Payment Transaction │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Feature Pipeline   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    XGBoost Model     │
                    └──────────┬───────────┘
                               │
                    ┌──────────────────────┐
                    │   Fraud Probability  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Risk Assessment   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Decision Layer    │
                    │                      │
                    │  Approve / Review /  │
                    │         Flag         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   SHAP Explanation   │
                    └──────────────────────┘
```

The main idea is simple:

**The model estimates risk. The decision layer turns that risk into an actionable recommendation.**

## Final Takeaway

I built this project to demonstrate how I approach a real Data Science problem from beginning to end.

I started with the business problem, worked through the data, built and evaluated a fraud-risk model, selected a decision threshold using evidence, added model explanations, and finally connected everything to a practical decision layer and application.

The final system is therefore best viewed as a **fraud decision-support system**, with machine learning as one part of the overall solution.

⭐ If you found this project interesting and useful, feel free to star and fork the repo.

Follow for more exciting work.😁
