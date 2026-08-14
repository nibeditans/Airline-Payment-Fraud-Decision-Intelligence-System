# **Business Decision Layer**

## Purpose

The model predicts the probability that a payment transaction is fraudulent.

The decision layer turns that probability into a practical recommendation. The goal is not to replace payment operations, but to help identify transactions that may need additional attention.

The basic flow is:

> **Transaction ⇰ Fraud Probability ⇰ Risk Decision ⇰ Business Recommendation**

## 1. Model Output

For each transaction, the model produces a fraud probability between 0 and 1.

This probability represents how strongly the model considers the transaction to be fraudulent based on the available transaction features.

A probability by itself is not a business decision. We therefore use a decision threshold.

## 2. Decision Threshold

The final model threshold is **0.91**.

* **Fraud probability < 0.91 → Approve**
* **Fraud probability ≥ 0.91 → Flag**

The threshold was selected during model evaluation by considering the trade-off between detecting fraud and unnecessarily interrupting legitimate transactions.

This means the system is intentionally designed around the business problem rather than simply maximizing classification accuracy.

## 3. Risk Interpretation

The model decision can be viewed together with the transaction's distance from the threshold.

### Low-risk

Transactions with probabilities well below 0.91 are strong approval candidates.

**Recommendation:** Approve.

### High-risk

Transactions with probabilities well above 0.91 have strong model evidence of fraud.

**Recommendation:** Flag for additional payment-risk handling.

### Near-threshold

Transactions close to 0.91 are less clear-cut. A small change in the model probability could change the model decision.

**Recommendation:** Treat as review-sensitive cases rather than interpreting the model output as absolute certainty.

This is particularly useful because a model threshold is a decision boundary, not a guarantee that transactions on either side are truly legitimate or fraudulent.

## 4. Using Model Explanations

The model's prediction should be supported by an explanation wherever possible.

SHAP analysis helps answer two questions:

* **Globally:** Which features have the strongest influence on model predictions?
* **For individual transactions:** Which feature values are pushing a prediction toward higher or lower fraud risk?

The selected XAI visuals are stored in: `assets/xai_visuals/`

The main findings from the XAI analysis are used as supporting evidence for the decision layer, rather than treating feature importance as a business rule by itself.

## 5. Business Recommendations

The system is intended to support payment-risk decisions in three broad situations:

| Risk situation | Model output                | Recommended action        |
| -------------- | --------------------------- | ------------------------- |
| Low risk       | Probability well below 0.91 | Approve                   |
| High risk      | Probability well above 0.91 | Flag                      |
| Near threshold | Probability close to 0.91   | Review-sensitive handling |

For flagged transactions, the model probability and supporting explanation can help an analyst understand **why the transaction received a high-risk score**.

The exact operational action after a flag, such as manual review, additional verification, or rejection would depend on the airline's fraud policy and is therefore outside the assumptions supported by this dataset.

## 6. Important Assumptions

The dataset does not contain information about the airline's actual fraud-handling costs, review capacity, customer impact, or operational policies.

Therefore, recommendations about specific operational actions are treated as business scenarios, not measured facts.

The model provides risk estimates and decision support. It does not guarantee that a transaction is fraudulent or legitimate.

## 7. Final Decision Framework

The overall decision process is:

1. **Score the transaction** ⇰ Generate fraud probability.
2. **Compare with the 0.91 threshold** ⇰ Approve or Flag.
3. **Consider how close the score is to the threshold** ⇰ Identify straightforward versus review-sensitive cases.
4. **Use model explanations** ⇰ Understand the factors contributing to the risk score.
5. **Apply the appropriate business process** ⇰ Use the model as decision support within the airline's payment-risk workflow.

The key idea is simple:

> **The model predicts risk. The decision layer turns that risk into an informed action.**

