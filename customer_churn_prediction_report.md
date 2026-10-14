# Executive Report: Customer Churn Prediction & Retention Strategy
**Author:** Data Science & Analytics Team  
**Dataset:** Telco Customer Churn (7,043 customer accounts)  
**Task Deliverable:** 1–2 Page Project Summary & Business Value Analysis

---

## 1. Executive Summary & Business Problem

### 1.1 The Challenge
Customer churn poses a direct threat to recurring subscription revenue and increases customer acquisition costs. The primary business objective is to build an automated machine learning classification model capable of proactively identifying customers at high risk of canceling their telecommunications service before they leave, allowing the marketing and customer retention teams to deploy targeted intervention programs.

### 1.2 Target & Problem Definition
* **Problem Type:** Supervised binary classification.
* **Target Variable:** `Churn` (encoded as `0 = Stay`, `1 = Churn`).
* **Cost of Misclassification:** A **False Negative (FN)**—predicting a customer will stay when they actually leave—is significantly more costly than a **False Positive (FP)**. Missing an actual churner results in the permanent loss of recurring customer lifetime value (LTV). Conversely, a False Positive merely incurs minor marketing expenditures (e.g., outreach emails or modest discounts). Consequently, **Recall for the churn class** is the primary decision-making evaluation metric.

---

## 2. Key Exploratory Data Analysis (EDA) Findings

The dataset consists of 7,043 customer records with 20 input attributes:
1. **Class Imbalance:** 73.46% (5,174) of customers stayed, while 26.54% (1,869) churned. Accuracy alone is an inadequate evaluation metric.
2. **Contract Type Disparity:** 
   * **Month-to-month contracts:** **42.71%** churn rate.
   * **One-year contracts:** **11.27%** churn rate.
   * **Two-year contracts:** **2.83%** churn rate.
   *(Note: Correlation does not prove causation; customers seeking flexibility or lacking initial confidence in service quality often opt for monthly contracts).*
3. **Tenure Relationship:** Customers with shorter tenure ($\le 12$ months) experience a **47.44%** churn rate, dropping sharply to **9.51%** for customers with over 48 months of tenure.
4. **Billing Impact:** Churners paid higher monthly charges on average ($74.44 vs. $61.27 median/mean), signaling high price sensitivity when service value is not perceived as competitive.
5. **Data Quality Integrity:** 11 whitespace entries in `TotalCharges` were converted to missing numeric values and imputed via median values within a strict leakage-free `Pipeline`.

---

## 3. Modeling Methodology & Preprocessing

To ensure reproducibility and eliminate data leakage:
* **Stratified Split:** Data was divided into an 80% training set ($N = 5,634$) and a 20% held-out test set ($N = 1,409$).
* **Pipeline Integration:** A `ColumnTransformer` was fitted strictly on training data:
  * **Numerical Features:** Median imputation followed by standard feature scaling (`StandardScaler`).
  * **Categorical Features:** Most-frequent imputation followed by one-hot encoding (`OneHotEncoder(handle_unknown='ignore')`).
* **Validation Setup:** Stratified 5-Fold Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True)`) on training data.

---

## 4. Benchmark Model Comparison

Evaluating untuned models on the held-out test set ($N = 1,409$):

| Model | Accuracy | Precision (Churn = 1) | Recall (Churn = 1) | F1-Score (Churn = 1) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Dummy Classifier)** | 0.735 | 0.000 | 0.000 | 0.000 |
| **Logistic Regression (Default)** | **0.801** | **0.657** | 0.527 | **0.585** |
| **Random Forest (Balanced)** | 0.783 | 0.621 | 0.468 | 0.534 |
| **K-Nearest Neighbors (KNN)** | 0.768 | 0.563 | **0.561** | 0.562 |
| **Decision Tree (Balanced)** | 0.735 | 0.500 | 0.497 | 0.499 |

*Takeaway:* Default models missed nearly half of all at-risk customers (Recall ~47%–56%), requiring hyperparameter optimization focused on class sensitivity.

---

## 5. Model Improvement (Hyperparameter Tuning)

Using `GridSearchCV` on the training set with 5-fold cross-validation optimizing for **Recall**:
* **Logistic Regression:** Tuned with $C = 1.0$ and `class_weight='balanced'`. Cross-validation recall surged from **53.8%** to **79.7%** (a gain of **+25.9 percentage points**).
* **Random Forest:** Tuned depth and estimators, improving cross-validation recall from **46.8%** to **71.5%** (+24.7 percentage points).

---

## 6. Final Evaluation on Held-Out Test Set

The tuned **Logistic Regression** pipeline was evaluated once on the test set:

| Metric | Score | Context |
| :--- | :---: | :--- |
| **Recall (Churn = 1)** | **79.1%** | Successfully flags 296 out of 374 actual churners. |
| **Precision (Churn = 1)** | **49.7%** | Approximately 1 in 2 flagged accounts will churn without action. |
| **F1-Score (Churn = 1)** | **0.611** | Optimal harmonic trade-off between recall and precision. |
| **Overall Accuracy** | **73.2%** | Robust generalization without collapsing to the majority class. |

### Final Confusion Matrix ($N = 1,409$):
* **True Positives (TP = 296):** Correctly identified churners enabling timely proactive retention.
* **False Positives (FP = 299):** Non-churners flagged as at-risk (false alarms incurring minor promotion cost).
* **True Negatives (TN = 736):** Correctly classified loyal accounts requiring no unnecessary spending.
* **False Negatives (FN = 78):** Missed churners (reduced from 177 in the untuned model to only 78).

---

## 7. Model Limitations

1. **Precision vs. Recall Trade-Off:** Achieving ~79% Recall yields a ~50% Precision rate (299 False Positives). Campaigns utilizing high-cost financial gifts could erode margins if not monitored.
2. **Class Imbalance Sensitivity:** Compensating for the 73.5 / 26.5 class imbalance via `class_weight='balanced'` effectively shifted the decision boundary to prioritize churners, naturally increasing the false alarm rate.
3. **Static Snapshot Nature:** The dataset lacks dynamic engagement telemetry (e.g., recent customer service ticket sentiment, local broadband outage history, competitor promotional price wars). Periodic retraining is required as market behaviors evolve.

---

## 8. Business Recommendations & Next Steps

### 8.1 Model Recommendation
Deploy **Logistic Regression (tuned with `class_weight='balanced'`, $C=1.0$)**:
* **Maximum Risk Capture:** Slashes missed churners by more than 50% compared to untuned models.
* **Operational Explainability:** Provides direct coefficient transparency for marketing stakeholders, unlike black-box models.
* **Lightweight Deployment:** Instantaneous scoring latency suitable for integration into CRM systems.

### 8.2 Actionable Retention Strategies
1. **Targeted Contract Conversion Incentives:** Proactively offer high-risk month-to-month customers attractive 3-month bill credits or free speed tiers in exchange for transitioning to an annual contract.
2. **First 90-Day Onboarding Support:** Route subscribers in months 1–3 flagged as at-risk to dedicated customer care agents for preventative technical assistance and billing clarity.
3. **Operational Threshold Control:**
   * Apply a **lower threshold (0.35–0.40)** for automated, zero-cost retention outreach (e.g., automated email surveys and account check-ins).
   * Apply a **higher threshold (0.60–0.70)** for expensive retention interventions (e.g., personal retention phone calls and subsidized hardware).