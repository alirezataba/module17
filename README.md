# Comparing Classifiers for Bank Marketing

## Project Overview

This project compares four classification algorithms for predicting whether a bank customer will subscribe to a term deposit after a telephone marketing campaign:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)

The analysis uses the **Bank Marketing** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing). The dataset contains results from 17 marketing campaigns conducted by a Portuguese banking institution between May 2008 and November 2010.

[View the complete Jupyter Notebook](prompt_III.ipynb)

## Business Objective

The objective is to build and compare classification models that predict whether a customer will subscribe to a term deposit. A useful model could help the bank prioritize customers who are more likely to subscribe, allocate marketing resources more efficiently, reduce unnecessary calls, and improve campaign conversion rates.

## Dataset

The full dataset contains:

- **41,188 observations**
- **20 input features**
- **1 binary target variable**, `y`
  - `0`: did not subscribe
  - `1`: subscribed

The target is highly imbalanced:

| Target | Count | Percentage |
|---|---:|---:|
| Did not subscribe | 36,548 | 88.7% |
| Subscribed | 4,640 | 11.3% |

Because a model that always predicts the majority class already achieves approximately **88.74% accuracy**, accuracy alone is not sufficient for evaluating model quality.

## Features Used for Modeling

To establish a focused initial comparison, the models use only the bank-client information available before the marketing call:

- `age`
- `job`
- `marital`
- `education`
- `default`
- `housing`
- `loan`

The `duration` feature was excluded from predictive modeling because call duration is unknown before a call takes place and would introduce information unavailable at prediction time.

## Data Preparation

The notebook performs the following preparation steps:

- Maps the target from `yes`/`no` to `1`/`0`.
- Retains the text value `unknown` as a valid categorical level rather than deleting or imputing those observations.
- Removes leading and trailing spaces from categorical values.
- Checks for conventional missing values and finds none.
- Finds 12 exact duplicate rows but retains them because no customer identifier is available to confirm that they are data-entry errors.
- Drops `pdays` after identifying that its sentinel value `999` is dominant and not fully consistent with the other previous-campaign variables.
- Standardizes `age` using `StandardScaler`.
- Encodes categorical features using `OneHotEncoder(handle_unknown="ignore")`.
- Uses an 80/20 stratified train-test split with `random_state=42`.

For hyperparameter tuning, preprocessing is placed inside a scikit-learn `Pipeline` so that scaling and encoding are fitted independently within each cross-validation fold.

## Exploratory Findings

The exploratory analysis confirms substantial class imbalance and shows that subscription rates vary across customer groups.

Selected observations include:

- Students had the highest job-based subscription rate at approximately **31.4%**.
- Retired customers had a subscription rate of approximately **25.2%**.
- Blue-collar customers had one of the lowest rates at approximately **6.9%**.
- Customers with a university degree had a subscription rate of approximately **13.7%**.
- The illiterate education category showed a high rate, but it contained only 18 observations and should therefore be interpreted cautiously.

These relationships are descriptive and should not be interpreted as causal.

## Baseline and Initial Logistic Regression

The majority-class baseline achieved **88.74% accuracy** by predicting that every customer would not subscribe. Its precision, recall, and F1-score for subscribers were all zero.

The initial Logistic Regression model produced the same practical outcome:

| Metric | Score |
|---|---:|
| Accuracy | 0.8874 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1-score | 0.0000 |

This demonstrates why accuracy is misleading for this imbalanced classification problem.

## Default Model Comparison

The four classifiers were first evaluated using their default settings.

| Model | Train Time (s) | Train Accuracy | Test Accuracy |
|---|---:|---:|---:|
| Logistic Regression | 0.1183 | 0.8873 | 0.8874 |
| KNN | 0.0057 | 0.8910 | 0.8781 |
| Decision Tree | 1.5403 | 0.9171 | 0.8650 |
| SVM | 124.7880 | 0.8882 | 0.8864 |

The default models had similar test accuracy, but scores near the baseline did not necessarily indicate useful detection of subscribers. KNN had a very short fit time because most of its computational work occurs during prediction. SVM required substantially more fitting time.

## Hyperparameter Tuning

`GridSearchCV` was used with three-fold stratified cross-validation. The search evaluated accuracy, precision, recall, and F1-score, while **F1-score was used to select and refit the best model**.

The tuned results were:

| Model | Search Time (s) | Best CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 14.9345 | 0.2556 | 0.5844 | 0.1581 | 0.6218 | 0.2521 |
| KNN | 212.8179 | 0.1399 | 0.8683 | 0.2671 | 0.0970 | 0.1423 |
| Decision Tree | 7.9630 | 0.2525 | 0.6506 | 0.1708 | 0.5453 | 0.2602 |
| SVM | 485.6427 | 0.2616 | 0.6066 | 0.1658 | 0.6185 | 0.2616 |

Execution times are hardware-dependent and should be interpreted comparatively within this run.

## Conclusions

The tuned SVM achieved the highest test F1-score, **0.2616**, but required by far the longest search time. The Decision Tree achieved an almost identical F1-score, **0.2602**, in approximately eight seconds, making it the strongest trade-off between positive-class performance and computational cost.

Logistic Regression achieved the highest recall, identifying approximately **62.2%** of actual subscribers. However, its precision of **15.8%** indicates that many customers predicted to subscribe were false positives. It may still be useful when the bank places greater value on identifying as many potential subscribers as possible.

KNN achieved the highest tuned test accuracy, **86.8%**, but detected only **9.7%** of actual subscribers. Its low F1-score shows that it remained strongly biased toward the majority class and is not the preferred model for this business objective.

Overall, the low precision and F1-scores indicate that the seven bank-client features alone provide limited predictive power. The Decision Tree is the preferred model among those tested when balancing performance and computation, while the final model choice should depend on the business cost of missed subscribers versus unnecessary calls.

## Recommendations and Next Steps

Future work should:

1. Add relevant campaign and economic-context features while continuing to exclude `duration` from pre-call prediction.
2. Evaluate probability-threshold adjustment instead of relying only on the default 0.50 threshold.
3. Compare models using precision-recall curves and PR-AUC.
4. Define a business-specific cost function for unnecessary calls and missed subscribers.
5. Explore additional imbalance-handling approaches, such as class weighting or carefully applied resampling.
6. Validate the selected model on new campaign data before operational deployment.

## Repository Structure

```text
.
├── README.md
├── prompt_III.ipynb
└── data/
    └── bank-additional-full.csv
```

## Technologies

- Python
- pandas
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

## Running the Notebook

1. Clone or download the repository.
2. Place `bank-additional-full.csv` in the `data/` directory.
3. Install the required packages:

```bash
pip install pandas matplotlib seaborn scikit-learn jupyter
```

4. Launch Jupyter Notebook:

```bash
jupyter notebook
```

5. Open and run `prompt_III.ipynb` from top to bottom.
