# Credit Risk ML System

An end-to-end machine learning project that predicts whether a loan applicant is likely to **default**. It covers data cleaning, model training and tuning, probability calibration, threshold selection, model explainability (SHAP), and deployment as a **FastAPI** service with a web interface.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7)

**Live demo:** https://credit-risk-ml-system-backend.onrender.com/

> The app runs on a free Render instance, so the first request after a period of inactivity can take up to a minute while the service wakes up.

<!-- Add a screenshot of the app here, for example: -->
<!-- ![App screenshot](assets/screenshot.png) -->

---

## Overview

Lenders need to estimate how likely an applicant is to default before approving a loan. This project takes an applicant's profile (income, employment, loan details, credit history) and returns:

- the **default probability**,
- a **High Risk / Low Risk** decision based on a tuned decision threshold.

## Features

- Cleaned and validated dataset with outlier handling
- Scikit-learn `Pipeline` + `ColumnTransformer` so preprocessing and model ship as one object
- Class imbalance handled with `scale_pos_weight`
- Logistic Regression baseline compared against XGBoost using stratified 5-fold cross-validation
- Hyperparameter tuning with `RandomizedSearchCV` (100 iterations)
- Probability calibration (sigmoid, 5-fold) so outputs behave like real probabilities
- Decision threshold chosen from the precision-recall curve to maximize F1
- SHAP analysis for global feature importance, plus a false positive / false negative review
- FastAPI backend with request validation (Pydantic) and a static HTML/CSS/JS frontend served from the same app

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.12 |
| Data & ML | pandas, NumPy, scikit-learn, XGBoost, SHAP |
| Visualization | Matplotlib, Seaborn |
| Backend | FastAPI, Uvicorn, Pydantic, joblib |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Render |

## Dataset

- **Source:** public credit risk dataset (`credit_risk_dataset.csv`)
- **Size:** 32,581 loan applications, 12 columns
- **Target:** `loan_status` (`1` = default, `0` = repaid), about 21.8% defaults, so the classes are imbalanced

**Cleaning steps**

| Step | Rows remaining |
|---|---|
| Original data | 32,581 |
| Remove duplicate rows | 32,416 |
| Keep ages between 18 and 100 | 32,411 |
| Remove employment length greater than age or above 60 years | 31,522 |

Missing values in `person_emp_length` and `loan_int_rate` are filled with the median inside the pipeline. The data was split 80/20 with stratification (6,305 test rows).

## Model Development

1. **Exploratory analysis:** target balance, distributions, box plots for outliers, correlation heatmap
2. **Validation and cleaning:** duplicates, impossible ages, unrealistic employment length
3. **Preprocessing:** median imputation for numeric features, constant imputation plus one-hot encoding for categorical features (with scaling for Logistic Regression only)
4. **Imbalance handling:** class weights for Logistic Regression, `scale_pos_weight` for XGBoost
5. **Model comparison:** stratified 5-fold cross-validation on the training set
6. **Tuning:** `RandomizedSearchCV` over `max_depth`, `learning_rate`, `subsample` and `gamma`, scored on average precision
7. **Calibration:** `CalibratedClassifierCV` (sigmoid, 5-fold) on the tuned XGBoost pipeline
8. **Threshold selection:** the F1-maximizing point on the precision-recall curve (about **0.57**)
9. **Interpretation:** SHAP summary plot and error analysis of false positives and false negatives

## Results

**Cross-validation (5-fold, training data)**

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.871 | 0.812 | 0.545 | 0.778 | 0.641 |
| XGBoost | 0.939 | 0.910 | 0.798 | 0.786 | 0.791 |

**Hold-out test set (6,305 applications)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.82 | 0.56 | 0.79 | 0.65 |
| XGBoost (tuned) | 0.93 | 0.85 | 0.80 | 0.82 |
| XGBoost (tuned, calibrated, threshold 0.57) | 0.94 | 0.95 | 0.76 | 0.84 |

The tuned XGBoost model clearly outperforms the baseline. The final calibrated model trades some recall for much higher precision, which means fewer good applicants are wrongly flagged as high risk.

## API

The backend loads the trained pipeline and threshold once at startup.

### `POST /predict`

**Request body**

```json
{
  "person_age": 30,
  "person_income": 60000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 10000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.17,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 6
}
```

**Response**

```json
{
  "default_probability": 0.02,
  "default_prediction": 0,
  "threshold": 0.57,
  "Result": "Low Risk"
}
```

**Input fields**

| Field | Type | Description |
|---|---|---|
| `person_age` | int | Applicant age in years |
| `person_income` | float | Annual income |
| `person_home_ownership` | str | `RENT`, `OWN`, `MORTGAGE` or `OTHER` |
| `person_emp_length` | float | Employment length in years |
| `loan_intent` | str | `PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT` or `DEBTCONSOLIDATION` |
| `loan_grade` | str | Loan grade `A` to `G` |
| `loan_amnt` | float | Requested loan amount |
| `loan_int_rate` | float | Interest rate in percent |
| `loan_percent_income` | float | Loan amount divided by income (ratio) |
| `cb_person_default_on_file` | str | Prior default on file: `Y` or `N` |
| `cb_person_cred_hist_length` | int | Credit history length in years |

Interactive API docs are available at `/docs` when the server is running.

## Project Structure

```
credit-risk-ml-system/
├── main.py                  # FastAPI app: loads model, /predict endpoint, serves frontend
├── Credit_Risk.ipynb        # Full analysis and model training notebook
├── credit_risk_model.pkl    # Trained, calibrated XGBoost pipeline
├── best_threshold.pkl       # Tuned decision threshold
├── requirements.txt
└── static/
    ├── index.html
    ├── style.css
    └── script.js
```

## Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn main:app --reload
```

Open http://127.0.0.1:8000 for the web interface, or http://127.0.0.1:8000/docs for the API docs.

> **Important:** the pickled model is sensitive to library versions. Install the same `scikit-learn` and `xgboost` versions the model was trained with (pinned in `requirements.txt`), otherwise loading can fail or give unreliable results.

## Deployment (Render)

1. Push the project to GitHub.
2. On Render, create a **Web Service** from the repository.
3. Set the build command to `pip install -r requirements.txt`.
4. Set the start command to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
5. Add the environment variable `PYTHON_VERSION` matching the version used for training (for example `3.12.10`).

Every push to the main branch redeploys automatically.

## Limitations and Future Work

- The decision threshold was selected on the same hold-out set used for the final evaluation, so the reported precision and recall for the final model may be slightly optimistic. A separate validation split (or nested cross-validation) would give a cleaner estimate.
- The dataset is a public benchmark, so this project is a demonstration and **must not be used for real lending decisions**.
- Planned improvements: SHAP explanations for individual predictions in the UI, Docker support, automated tests and CI, and input monitoring for data drift.

## Author

**[Yuvraj Singh Rathore]**
[LinkedIn](https://www.linkedin.com/in/yuvrajrathore54) · [GitHub](https://github.com/yuvrajrathore672)

## License

Released under the MIT License. Add a `LICENSE` file to the repository to make this official.
