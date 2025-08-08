# AutoML App (Advanced Edition)

**AutoML** is a no-code Streamlit web app for end-to-end statistical machine learning. Upload any CSV dataset, perform preprocessing, select features, train models, view performance metrics, and download the trained model — all in a simple, interactive UI.

## Features

- File upload and dataset preview  
- Missing value imputation  
- Encoding and feature scaling  
- Model selection: Logistic Regression, Random Forest, XGBoost, etc.  
- Performance metrics, confusion matrix, ROC curve  
- SHAP-based model explainability  
- Cross-validation (5-fold)  
- Export trained model as `.pkl`

## Supported Models

- Logistic Regression  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- XGBoost  
- Voting Ensemble

## How to Run

```bash
pip install -r requirements.txt
streamlit run ultimate_automl_app_advanced.py


