# ultimate_automl_app_advanced.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import io

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import shap
import optuna

#st.set_page_config(layout="wide")
st.title("AutoML App (Advanced Edition)")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head(5))
    st.dataframe(df.tail(5))




    #Show basic statistics;
    st.subheader("Basic Statistics")
    st.write(df.describe(include='all'))





    #Show NaN:
    st.subheader("Missing Values Summary")
    nan_summary = df.isna().sum()
    nan_summary = nan_summary[nan_summary > 0].sort_values(ascending=False)
    if not nan_summary.empty:
        st.write(nan_summary.to_frame("Missing Count: "))
    else:
        st.write("No missing values detected!")






    # Imputation
    st.subheader("Missing Values Imputation (If Any)")
    impute_option = st.selectbox("Choose Imputation Method", ["None", "Mean", "Median", "Most Frequent"])
    strategy = impute_option.lower().replace("most frequent", "most_frequent")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(num_cols):
        df[num_cols] = SimpleImputer(strategy=strategy).fit_transform(df[num_cols])
    if len(cat_cols):
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])





    # Encoding
    st.subheader("Encoding & Scaling")
    encode_cols = st.multiselect("Select columns to encode", cat_cols.tolist())
    for col in encode_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    scale_cols = st.multiselect("Select columns to scale", num_cols.tolist())
    scaler = st.selectbox("Select Scaling Method", ["None", "StandardScaler", "MinMaxScaler"])
    if scaler != "None" and scale_cols:
        scaler_cls = StandardScaler if scaler == "StandardScaler" else MinMaxScaler
        df[scale_cols] = scaler_cls().fit_transform(df[scale_cols])

    st.subheader("Transformed Data Preview")
    st.dataframe(df.head())




    # Distribution
    #st.subheader("Feature Distribution")
    #for col in num_cols:
       # fig, ax = plt.subplots()
       # sns.histplot(df[col], kde=True, ax=ax)
        #st.pyplot(fig)





    # Feature selection
    target  = st.selectbox("Select Target Column", df.columns)
    features = st.multiselect("Select Feature Columns", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target])

    if features and target:
        X, y = df[features], df[target]
        test_size = st.slider("Test Size", 0.1, 0.2, 0.3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model_name = st.selectbox("Select Model", [
            "Logistic Regression", "Random Forest", "Gradient Boosting", "SVM", "Naive Bayes", "KNN", "Decision Tree", "Voting Ensemble", "XGBoost"
        ])

        if model_name == "Voting Ensemble":
            clf1 = LogisticRegression(max_iter=1000)
            clf2 = RandomForestClassifier(n_estimators=100)
            clf3 = GradientBoostingClassifier()
            model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3)], voting='soft')
        elif model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "Random Forest":
            model = RandomForestClassifier()
        elif model_name == "Gradient Boosting":
            model = GradientBoostingClassifier()
        elif model_name == "SVM":
            model = SVC(probability=True)
        elif model_name == "Naive Bayes":
            model = GaussianNB()
        elif model_name == "KNN":
            model = KNeighborsClassifier()
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_name == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        st.subheader("Performance Metrics")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        if y_prob is not None and len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title("ROC Curve")
            ax.legend()
            st.pyplot(fig)




        # SHAP Explainability
        if model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            st.subheader("Feature Importance (SHAP)")
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
            st.pyplot(bbox_inches='tight')




        # Cross Validation
        st.subheader("Cross Validation")
        scores = cross_val_score(model, X, y, cv=5)
        st.write(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

        buffer = io.BytesIO()
        pickle.dump(model, buffer)
        st.download_button("Download Model", buffer, file_name="trained_model.pkl")
