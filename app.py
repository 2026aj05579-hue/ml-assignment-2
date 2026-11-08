import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

st.set_page_config(page_title="ML Classifier Evaluator", layout="wide")

st.title("🤖 Machine Learning Classifier Demonstration App")
st.write("Upload test data and select trained models to calculate evaluation metrics instantly.")

# Sidebar: Model Selection
model_option = st.sidebar.selectbox(
    "Select Classifier",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
)

# Dataset upload
uploaded_file = st.file_uploader("Upload your test dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
    st.dataframe(test_data.head())
    
    if "target" not in test_data.columns:
        st.error("The dataset must contain a column named 'target' as the class label.")
    else:
        X_test = test_data.drop(columns=["target"])
        y_test = test_data["target"]
        
        # Load selected model
        model_filename = f"model/{model_option.lower().replace(' ', '_')}.pkl"
        if os.path.exists(model_filename):
            model = joblib.load(model_filename)
            
            # Predict
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
            
            # Metrics
            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, probs)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            mcc = matthews_corrcoef(y_test, preds)
            
            # Layout for displaying metrics
            st.subheader(f"📊 Metrics for {model_option}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC", f"{mcc:.4f}")
            
            # Confusion Matrix & Classification Report
            st.subheader("📈 Classification Insights")
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.text("Classification Report:")
                st.text(classification_report(y_test, preds))
                
            with col_right:
                st.text("Confusion Matrix:")
                st.write(confusion_matrix(y_test, preds))
        else:
            st.error(f"Trained model file '{model_filename}' not found. Please train and save the model first.")
else:
    st.info("Please upload a test dataset CSV to begin evaluation.")
