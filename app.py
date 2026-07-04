import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.title("🧬 BioTrial Finder - Prediction & Gap Analysis")

# Data load karna
df = pd.read_csv('final_cleaned_trials (2).xls')

# Target banana
success_labels = ['COMPLETED']
fail_labels = ['TERMINATED', 'WITHDRAWN', 'SUSPENDED']
df_pred = df[df['Study Status'].isin(success_labels + fail_labels)].copy()
df_pred['target'] = df_pred['Study Status'].apply(lambda x: 1 if x in success_labels else 0)

# Features ready karna
features = ['primary_phase', 'Enrollment', 'disease_category', 'sponsor_type', 'duration_days']
X = df_pred[features].copy()
y = df_pred['target']
X['Enrollment'] = X['Enrollment'].fillna(X['Enrollment'].median())
X['duration_days'] = X['duration_days'].fillna(X['duration_days'].median())
X['primary_phase'] = X['primary_phase'].fillna('UNKNOWN')
X['disease_category'] = X['disease_category'].fillna('UNKNOWN')
X['sponsor_type'] = X['sponsor_type'].fillna('UNKNOWN')

encoders = {}
for col in ['primary_phase', 'disease_category', 'sponsor_type']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Model train karna
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

st.header("Naya Trial Check Karo")

# Website pe input boxes
phase = st.selectbox("Phase chuno", df['primary_phase'].dropna().unique())
enrollment = st.number_input("Enrollment (patients)", min_value=1, value=100)
disease = st.selectbox("Disease Category chuno", df['disease_category'].dropna().unique())
sponsor = st.selectbox("Sponsor Type chuno", df['sponsor_type'].dropna().unique())
duration = st.number_input("Duration (days)", min_value=1, value=180)

if st.button("Predict Karo"):
    new_trial = {
        'primary_phase': phase,
        'Enrollment': enrollment,
        'disease_category': disease,
        'sponsor_type': sponsor,
        'duration_days': duration
    }
    new_df = pd.DataFrame([new_trial])
    for col in ['primary_phase', 'disease_category', 'sponsor_type']:
        le = encoders[col]
        try:
            new_df[col] = le.transform(new_df[col].astype(str))
        except:
            new_df[col] = 0

    prediction = model.predict(new_df[features])
    probability = model.predict_proba(new_df[features])

    if prediction[0] == 1:
        st.success(f"✅ SUCCESS predicted! Confidence: {round(max(probability[0])*100,2)}%")
    else:
        st.error(f"❌ FAILURE predicted! Confidence: {round(max(probability[0])*100,2)}%")
