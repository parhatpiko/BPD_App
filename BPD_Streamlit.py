import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt


# Load the random forest model
model = joblib.load('RF_BPD.pkl')

# Load the explainer object from the file
with open('explainer_BPD.pkl', 'rb') as f:
    explainer = pickle.load(f)

# Load the scaler
scaler = joblib.load('Scaler_BPD.pkl')

# Define feature names
feature_names = [
    'GA', 'Weight', 'Hct', 'Lymphcytes', 'CRP', 'K', 'ALB', 'CK', 'TC', 'Creatinine'
]

# Streamlit user interface
st.title("Bronchopulmonary Dysplasia (BPD) Risk Predictor")

# Define input fields for each feature
inputs = {}
for feature in feature_names:
    if feature == 'GA':
        inputs[feature] = st.number_input(f"{feature} (Gestational Age):", min_value=0.0, max_value=50.0)
    elif feature == 'Weight':
        weight_value = st.number_input(f"{feature} (grams):", min_value=0, max_value=5000)
        inputs[feature] = 0 if weight_value >= 1000 else 1
    elif feature == 'Hct':
        inputs[feature] = st.number_input(f"{feature} (Hematocrit):", min_value=0.0, max_value=100.0)
    elif feature == 'Lymphcytes':
        inputs[feature] = st.number_input(f"{feature}:", min_value=0.0, max_value=100.0)
    elif feature == 'CRP':
        inputs[feature] = st.number_input(f"{feature} (C-Reactive Protein):", min_value=0.0, max_value=100.0)
    elif feature == 'K':
        inputs[feature] = st.number_input(f"{feature} (Potassium):", min_value=0.0, max_value=10.0)
    elif feature == 'ALB':
        inputs[feature] = st.number_input(f"{feature} (Albumin):", min_value=0.0, max_value=100.0)
    elif feature == 'CK':
        inputs[feature] = st.number_input(f"{feature} (Creatine Kinase):", min_value=0.0, max_value=10000.0)
    elif feature == 'TC':
        inputs[feature] = st.number_input(f"{feature} (Total Cholesterol):", min_value=0.0, max_value=400.0)
    elif feature == 'Creatinine':
        inputs[feature] = st.number_input(f"{feature}:", min_value=0.0, max_value=10.0)

# Process inputs and make predictions
feature_values = [inputs[feature] for feature in feature_names]
# Convert to DataFrame for scaling
features_df = pd.DataFrame([feature_values], columns=feature_names)
# Define which features are continuous
continuous_variables = [
    'GA', 'Hct', 'Lymphcytes', 'CRP', 'K', 'ALB', 'CK', 'TC', 'Creatinine'
]
# Scale continuous features
features_df[continuous_variables] = scaler.transform(features_df[continuous_variables])

# Convert scaled DataFrame to numpy array for prediction
features = features_df.values
class_name = ['Negative Outcome', 'Positive Outcome']
if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    # Display prediction results
    st.write(f"**Predicted Class:** {class_name[predicted_class]}")
    st.write(f"**Prediction Probabilities:** {predicted_proba[predicted_class] * 100:.1f}%")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of Bronchopulmonary Dysplasia (BPD). "
            f"The model predicts that your probability of health risk is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a healthcare professional for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of Bronchopulmonary Dysplasia (BPD). "
            f"The model predicts that your probability of not having health issues is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)
    # Calculate SHAP values for both classes
    shap_values = explainer.shap_values(features_df)[0]
    combined_shap_values = np.vstack((shap_values[:, 0], shap_values[:, 1]))
    # Display SHAP force plot for each class
    for which_class in range(2):
        st.subheader(f"SHAP Force Plot for Class {class_name[which_class]}")
        plt.figure()
        shap.force_plot(
            base_value=explainer.expected_value[which_class],
            shap_values=combined_shap_values[which_class],  # Correctly index for the first instance
            features=features_df,             # Use the first instance
            feature_names=feature_names,
            matplotlib=True,
            show=False,
            text_rotation=30
        )
        st.pyplot(plt.gcf())
        plt.close()

        st.subheader(f"SHAP Waterfall Plot for Class {class_name[which_class]}")
        plt.figure()
        shap.waterfall_plot(shap.Explanation(values=combined_shap_values[which_class],
                                             base_values=explainer.expected_value[which_class],
                                             data=features_df.iloc[0].squeeze(),  # added this line
                                             feature_names=feature_names), show=False)
        st.pyplot(plt.gcf())
        plt.close()