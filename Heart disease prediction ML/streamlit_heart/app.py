import streamlit as st
import pickle
import pandas as pd

# Load the trained model from the pickle file
model = pickle.load(open('./heart_disease_model2.pkl', 'rb'))

# Function to preprocess input features
def preprocess_input(features):
    # Convert categorical variables to numerical values
    features['sex'] = 1 if features['sex'] == 'Male' else 0
    features['cp'] = {'typical angina': 0, 'atypical angina': 1, 'non-anginal': 2, 'asymptomatic': 3}.get(features['cp'], -1)
    features['fbs'] = int(features['fbs'])
    
    restecg_mapping = {'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2}
    features['restecg'] = restecg_mapping.get(features['restecg'], -1)
    
    features['thalach'] = int(features['thalach'])
    features['exang'] = int(features['exang'])
    features['oldpeak'] = float(features['oldpeak'])
    
    slope_mapping = {'upsloping': 0, 'flat': 1, 'downsloping': 2}
    features['slope'] = slope_mapping.get(features['slope'], -1)
    
    features['ca'] = int(features['ca'])
    
    thal_mapping = {'normal': 0, 'fixed defect': 1, 'reversible defect': 2}
    features['thal'] = thal_mapping.get(features['thal'], -1)

    return list(features.values())

# Function to predict heart disease based on input features
def predict_heart_disease(features):
    prediction = model.predict([features])
    disease_mapping = {
        0: "No Heart Disease",
        1: "Mild Heart Disease",
        2: "Moderate Heart Disease",
        3: "Severe Heart Disease",
        4: "Critical Heart Disease"
    }
    prediction_label = disease_mapping.get(prediction[0], "Unknown")
    return prediction_label

# Streamlit app code
def main():
    # Set app title and heading in the center
    st.markdown("""
        <style>
        .title {
            text-align: center;
            font-weight: bold;
            color: #FF0000; /* Specify the desired color */
            font-size: 40px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Display a centered heading
    st.markdown('<p class="title">AI-Powered Heart Disease Diagnosis</p>', unsafe_allow_html=True)

    # Display the heart image using st.image
    st.image('./heart-img2.jpg', caption="Heart Health", use_column_width=True)

    # Move input fields to the sidebar
    st.sidebar.header('Input Features')
    
    dataset = pd.read_csv('./heart_disease_uci.csv')
    dataset = dataset.drop(columns=['dataset'])

    age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=25)
    sex = st.sidebar.selectbox("Sex", dataset['sex'].unique())
    cp = st.sidebar.selectbox("Chest Pain Type", dataset['cp'].unique())
    trestbps = st.sidebar.number_input("Resting Blood Pressure", min_value=0, max_value=300, value=120)
    chol = st.sidebar.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar", dataset['fbs'].unique())
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", dataset['restecg'].unique())
    thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=300, value=150)
    exang = st.sidebar.selectbox("Exercise-Induced Angina", dataset['exang'].unique())
    oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", value=0.0)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", dataset['slope'].unique())
    ca = st.sidebar.number_input("Number of Major Vessels", min_value=0, max_value=3, value=0)
    thal = st.sidebar.selectbox("Thalassemia", dataset['thal'].unique())

    # Create a dictionary of input features
    features = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'num': 0  # Placeholder value for the target variable
    }

    # Preprocess the input features
    preprocessed_features = preprocess_input(features)

    # Perform prediction and show result in the center
    if st.sidebar.button("Predict"):
        prediction_label = predict_heart_disease(preprocessed_features)
        st.markdown(f"<p style='font-size:35px;font-weight:bold;text-align:center;color: #FF0000;'>Result :: {prediction_label}</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
