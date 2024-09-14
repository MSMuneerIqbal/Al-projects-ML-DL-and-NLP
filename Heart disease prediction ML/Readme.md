# Heart Disease Prediction Project

### Predicting Stages of Heart Disease Using Machine Learning

## Project Overview
This project is an AI-powered application designed to predict heart disease severity based on key medical features provided by the user. The model classifies heart disease into five distinct stages:
- **0**: No Heart Disease
- **1**: Mild Heart Disease
- **2**: Moderate Heart Disease
- **3**: Severe Heart Disease
- **4**: Critical Heart Disease

The project uses machine learning to assist in early detection and classification of heart disease, enabling users to receive timely health insights.

---

## Features
- **User-Friendly Interface**: Built using Streamlit, the application offers a simple and interactive interface for entering medical information.
- **Heart Disease Prediction**: Based on the user input, the application predicts one of the five heart disease stages.
- **Visual Output**: Displays the predicted result in an intuitive and visually appealing format.
- **Data Input**: The application takes the following input features from users:
  - Age
  - Sex
  - Chest Pain Type
  - Resting Blood Pressure
  - Cholesterol Level
  - Fasting Blood Sugar
  - Resting Electrocardiographic Results
  - Maximum Heart Rate Achieved
  - Exercise-Induced Angina
  - ST Depression Induced by Exercise
  - Slope of Peak Exercise ST Segment
  - Number of Major Vessels
  - Thalassemia

---

## Technologies Used
- **Python**: For data processing, machine learning, and backend logic.
- **Streamlit**: Used for building the web interface.
- **Scikit-Learn**: Used for training the machine learning model.
- **Pandas**: For handling data input and transformation.
- **Pickle**: For loading the pre-trained model.
- **Matplotlib**: For visualizing results (if required for future features).

---

## How to Run the Project

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or above
- The required Python libraries (listed in the `requirements.txt` file)

### Installation Steps
1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/heart-disease-prediction.git
    ```
2. **Navigate to the project directory**:
    ```bash
    cd heart-disease-prediction
    ```
3. **Create and activate a virtual environment**:
    ```bash
    python -m venv env
    source env/bin/activate  # For Windows: env\Scripts\activate
    ```
4. **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Run the application**:
    ```bash
    streamlit run app.py
    ```

6. **Access the application**: Open your browser and go to `http://localhost:8501/`.

---

## Input Features Description
The application takes the following medical parameters to predict heart disease stages:
- **Age**: The age of the patient.
- **Sex**: Gender (Male/Female).
- **Chest Pain Type**: Four types of chest pain (Typical Angina, Atypical Angina, Non-Anginal, Asymptomatic).
- **Resting Blood Pressure**: Measured in mmHg.
- **Cholesterol Level**: Cholesterol levels in mg/dl.
- **Fasting Blood Sugar**: Whether fasting blood sugar > 120 mg/dl (True/False).
- **Resting Electrocardiographic Results**: The results of the electrocardiogram (Normal, ST-T abnormality, LV hypertrophy).
- **Maximum Heart Rate Achieved**: Heart rate during exercise.
- **Exercise-Induced Angina**: Whether the patient experiences chest pain during exercise (Yes/No).
- **ST Depression Induced by Exercise**: ST depression relative to rest.
- **Slope of Peak Exercise ST Segment**: The slope of the peak exercise segment (Upsloping, Flat, Downsloping).
- **Number of Major Vessels**: The number of major vessels colored by fluoroscopy.
- **Thalassemia**: A blood disorder, with categories (Normal, Fixed Defect, Reversible Defect).

---

## Model Description
The trained model used in this application is a **Gradient Boosting Classifier**, trained on the UCI Heart Disease Dataset. It has been fine-tuned to classify heart disease into five stages. The model was serialized using **Pickle** for efficient loading and prediction.

### Dataset
The **UCI Heart Disease Dataset** was used for training the model. This dataset contains the medical history and diagnostic information of patients, which has been used to train the model to predict heart disease stages.

---

## File Structure
- `app.py`: The main Streamlit application script.
- `heart_disease_model2.pkl`: The pre-trained machine learning model.
- `heart_disease_uci.csv`: The dataset used for generating the input feature options.
- `requirements.txt`: List of all dependencies required to run the project.

---

## Future Improvements
- **More Detailed Prediction**: The model can be enhanced to offer additional insights, such as risk factors or personalized recommendations based on prediction results.
- **Expanded Input Fields**: Additional medical factors like family history or lifestyle habits could improve the prediction accuracy.
- **Visualization**: Graphical results showing the probability of each heart disease stage or trends over time.

---

## Contributing
If you'd like to contribute to this project, please fork the repository and submit a pull request. For significant changes, please open an issue to discuss what you'd like to change.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Acknowledgements
- The UCI Machine Learning Repository for providing the heart disease dataset.
- Streamlit for making interactive and user-friendly web applications possible.

---

## Contact
For any questions or inquiries, please reach out:
- **Email**: your-email@example.com
- **GitHub**: [your-github-profile](https://github.com/your-profile)

---

