# AI-Based Pneumonia Detection from X-ray Images

This project is a deep learning application that uses a Convolutional Neural Network (CNN) to detect pneumonia in chest X-ray images. The model is built using TensorFlow, and the user interface is implemented with Streamlit for easy interaction.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [How to Run the App](#how-to-run-the-app)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to accurately detect pneumonia in chest X-ray images using a deep learning model. The model is trained to classify X-ray images into two categories:
- **Pnemonia Not Detected**
- **Pneumonia Detected**

This project leverages CNNs, which are known for their effectiveness in image classification tasks. The Streamlit web app allows users to upload X-ray images and get real-time predictions.

## Dataset
The dataset used for training, testing, and evaluation of the model is sourced from [Kaggle](https://www.kaggle.com), which contains a large collection of labeled chest X-ray images.

## Model Architecture
The model is a Convolutional Neural Network (CNN) with the following layers:
- Convolutional layers
- MaxPooling layers
- Fully connected layers
- Softmax output layer for binary classification

The model was trained using the Adam optimizer and binary cross-entropy loss.

## Technologies Used
- **TensorFlow**: For building and training the CNN model.
- **Streamlit**: For creating the web-based user interface.
- **Pandas**: For handling data preprocessing.
- **NumPy**: For numerical operations.
- **PIL**: For image processing.

## Setup and Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/MSMuneerIqbal/Ai-projects-ML-DL-NLP-GenAI/tree/main/Lungs-Pnemonia-Classification-project
    ```
2. Navigate to the project directory:
    ```bash
    cd pneumonia-detector
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the App
1. Ensure you have the trained model (`model.h5`) in the project folder.
2. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3. Upload a chest X-ray image in the Streamlit interface and click "Predict" to get the result.

## Results
The CNN model achieves high accuracy in detecting pneumonia from X-ray images. Predictions are displayed along with a confidence score, indicating the model's certainty in its classification.

## Contributing
Contributions to improve the model or the app interface are welcome! Feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License.
