# Quantum-Enhanced Portfolio Management with Sentiment Analysis

This project demonstrates a hybrid quantum-classical machine learning approach for portfolio management, leveraging sentiment analysis and a Quantum Neural Network (QNN) to predict market trends and optimize investment decisions.

## Overview

This implementation combines:

1.  **Sentiment Analysis (Simulated):** We simulate sentiment data related to financial assets, representing market sentiment as numerical values. In a real-world scenario, this would be derived from news articles, social media, and financial reports.
2.  **Quantum Neural Network (QNN):** A hybrid QNN is built using PennyLane and TensorFlow/Keras to process and learn patterns from the combined sentiment and financial data.
3.  **Portfolio Management:** The QNN predictions are used to inform investment decisions, aiming to maximize returns and minimize risk.

## Key Components

* **Data Generation:**
    * Simulates financial data (time, amount) and associated sentiment data.
    * Labels are generated to represent market trends (e.g., 0 for negative, 1 for positive).
* **Data Preprocessing:**
    * Normalizes the financial data using `MinMaxScaler`.
    * Splits the dataset into training and testing sets.
* **Quantum Circuit:**
    * Implements amplitude encoding to map classical data into quantum states.
    * Utilizes a Quantum Fourier Transform (QFT) and parameterized quantum gates to process the quantum information.
    * Outputs expectation values that serve as features for the classical neural network.
* **Hybrid QNN Model:**
    * Combines a PennyLane QNode with a Keras neural network.
    * The QNode acts as a quantum feature extractor, and the Keras layers perform classical classification.
* **Model Training and Evaluation:**
    * Trains the hybrid model using `adam` optimizer and `binary_crossentropy` loss.
    * Evaluates the model's performance using classification reports, confusion matrices, and ROC curves.
* **Visualizations:**
    * Generates visualizations of the confusion matrix and ROC curve to assess model performance.

## Prerequisites

* Python 3.6+
* pip
* Libraries:
    * `numpy`
    * `pennylane`
    * `scikit-learn`
    * `matplotlib`
    * `seaborn`
    * `tensorflow`
    * `keras`

## Installation

1.  **Clone the repository (if applicable):**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install the required Python packages:**

    ```bash
    pip install numpy pennylane scikit-learn matplotlib seaborn tensorflow
    ```

## Usage

1.  **Run the Python script:**

    ```bash
    python your_script_name.py
    ```

2.  **Interpret the results:**
    * The script will output the classification report and confusion matrix, providing insights into the model's accuracy.
    * The confusion matrix and ROC curve visualizations will be displayed, allowing for a visual assessment of the model's performance.

## Code Explanation

* **Amplitude Encoding:** The `amplitude_encoding` function encodes the normalized financial data into the amplitudes of a quantum state.
* **Quantum Fourier Transform (QFT):** The `qft` function implements the QFT, a crucial quantum algorithm for signal processing.
* **Quantum Circuit:** The `quantum_circuit` QNode defines the quantum part of the hybrid model, processing the encoded data and outputting expectation values.
* **Hybrid Model:** The `build_qnn_model` function constructs the hybrid model using `qml.qnn.KerasLayer` to integrate the QNode into the Keras network.

## Future Enhancements

* **Real Sentiment Data:** Integrate real-time sentiment analysis from financial news and social media.
* **Portfolio Optimization:** Implement portfolio optimization algorithms to allocate investments based on QNN predictions.
* **Risk Management:** Incorporate risk management strategies to minimize potential losses.
* **Advanced Quantum Circuits:** Explore more complex quantum circuits to improve feature extraction.
* **Time Series Analysis:** Implement time series models to capture temporal dependencies in financial data.

## Note

This project is a starting point for exploring quantum-enhanced portfolio management. Real-world applications require careful consideration of data quality, model validation, and risk management.
