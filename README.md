# Hybrid CNN-LSTM for Predictive Maintenance

An end-to-end Predictive Maintenance system that estimates the **Remaining Useful Life (RUL)** of turbofan engines using a **Hybrid CNN-LSTM with Statistical Feature Fusion** trained on the NASA C-MAPSS FD001 dataset.

The project combines deep learning, feature engineering, interactive visualization, and web-based interfaces to provide an end-to-end RUL prediction workflow.

---

# Overview

Unexpected machine failures lead to costly downtime and maintenance expenses.

This project predicts the Remaining Useful Life (RUL) of aircraft turbofan engines using multivariate sensor data and deep learning models.

Unlike traditional implementations, this project includes:

- Hybrid CNN-LSTM architecture
- Statistical feature fusion
- Interactive Streamlit dashboard
- React frontend
- Flask backend APIs
- Model optimization
- Ablation study
- Single engine prediction
- Batch evaluation
- Performance visualization

---

# Features

- Hybrid CNN-LSTM architecture
- Statistical feature fusion
- Remaining Useful Life prediction
- NASA C-MAPSS FD001 dataset support
- Interactive Streamlit dashboard
- React-based web interface
- Flask backend APIs
- Single engine prediction
- Batch engine evaluation
- Model comparison
- Ablation study
- Performance visualization
- Model upload support

---

# Project Architecture

```
Sensor Data
      │
      ▼
Data Preprocessing
      │
Sliding Window Generation
      │
      ▼
CNN Feature Extraction
      │
      ▼
LSTM Temporal Learning
      │
      ▼
Statistical Feature Fusion
      │
      ▼
Dense Layers
      │
      ▼
Remaining Useful Life Prediction
```

---

# Project Structure

```
Predictive-Maintenance-LSTM/

│── frontend/
│── app_streamlit.py
│── backend.py
│── case_study.py
│── evaluate_model.py
│── hybrid_model.py
│── model_train.py
│── model_optimize.py
│── ablation_study.py
│── SlidingWindow.py
│── requirements.txt
│── train_FD001.txt
│── test_FD001.txt
│── RUL_FD001.txt
```

---

# Tech Stack

### Languages

- Python
- JavaScript

### Machine Learning

- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn

### Backend

- Flask
- Flask-CORS

### Frontend

- React
- Vite

### Visualization

- Matplotlib
- Streamlit

---

# Dataset

**NASA C-MAPSS FD001**

The project uses the publicly available NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset for Remaining Useful Life prediction.

---

# Model Architecture

The proposed architecture combines:

- CNN for local feature extraction
- LSTM for temporal sequence learning
- Statistical feature fusion
- Dense regression layers

This hybrid approach improves prediction performance while maintaining lightweight inference suitable for practical predictive maintenance applications.

---

# Dashboard

The project includes a full interactive Streamlit dashboard supporting:

- Model upload
- Scaler upload
- Single engine prediction
- Batch evaluation
- Prediction visualization
- Sensor trend visualization
- Performance metrics

> Dashboard screenshots will be added below.

---

# Results

The Hybrid CNN-LSTM model was evaluated using the NASA C-MAPSS FD001 dataset.

Evaluation metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Prediction plots and evaluation graphs are generated automatically through the dashboard.

---

# Model Comparison

The project includes comparisons between multiple architectures:

- CNN
- LSTM
- CNN-LSTM
- Hybrid CNN-LSTM with Feature Fusion

An ablation study is included to evaluate the contribution of each component.

---

# Installation

Clone the repository

```bash
git clone https://github.com/zoro7204/Predictive-Maintenance-LSTM.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Run Streamlit Dashboard

```bash
streamlit run app_streamlit.py
```

---

# Run Flask Backend

```bash
python backend.py
```

---

# Run React Frontend

```bash
cd frontend
npm install
npm run dev
```

---

# Future Improvements

- Transformer-based RUL prediction
- Explainable AI (XAI)
- Real-time IoT integration
- Docker deployment
- Cloud deployment
- Multi-engine monitoring dashboard

---

# Author

**Veeresh**

GitHub

https://github.com/zoro7204

LinkedIn

https://linkedin.com/in/veeresh-k7204

---

# License

This project is licensed under the MIT License.
