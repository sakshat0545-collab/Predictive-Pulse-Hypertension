# 🩺 Predictive Pulse – Hypertension Risk Assessment System

An AI-powered Hypertension Risk Prediction System built using Machine Learning and Flask.  
This project analyzes patient health parameters and predicts cardiovascular disease risk.

---

## 🚀 Live Demo
https://predictive-pulse-hypertension.onrender.com

---

## 📌 Project Overview

Predictive Pulse is a supervised machine learning-based web application that predicts the risk of cardiovascular disease using clinical and lifestyle features such as:

- Age
- Gender
- Height
- Weight
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Cholesterol Level
- Glucose Level
- Smoking Habit
- Alcohol Consumption
- Physical Activity

The system classifies risk into:

- 🟢 Low Risk  
- 🟡 Moderate Risk  
- 🔴 High Risk  

---

## 🧠 Machine Learning Approach

### Dataset
- Cardiovascular Disease Dataset (70,000 records)
- 12 processed features
- Binary target variable (`cardio`)

### Preprocessing
- Removed invalid BP values
- Converted age from days to years
- Feature scaling using `StandardScaler`
- Train-Test Split (80-20)

### Models Compared

| Model | Accuracy |
|-------|----------|
| Logistic Regression (Scaled) | **72.4%** |
| Random Forest | 71.2% |
| Decision Tree | 62.8% |

✔ Logistic Regression (Scaled) selected as final model.

---

## 🏗 Project Structure

```
Predictive-Pulse-Hypertension/
│
├── models/                # Saved ML model
├── templates/             # HTML templates
│   ├── index.html
│   └── result.html
├── app.py                 # Flask application
├── train.py               # Model training script
├── requirements.txt       # Dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Installation (Local Setup)

1. Clone repository:
```
git clone https://github.com/sakshat0545-collab/Predictive-Pulse-Hypertension.git
cd Predictive-Pulse-Hypertension
```

2. Create virtual environment:
```
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Run application:
```
python app.py
```

5. Open browser:
```
http://127.0.0.1:5000
```

---

## 🌐 Deployment

This project is deployable on:
- Render
- Railway
- PythonAnywhere
- Heroku (legacy)

Start command:
```
gunicorn app:app
```

---

## 📊 Features

✔ Clean Professional UI  
✔ ML Pipeline Implementation  
✔ Risk Level Classification  
✔ Model Comparison  
✔ Deployment Ready  
✔ Scalable Structure  

---

## ⚠ Disclaimer

This system is for educational and decision-support purposes only.  
It is NOT a medical diagnosis tool.

---

## 👨‍💻 Author

Sakshat Shrivatra  
B.Tech CSE – AI/ML  
SmartBridge TNP Project

---

## ⭐ Future Improvements

- Probability confidence score
- BMI auto-calculation
- Model explainability (SHAP)
- Database integration
- User authentication
- REST API version
- Cloud scaling architecture

---

If you find this project helpful, consider giving it a ⭐ on GitHub.
