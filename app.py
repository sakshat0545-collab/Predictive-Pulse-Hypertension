# ==============================================
# Predictive Pulse - Professional Flask App
# Modern UI + Validation + Risk Color Logic
# ==============================================

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("models/hypertension_model.pkl", "rb"))


# -------------------------------
# Input Validation Function
# -------------------------------
def validate_inputs(age, ap_hi, ap_lo):
    if age < 18 or age > 100:
        return "Age must be between 18 and 100."

    if ap_hi < 70 or ap_hi > 250:
        return "Systolic BP must be between 70 and 250."

    if ap_lo < 40 or ap_lo > 150:
        return "Diastolic BP must be between 40 and 150."

    if ap_hi <= ap_lo:
        return "Systolic BP must be greater than Diastolic BP."

    return None


# -------------------------------
# Home Route
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------
# Prediction Route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect Form Data
        age = float(request.form["age"])
        gender = int(request.form["gender"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        ap_hi = float(request.form["ap_hi"])
        ap_lo = float(request.form["ap_lo"])
        cholesterol = int(request.form["cholesterol"])
        gluc = int(request.form["gluc"])
        smoke = int(request.form["smoke"])
        alco = int(request.form["alco"])
        active = int(request.form["active"])

        # Validate Inputs
        validation_error = validate_inputs(age, ap_hi, ap_lo)

        if validation_error:
            return render_template(
                "result.html",
                prediction_text="Invalid Input",
                probability="N/A",
                recommendation=validation_error,
                risk_color="#e74c3c"
            )

        # Prepare Input Array
        input_data = np.array([[age, gender, height, weight,
                                ap_hi, ap_lo, cholesterol,
                                gluc, smoke, alco, active]])

        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        probability_percent = round(probability * 100, 2)

        # Risk Interpretation
        if probability_percent < 40:
            risk_level = "Low Risk"
            recommendation = "Maintain a healthy lifestyle and continue regular health monitoring."
            risk_color = "#2ecc71"  # Green

        elif 40 <= probability_percent < 70:
            risk_level = "Moderate Risk"
            recommendation = "Lifestyle modifications are recommended. Consider consulting a healthcare professional."
            risk_color = "#f39c12"  # Orange

        else:
            risk_level = "High Risk"
            recommendation = "Immediate medical consultation is strongly advised."
            risk_color = "#e74c3c"  # Red

        return render_template(
            "result.html",
            prediction_text=risk_level,
            probability=str(probability_percent) + "%",
            recommendation=recommendation,
            risk_color=risk_color
        )

    except Exception as e:
        return f"Error occurred: {e}"


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)