import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# -------------------------------
# Robust Model Path (Production Safe)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "hypertension_model.pkl")

# Load model safely
model = pickle.load(open(model_path, "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form["age"])
        gender = int(request.form["gender"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        ap_hi = int(request.form["ap_hi"])
        ap_lo = int(request.form["ap_lo"])
        cholesterol = int(request.form["cholesterol"])
        glucose = int(request.form["glucose"])
        smoke = int(request.form["smoke"])
        alcohol = int(request.form["alcohol"])
        active = int(request.form["active"])

        input_data = np.array([[age, gender, height, weight,
                                ap_hi, ap_lo,
                                cholesterol, glucose,
                                smoke, alcohol, active]])

        prediction = model.predict(input_data)[0]

        if prediction == 0:
            result = "Low Risk"
            badge = "success"
        elif prediction == 1:
            result = "High Risk"
            badge = "danger"
        else:
            result = "Moderate Risk"
            badge = "warning"

        return render_template("result.html",
                               prediction_text=result,
                               badge_color=badge)

    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)