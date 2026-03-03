import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model
model = pickle.load(open("models/best_model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
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

        features = np.array([[age, gender, height, weight, ap_hi,
                              ap_lo, cholesterol, gluc, smoke, alco, active]])

        prediction = model.predict(features)[0]

        if prediction == 0:
            result = "Low Risk"
            color = "green"
        else:
            result = "High Risk"
            color = "red"

        return render_template("result.html", prediction=result, color=color)

    except Exception:
        return render_template("result.html",
                               prediction="Invalid Input",
                               color="orange")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)