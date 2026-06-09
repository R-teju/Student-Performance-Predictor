from flask import Flask, render_template, request
import pickle
import pandas as pd
import json
import os

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load model metadata
metadata = {}
if os.path.exists("model_metadata.json"):
    try:
        with open("model_metadata.json", "r") as f:
            metadata = json.load(f)
    except Exception as e:
        print("Error loading model_metadata.json:", e)

# Load dataset statistics on startup
try:
    df = pd.read_csv("student-mat.csv", sep=";")
    dataset_stats = {
        "total_students": int(len(df)),
        "avg_absences": round(float(df["absences"].mean()), 2),
        "avg_g2": round(float(df["G2"].mean()), 2),
        "pass_rate": round(float((df["G3"] >= 10).mean() * 100), 2)
    }
except Exception as e:
    print("Error reading dataset stats:", e)
    dataset_stats = {
        "total_students": 395,
        "avg_absences": 5.71,
        "avg_g2": 10.71,
        "pass_rate": 67.09
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        stats=dataset_stats,
        metadata=metadata
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        studytime = float(request.form["studytime"])
        failures = float(request.form["failures"])
        absences = float(request.form["absences"])
        G2 = float(request.form["G2"])
    except (ValueError, KeyError):
        return render_template("index.html", error="Please provide valid inputs.")

    input_data = pd.DataFrame({
        "studytime": [studytime],
        "failures": [failures],
        "absences": [absences],
        "G2": [G2]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    result = "Pass" if prediction == 1 else "Fail"

    if probability >= 80:
        risk = "Low Risk"
    elif probability >= 50:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    # Actionable feedback logic
    recommendations = []
    if result == "Fail":
        if failures > 0:
            recommendations.append("Prior failures detected. Consider seeking active tutoring or peer-mentoring groups to target weak topics.")
        if absences > 5:
            recommendations.append(f"High absences ({int(absences)} days). Consistent attendance is crucial; try to keep absences below 5 days to stay on track.")
        if studytime < 2:
            recommendations.append("Very low weekly study time. Dedicating 3+ hours per week specifically for this subject is highly recommended.")
        if G2 < 10:
            recommendations.append(f"Midterm grade ({G2}/20) is low. Request extra study materials or consultations with your teacher to catch up.")
        if not recommendations:
            recommendations.append("Focus on systematic daily reviews and work closely with course instructors to raise test scores.")
    else:  # Pass
        recommendations.append("Keep up the solid work! Maintain your attendance and current study habits to secure your final grade.")
        if absences > 3:
            recommendations.append("Tip: Reducing absences further can boost your probability of achieving an excellent final grade.")
        if studytime < 3:
            recommendations.append("Tip: Increasing study hours slightly can help shift you from a moderate pass to a top-performing grade.")

    return render_template(
        "result.html",
        result=result,
        probability=f"{probability:.2f}",
        risk=risk,
        inputs={
            "studytime": studytime,
            "failures": failures,
            "absences": absences,
            "G2": G2
        },
        recommendations=recommendations
    )

if __name__ == "__main__":
    app.run(debug=True)