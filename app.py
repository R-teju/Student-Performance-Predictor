from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    studytime = float(request.form["studytime"])
    failures = float(request.form["failures"])
    absences = float(request.form["absences"])
    G2 = float(request.form["G2"])

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

    return render_template(
        "result.html",
        result=result,
        probability=f"{probability:.2f}",
        risk=risk
    )

if __name__ == "__main__":
    app.run(debug=True)