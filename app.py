from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        final_input = np.array(data).reshape(1, -1)
        scaled = scaler.transform(final_input)

        prob = model.predict_proba(scaled)[0][1]

        if prob < 0.3:
            stage = "Low Risk"
            diet = "Balanced diet"
            exercise = "Walk daily"
        elif prob < 0.6:
            stage = "Prediabetic"
            diet = "Reduce sugar"
            exercise = "Jogging"
        else:
            stage = "High Risk"
            diet = "Strict diet + doctor"
            exercise = "Daily exercise"

        return render_template("result.html",
                               probability=round(prob,2),
                               stage=stage,
                               diet=diet,
                               exercise=exercise)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)