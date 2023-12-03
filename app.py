from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("models/model_perceptron.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = request.form['Age']
    sex = request.form['Sex']
    physActivity = request.form['PhysActivity']
    fruits = request.form['Fruits']
    veggies = request.form['Veggies']
    hvyAlcoholConsump = request.form['HvyAlcoholConsump']
    smoker = request.form['Smoker']
    highBP = request.form['HighBP']
    highChol = request.form['HighChol']
    bMI = request.form['BMI']
    genHlth = request.form['GenHlth']
    physHlthad = request.form['PhysHlth']
    diffWalk = request.form['DiffWalk']
    heartDiseaseorAttack = request.form['HeartDiseaseorAttack']
    arr = np.array([[age, sex, physActivity, fruits, veggies, hvyAlcoholConsump, smoker, highBP, highChol, bMI, genHlth, physHlthad, diffWalk, heartDiseaseorAttack]], dtype=np.float64)
    pred = model.predict(arr)
    return render_template("index.html", predictions = pred)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)