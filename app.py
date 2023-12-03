from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("models/model_perceptron.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    return render_template("index.html", predictions = 2)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)