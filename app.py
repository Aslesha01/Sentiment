from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home_page():
    return render_template("sentiment.html")


from sent import vec


@app.route("/predict", methods=["POST", "GET"])
def predict():
    review = request.form.get("Review")
    req = [review]

    rev_vec = vec.transform(req)
    prediction = model.predict(rev_vec)[0]

    if prediction == 1:
        return render_template("sentiment.html", pred="The review is Positive")
    else:
        return render_template("sentiment.html", pred="The review is Negative")


if __name__ == "__main__":
    app.run()
