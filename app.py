from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pickle
flask_app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@flask_app.route("/")
def Home():
    return render_template("index.html")
if __name__ == "__main__":
    flask_app.run(debug=True)