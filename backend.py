# backend.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Placeholder for CSV path
CSV_PATH = 'your_dataset.csv'

# render home route
@app.route('/')
def home():
    return render_template('index.html')

# Load and train model
@app.route('/train', methods=['POST'])
def train_model():
    df = pd.read_csv(CSV_PATH)
    X = df.drop('target', axis=1) # Replace 'target' with your actual label column
    y = df['target']
    print("CSV loaded...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, 'model.pkl')
    acc = accuracy_score(y_test, model.predict(X_test))
    print("Model Trained...")
    return jsonify({'message': 'Model trained', 'accuracy': acc})

# Inference endpoint
@app.route('/predict', methods=['POST'])
def predict():

    input_data = request.json['features']
    model = joblib.load('model.pkl')
    prediction = model.predict([input_data])
    print({'prediction': prediction.tolist()})
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)