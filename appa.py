from flask import Flask, request, jsonify, render_template 
import pandas as pd
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from train import fine_tune, load_tokenizer  # load_tokenizer imported from train.py
from model import get_model

# app = Flask(__name__)
app = Flask(__name__, static_folder='static')

UPLOAD_PATH = "data/new_data.csv"

# Mapping for a three-class sentiment problem.
LABEL_MAPPING = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# --- Web UI Route ---
@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    statement = None

    if request.method == 'POST':
        statement = request.form['statement']
        sarcasm_flag = int(request.form.get('sarcasm', 0))

        tokenizer = load_tokenizer()
        if tokenizer is None:
            sentiment = "Tokenizer not found. Train the model first."
        else:
            text_sequence = tokenizer.texts_to_sequences([statement.lower()])
            padded_text = pad_sequences(text_sequence, maxlen=100, padding='post')
            sarcasm_input = np.array([[sarcasm_flag]])

            model = get_model(use_sarcasm=True)
            preds = model.predict([padded_text, sarcasm_input])
            pred_index = int(np.argmax(preds, axis=1)[0])
            sentiment = LABEL_MAPPING.get(pred_index, "unknown")

    return render_template('index.html', sentiment=sentiment, statement=statement)

# --- API Route for Programmatic Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    statement = data.get('statement')
    if statement is None or statement.strip() == "":
        return jsonify({"error": "No statement provided"}), 400

    sarcasm_flag = data.get('sarcasm', 0)

    tokenizer = load_tokenizer()
    if tokenizer is None:
        return jsonify({"error": "Tokenizer not found. Please train the model first."}), 500

    text_sequence = tokenizer.texts_to_sequences([statement.lower()])
    padded_text = pad_sequences(text_sequence, maxlen=100, padding='post')
    sarcasm_input = np.array([[sarcasm_flag]])

    model = get_model(use_sarcasm=True)
    preds = model.predict([padded_text, sarcasm_input])
    pred_index = int(np.argmax(preds, axis=1)[0])

    sentiment = LABEL_MAPPING.get(pred_index, "unknown")
    return jsonify({"statement": statement, "sentiment": sentiment})

# --- Upload Route ---
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    df = pd.read_csv(file)
    
    if os.path.exists(UPLOAD_PATH):
        df.to_csv(UPLOAD_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(UPLOAD_PATH, index=False)
    
    result = fine_tune()
    return jsonify({"message": result}), 200

if __name__ == '__main__':
    app.run(port=6060, debug=True)
