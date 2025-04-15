from flask import Flask, request, jsonify, render_template, url_for, redirect
import pandas as pd
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from train import fine_tune, load_tokenizer  # load_tokenizer imported from train.py
from model import get_model

app = Flask(__name__)
UPLOAD_PATH = "data/new_data.csv"

# Mapping for a three-class sentiment problem.
LABEL_MAPPING = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# --- Web UI Route for Prediction ---
@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    statement = None
    emoji_gif = None

    # Define emoji_map at the top
    emoji_map = {
        'positive': '/static/positive.gif',
        'negative': '/static/negative.gif',
        'neutral': '/static/neutral.gif',
        'unknown': '/static/neutral.gif'  # fallback
    }

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
            emoji_gif = emoji_map.get(sentiment, emoji_map['unknown'])

    return render_template('index.html', sentiment=sentiment, emoji_gif=emoji_gif, statement=statement)

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

# --- Upload Route for Data ---
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    upload_message = None
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            upload_message = "No file uploaded. Please select a file."
        else:
            try:
                # Read CSV file into pandas DataFrame
                df = pd.read_csv(file)
                # Save new data in the designated UPLOAD_PATH
                if os.path.exists(UPLOAD_PATH):
                    df.to_csv(UPLOAD_PATH, mode='a', header=False, index=False)
                else:
                    df.to_csv(UPLOAD_PATH, index=False)
                # Fine-tune the model with new data
                upload_message = fine_tune()
            except Exception as e:
                upload_message = f"Error during file processing: {e}"

    return render_template('upload.html', upload_message=upload_message)

if __name__ == '__main__':
    app.run(port=6060, debug=True)
