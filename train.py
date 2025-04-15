import os
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, regexp_replace, col
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import get_model, build_model

# File paths
BASE_DATA_PATH = "data/sentiment_dataset.csv"
NEW_DATA_PATH = "data/new_data.csv"
MODEL_PATH = "models/sentiment_model.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"
CHECKPOINT_DIR = "models/checkpoints"

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_tokenizer():
    if os.path.exists(TOKENIZER_PATH):
        with open(TOKENIZER_PATH, 'rb') as f:
            return pickle.load(f)
    return None

def save_tokenizer(tokenizer):
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)

def get_spark_session():
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
    return spark

def spark_load_csv(path):
    spark = get_spark_session()
    df = spark.read.option("header", "true").csv(path)
    # Clean text: remove URLs, mentions, hashtags, and lowercase text.
    df = df.withColumn("cleaned_text", lower(regexp_replace(col("text"), r"http\S+|@\w+|#\w+", "")))
    return df

def preprocess_data(df, tokenizer=None):
    """
    Converts a Spark DataFrame into a processed Pandas DataFrame:
      - Tokenizes the 'cleaned_text'
      - Pads sequences to a fixed length
      - Encodes sentiment labels using LabelEncoder
      - Extracts the optional 'sarcasm' column (or uses zeros if not present)
    """
    # Convert Spark DataFrame to pandas
    if "sarcasm" in df.columns:
        pdf = df.select("cleaned_text", "sentiment", "sarcasm").toPandas()
    else:
        pdf = df.select("cleaned_text", "sentiment").toPandas()
    
    texts = pdf["cleaned_text"].fillna("").tolist()
    
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        save_tokenizer(tokenizer)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=100, padding='post')
    
    # Encode sentiment labels
    labels = pdf["sentiment"].tolist()
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    
    # Extract sarcasm indicator if available, else zeros
    if "sarcasm" in pdf.columns:
        sarcasm = pdf["sarcasm"].astype(float).values.reshape(-1, 1)
    else:
        sarcasm = np.zeros((len(pdf), 1))
    
    return padded, encoded_labels, tokenizer, sarcasm

def fine_tune():
    """
    Fine-tunes the model on the new daily data.
    Then appends this data to the base dataset and deletes the new data file.
    """
    if not os.path.exists(NEW_DATA_PATH):
        return "No new data to train on."

    new_df = spark_load_csv(NEW_DATA_PATH)
    tokenizer = load_tokenizer()
    X_new, y_new, tokenizer, sarcasm_new = preprocess_data(new_df, tokenizer)
    
    model = get_model(use_sarcasm=True)
    model.fit([X_new, sarcasm_new], y_new, epochs=2, batch_size=32)
    model.save(MODEL_PATH)
    
    # Append new data into the base dataset
    df_new = pd.read_csv(NEW_DATA_PATH)
    if os.path.exists(BASE_DATA_PATH):
        df_base = pd.read_csv(BASE_DATA_PATH)
        df_total = pd.concat([df_base, df_new], ignore_index=True)
    else:
        df_total = df_new
    df_total.to_csv(BASE_DATA_PATH, index=False)
    
    os.remove(NEW_DATA_PATH)
    return "Fine-tuning complete with new data."

def full_retrain():
    """
    Fully retrains the model using the complete base dataset.
    Saves a checkpoint with a timestamp and overwrites the main model file.
    """
    if not os.path.exists(BASE_DATA_PATH):
        return "No base data available for retraining."

    base_df = spark_load_csv(BASE_DATA_PATH)
    tokenizer = load_tokenizer()
    X_all, y_all, tokenizer, sarcasm_all = preprocess_data(base_df, tokenizer)
    if tokenizer is None:
        save_tokenizer(tokenizer)
    
    # Create a new model instance and train it
    model = build_model(use_sarcasm=True)
    model.fit([X_all, sarcasm_all], y_all, epochs=5, batch_size=32)
    
    # Save a checkpoint with timestamp (e.g., model_2025_04_15.h5)
    timestamp = datetime.now().strftime("%Y_%m_%d")
    checkpoint_path = f"{CHECKPOINT_DIR}/model_{timestamp}.h5"
    model.save(checkpoint_path)
    
    model.save(MODEL_PATH)
    return "Full retrain complete and checkpoint saved."
