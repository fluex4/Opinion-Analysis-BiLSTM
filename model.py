import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam

MODEL_PATH = "models/sentiment_model.h5"

def build_model(vocab_size=10000, embedding_dim=64, input_length=100, num_classes=3, use_sarcasm=True):
    """
    Builds a two-input model. The main text branch uses Embedding + Bi-LSTM,
    and if `use_sarcasm` is True, a second input is concatenated.
    """
    text_input = Input(shape=(input_length,), name="text_input")
    x = Embedding(vocab_size, embedding_dim, input_length=input_length)(text_input)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.5)(x)
    
    if use_sarcasm:
        sarcasm_input = Input(shape=(1,), name="sarcasm_input")
        concat = Concatenate()([x, sarcasm_input])
        dense = Dense(64, activation='relu')(concat)
        output = Dense(num_classes, activation='softmax')(dense)
        model = Model(inputs=[text_input, sarcasm_input], outputs=output)
    else:
        dense = Dense(64, activation='relu')(x)
        output = Dense(num_classes, activation='softmax')(dense)
        model = Model(inputs=text_input, outputs=output)
        
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

def get_model(use_sarcasm=True):
    """
    Returns the trained model if exists; otherwise builds a new one.
    """
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        return build_model(use_sarcasm=use_sarcasm)
