import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import pandas as pd
import joblib

st.title('Emotion Recognition From Audio :sunglasses:')
st.divider()
file = st.file_uploader("choose a audio file of wav format", type='wav', accept_multiple_files=False)

model = tf.keras.models.load_model('Model/model.h5')
encoder =joblib.load('Encoder/encoder.joblib')

def extract_features(data,sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result
if file is not None:
    data , sr = librosa.load(file)
    res = extract_features(data,sr)
    res = res.reshape(1,162,1)
    pred = model.predict(res)
    max_index = np.argmax(pred)
    original_class_labels = encoder.categories_
    result = original_class_labels[0][max_index]
    st.title("Emotion : "+ result)
