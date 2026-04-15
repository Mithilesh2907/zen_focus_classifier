import gradio as gr
import tensorflow as tf
import numpy as np 
import librosa
import os


model_path = 'notebooks/zen_focus_model.h5'

if os.path.exists(model_path) :
    model = tf.keras.models.load_model(model_path)
    print("Model successfully loaded")
else :
    print("Can't find model")

def predictZen(audio):
    sr, y = audio
    
    y = y.astype(np.float32)
    
    if len(y.shape) >  1 :
        y = np.mean(y, axis=1)
    
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_spec = librosa.power_to_db(spec, ref=np.max)
    
    if log_spec.shape[1] > 216 :
        log_spec = log_spec[:, :216]
    else :
        pad_width = 216 - log_spec.shape[1]
        log_spec = np.pad(log_spec, ((0, 0), (0, pad_width)), mode='constant')
        
    log_spec = (log_spec - np.min(log_spec)) / (np.max(log_spec) - np.min(log_spec))
    log_spec = log_spec.reshape(1, 128, 216, 1)
    
    prediction = model.predict(log_spec)
    labels = ['Productive (focus)', 'Distracted']
    
    return {labels[i]: float(prediction[0][i]) for i in range(2)}

demo = gr.Interface(
    fn=predictZen,
    inputs=gr.Audio(sources=["microphone"]),
    outputs=gr.Label(num_top_classes=2) ,
    title="Zen_Focus_Classifier" ,
    description="Record 5 seconds of your environment to check your focus status!"
)

demo.launch()