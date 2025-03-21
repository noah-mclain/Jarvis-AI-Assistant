import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from utils import extract_mfcc
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write


SAMPLING_RATE = 16000  # Match the sample rate your model expects (16kHz)
DURATION = 5  # Seconds to record
FILENAME = "test_audio.wav"

# DATA_DIR = '../Datasets/Google Speech Commands/'

CLASSES = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']
print(CLASSES)
# load model
savedModel=load_model('simple_speech.h5')


print('Recording...')
audio = sd.rec(
    int(DURATION * SAMPLING_RATE),
    samplerate=SAMPLING_RATE,
    channels=2,  
    dtype=np.int16
)
sd.wait()  # Wait until recording finishes
print("Recording complete!")

print("Recording saved to:", FILENAME)

# Save to .wav file
write(FILENAME, SAMPLING_RATE, audio)

file_path = r'test_audio.wav'
x = extract_mfcc(file_path,sr = 16000)

x = x[np.newaxis, ..., np.newaxis]

predictions = savedModel(x)
predict_index = np.argmax(predictions, axis=1).item()
print(CLASSES[predict_index])