import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split

emotions = {'01':'neutral',
            '02':'calm',
            '03':'happy',
            '04':'sad',
            '05':'angry',
            '06':'fearful',
            '07':'disgust',
            '08':'surprised'}

observed_emotions=['calm', 
                   'happy', 
                   'fearful', 
                   'disgust']

def load_data(test_size = 0.2, valid_size = 0.1):
    #MLP Classifier
    x, y = [], []
    for file in glob.glob("SoundData\\Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = feature_extraction(file)
        x.append(feature)      
        y.append(emotion)
    
    """#Deep Neural Network
    x = []
    y = np.zeros((768, 4))
    i=0
    for file in glob.glob("SoundData\\Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = feature_extraction(file)
        x.append(feature)      
        if emotion == 'calm':
            y[i,:]=[1, 0, 0, 0]
        elif emotion == 'happy':
            y[i,:]=[0, 1, 0, 0]
        elif emotion == 'happyfearful':
            y[i,:]=[0, 0, 1, 0]
        else:
            y[i,:]=[0, 0, 0, 1]
        i = i + 1"""

    x_train, x_test, y_train, y_test = train_test_split(np.array(x), y, test_size = test_size, random_state = 9)
    valid = (int) (len(x) * valid_size)
    x_valid = x_train[0:valid, :]
    y_valid = y_train[0:valid]
    return x_train, x_test, x_valid, y_train, y_test, y_valid 

def feature_extraction(file_name):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype = "float32")
        sample_rate = sound_file.samplerate
        result = np.array([])
            
        lpc = np.mean(librosa.lpc(X, 16).T, axis=0)
        result = np.hstack((result, lpc))
    
        mfccs = np.mean(librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc = 50).T, axis = 0)
        result = np.hstack((result, mfccs))
        

    return result
