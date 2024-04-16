#!/usr/bin/env python
# coding: utf-8

# # SonicNet: Deep Learning for Dynamic Sound Classification using CNN
# 
# This project consists of 3 main steps:
# 
# - Step 1. We will prepare our dataset for analysis and extract sound signal features from  audio files using Mel-Frequency Cepstral Coefficients(MFCC).
# - Step 2. Then we will build a Convolutional Neural Networks (CNN) model and train our model with our dataset. 
# - Step 3: Finally We Predict an Audio File's Class Using Our CNN Deep Learning Model
#    
# 
#         

# ### Step 1: Extracting Sound Features with Mel-Frequency Cepstral Coefficients (MFCC) for Dataset Analysis

# Each signal possesses unique traits. In sound processing, the mel-frequency cepstrum (MFC) captures the short-term power spectrum of a sound. Mel-frequency cepstral coefficients (MFCCs) represent this spectrum collectively.
# 
# Through the librosa library, we'll gather the distinct attributes of each audio signal in our dataset, storing them within a list

# In[ ]:


import tensorflow as tf
print(tf.__version__)


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import librosa
import numpy as np
import os, fnmatch
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint


# In[ ]:


# sample of how librosa handles sound signals, by reading an example audio signal using librosa
audio_file_path='UrbanSound8K/17973-2-0-32.wav'

librosa_audio_data, librosa_sample_rate = librosa.load(audio_file_path)


# In[ ]:


# librosa converts any stereo(2 channel) signal into mono(single channel) 
# so converted signal is one dimensional since the signals(2 channels) are converted into single channel(mono) 

print(librosa_audio_data)


# In[ ]:


librosa_audio_data.shape


# In[ ]:


# Plot the librosa audio data
# Audio with 1 channel 
plt.figure(figsize=(10, 4))
plt.plot(librosa_audio_data)
plt.show()


# In[ ]:


librosa_sample_rate


# #### Extracting features from all sound signals within the dataset.
# 
# Now we will calculate the Mel-Frequency Cepstral Coefficients(MFCC) of the audio samples. The MFCC calculate the frequency distribution across the window size, so it is possible to analyse both the frequency and time characteristics of the sound. Using this audio signal characteristics we can identify audio features for classification.
# 

# In[ ]:


mfccs = librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=45)   #n_mfcc: number of MFCCs to return 
print(mfccs.shape)


# In[ ]:


mfccs


# In[ ]:


# function for extracting MFC coefficients from signals using librosa

def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=45)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)    
    return mfccs_scaled_features


# In[ ]:


# finding all the files in directory
def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


dataset =[]                
                
for filename in find_files("UrbanSound8K/audio", "*.wav"):    
#    print("Found wav source:", filename)
    label = filename.split(".wav")[0][-5]
    if label == '-':
        label = filename.split(".wav")[0][-6]
    dataset.append({"file_name" : filename, "label" : label})
  
    
    
dataset


# In[ ]:


dataset = pd.DataFrame(dataset)

dataset.head()


# In[ ]:


dataset.shape


# In[ ]:


# iterating every sound file and extract features using MFC Coefficients of librosa 
# using features_extractor method defined above
extracted_features=[]

dataset['data'] = dataset['file_name'].apply(features_extractor)


# In[ ]:


dataset.head()


# In[ ]:


# changing column names
dataset = dataset.rename(columns={'label': 'class'})
dataset = dataset.rename(columns={'data': 'feature'})


# In[ ]:


dataset.head()


# In[ ]:


# dropping unnecessary column from dataframe
dataset.drop(['file_name'], axis=1, inplace=True)


# In[ ]:


dataset.head()


# In[ ]:


# converting extracted_features to Pandas dataframe
extracted_features_df = pd.DataFrame(dataset,columns=['class','feature'])
extracted_features_df.head()


# ### Defining Train and Validation Test Subsets

# In[ ]:


# splitting the dataset into independent and dependent dataset
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())


# In[ ]:


X.shape


# In[ ]:


X


# In[ ]:


y


# In[ ]:


y.shape


# In[ ]:


# performing Label Encoding since we need one hot encoded values for output classes in our model (1s and 0s)

# logic of one-hot encoding:
# 1 0 0 0 0 0 0 0 0 0 => air_conditioner
# 0 1 0 0 0 0 0 0 0 0 => car_horn
# 0 0 1 0 0 0 0 0 0 0 => children_playing
# 0 0 0 1 0 0 0 0 0 0 => dog_bark
# ...
# 0 0 0 0 0 0 0 0 0 1 => street_music

labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))


# In[ ]:


y


# In[ ]:


y[0]


# In[ ]:


# splitting dataset as Train and Test

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[ ]:



X_train


# In[ ]:


y


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# ### Step 2: Building and Training a CNN Model with Processed Sound Signals
# 

# In[ ]:


num_labels = 10


# In[ ]:


# building our CNN model

model=Sequential()
# 1. hidden layer
model.add(Dense(125,input_shape=(45,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# 2. hidden layer
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# 3. hidden layer
model.add(Dense(125))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# output layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# In[ ]:


# training the model

epochscount = 300
num_batch_size = 32

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=epochscount, validation_data=(X_test, y_test), verbose=1)


# In[ ]:


validation_test_set_accuracy = model.evaluate(X_test,y_test,verbose=0)
print(validation_test_set_accuracy[1])


# In[ ]:


X_test[1]


# In[ ]:


model.predict_classes(X_test)


# ### Step 3: Finally We Predict an Audio File's Class Using Our CNN Deep Learning Model
# 
# We preprocess the incoming audio data before making class predictions.
# 
# 

# In[ ]:


filename="UrbanSound8K/example_sound1_children_playing.wav"
sound_signal, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=45)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)


# In[ ]:


print(mfccs_scaled_features)


# In[ ]:


mfccs_scaled_features = mfccs_scaled_features.reshape(1,-1)


# In[ ]:


mfccs_scaled_features.shape


# In[ ]:


print(mfccs_scaled_features)


# In[ ]:


print(mfccs_scaled_features.shape)


# In[ ]:


result_array = model.predict(mfccs_scaled_features)


# In[ ]:


result_array


# In[ ]:


result_classes = ["air_conditioner","car_horn","children_playing","dog_bark","drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

result = np.argmax(result_array[0])
print(result_classes[result]) 

