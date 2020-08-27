import numpy as np 
from cv2 import cv2 
import pandas as pd
import matplotlib.pyplot as plt


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import utils

tf.keras.backend.set_image_data_format('channels_first')

###import data###
df = pd.read_csv('mnist/train.csv').to_numpy()

X,y = df[:,1:],df[:,0]
X = X.reshape(X.shape[0],1,28,28)
X = (X / 255).astype('float32')
Input_shape = X.shape[1:]

print(X.shape,Input_shape)

model=Sequential([
   Conv2D(filters= 32, kernel_size=(5, 5), input_shape=Input_shape, activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
 
    Conv2D(filters=16,kernel_size=(3,3),activation='relu'),

    MaxPooling2D(2,2),
    Dropout(0.2),

    Flatten(),
    Dense(128,activation='relu'),
	
    Dense(64,activation='relu'),
  

    Dense(units=10,activation='softmax')
]) 

print(model.summary())
y_train=utils.to_categorical(y,10)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
epoch = 20
history=model.fit(X,y_train,batch_size=200,epochs=epoch,shuffle=True)

model.save('model.h5')