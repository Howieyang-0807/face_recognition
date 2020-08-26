import sys, os
import pandas as pd
import numpy as np

import keras
from keras import layers
from tensorflow.keras import activations
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
df=pd.read_csv('fer2013.csv')

X_train,train_y,X_test,test_y=[],[],[],[]

for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
           X_train.append(np.array(val,'float32'))
           train_y.append(tf.one_hot(row['emotion'],7))
        elif 'PublicTest' in row['Usage']:
           X_test.append(np.array(val,'float32'))
           test_y.append(tf.one_hot(row['emotion'],7))
    except:
        print(f"error occured at index :{index} and row:{row}")

X_train = np.array(X_train,'float32')
train_y = np.array(train_y,'float32')
X_test = np.array(X_test,'float32')
test_y = np.array(test_y,'float32')
train_y=np_utils.to_categorical(train_y, num_classes=num_labels)
test_y=np_utils.to_categorical(test_y, num_classes=num_labels)


num_features = 64
num_labels = 7
batch_size = 128
epochs = 300
width, height = 48, 48

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

train_datagen = ImageDataGenerator(#rotation_range=10, 
                                             width_shift_range=0.2, 
                                             height_shift_range=0.2,
                                             shear_range=0.2, 
                                             zoom_range=0.2, 
                                             channel_shift_range=0.0, 
                                             horizontal_flip=True,
                                             fill_mode='nearest',
                                             rescale=1.0/255.0)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)
train_iterator = train_datagen.flow(X_train , train_y, batch_size=128)
test_iterator = test_datagen.flow(X_test, test_y, batch_size=128)

#1st convolution layer
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(X_train.shape[1:])))
model.add(layers.MaxPool2D((2,2)))
model.add(Dropout(0.3))

model.add(layers.Conv2D(64, (3,3), activation ='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(Dropout(0.3))

model.add(layers.Conv2D(64, (3,3), activation ='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(Dropout(0.3))

model.add(layers.Conv2D(128, (3,3), activation ='relu'))
#model.add(layers.MaxPool2D((2,2)))
model.add(Dropout(0.3))

model.add(layers.Conv2D(128, (1,1), activation ='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(2048, activation = 'relu'))
#model.add(layers.BatchNormalization())
#model.add(layers.Activation(activations.relu))
model.add(Dropout(0.3))
model.add(Dense(num_labels, activation='softmax'))

model.compile(keras.optimizers.Adam(0.001),
             loss=keras.losses.CategoricalCrossentropy(),
             metrics=[keras.metrics.CategoricalAccuracy()])
model_dir='labs-logs'
#os.makedirs(model_dir)
log_dir=os.path.join('labs-logs','model-emotinoal')
model_cbk=keras.callbacks.TensorBoard(log_dir = log_dir)
model_mckp = keras.callbacks.ModelCheckpoint(model_dir+'/Best-model-1.h5',
                                            monitor='val_categorical_accuracy',
                                            save_best_only=True,
                                            mode='max')
history=model.fit(train_iterator ,
         batch_size=batch_size,
         epochs=epochs,
         verbose=1,
         validation_data=(test_iterator),
         shuffle=True)

emotion_json = model.to_json()
with open("emotion.json", "w") as json_file:
    json_file.write(emotion_json)
model.save_weights("emotion.h5")

import matplotlib.pyplot as plt
history_dict = history.history
acc = history.history['categorical_accuracy']
val_acc = history_dict['val_categorical_accuracy']
loss_values = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss_values)+1)
plt.plot(epochs, acc, 'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'b', label = 'validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
