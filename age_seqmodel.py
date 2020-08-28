import os, shutil

original_dataset_dir = r'dataset'
train_dir = os.path.join(base_dir, 'train')
if not os.path.isdir(train_dir): os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
if not os.path.isdir(validation_dir): os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
if not os.path.isdir(test_dir): os.mkdir(test_dir)

from keras import layers
from keras import models
from tensorflow.keras import activations
from tensorflow.keras import initializers 
from keras.layers import LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten

model = models.Sequential()
model.add(layers.Conv2D(64, (3,3), activation ='relu',input_shape =(150,150,3)))
model.add(layers.Conv2D(16, (3,3), kernel_initializer=initializers.he_normal(), activation ='relu'))
#model.add(BatchNormalization())
model.add(layers.MaxPool2D((2,2)))
#model.add(Dropout(0.5))

#2nd convolution layer
model.add(layers.Conv2D(16, (3,3),kernel_initializer=initializers.he_normal(), activation ='relu'))
model.add(layers.Conv2D(16, (3,3),kernel_initializer=initializers.he_normal(), activation ='relu'))
#model.add(BatchNormalization())
model.add(layers.MaxPool2D((2,2)))
#model.add(Dropout(0.5))

#3rd convolution layer
model.add(Conv2D(128, (3, 3),kernel_initializer=initializers.he_normal(), activation='relu'))
model.add(Conv2D(128, (3, 3),kernel_initializer=initializers.he_normal(), activation='relu'))
#model.add(BatchNormalization())
model.add(layers.MaxPool2D((2,2)))

#4rd convolution layer
model.add(Conv2D(256, (3, 3),kernel_initializer=initializers.he_normal(), activation='relu'))
model.add(Conv2D(256, (3, 3),kernel_initializer=initializers.he_normal(), activation='relu'))
#model.add(BatchNormalization())
model.add(layers.MaxPool2D((2,2)))

#5rd convolution layer
model.add(Conv2D(512, (3, 3),kernel_initializer=initializers.he_normal(), activation='relu'))
model.add(Conv2D(512, (3, 3),kernel_initializer=initializers.he_normal(), activation='relu'))
#model.add(BatchNormalization())
#model.add(layers.MaxPool2D((2,2)))

model.add(Flatten())
#fully connected neural networks
model.add(Dense(1024,kernel_initializer=initializers.he_normal(), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024,kernel_initializer=initializers.he_normal(), activation='relu'))
model.add(Dropout(0.2))
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.2))
model.add(layers.Dense(1))

from keras import optimizers
model.compile(optimizer = optimizers.Adam(), loss = 'mse', metrics = ['mae'])

from keras.preprocessing.image import ImageDataGenerator

#train_datagen = ImageDataGenerator(rescale =1./255)
train_datagen = ImageDataGenerator(
#rotation_range=40,
#width_shift_range=0.2,
#height_shift_range=0.2,
#shear_range=0.2,
#zoom_range=0.2,
#horizontal_flip=True,
rescale = 1./255
)

test_datagen = ImageDataGenerator(rescale =1./255)

train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(150,150),
batch_size = 100,
class_mode='sparse'
)

validation_generator = test_datagen.flow_from_directory(
validation_dir,
target_size=(150,150),
batch_size = 100,
class_mode='sparse'
)

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
    
print(train_datagen)

history = model.fit(
train_generator,
steps_per_epoch=224,
epochs=100,
validation_data=validation_generator,
validation_steps=90
)

model.save('faceagereg1.h5')
faceagereg1_json = model.to_json()
with open("faceagereg1.json", "w") as json_file:
    json_file.write(faceagereg1_json)
model.save_weights("faceagereg1.h5")

import matplotlib.pyplot as plt

history_dict = history.history
mae = history.history['mae']
val_mae = history_dict['val_mae']
loss_values = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss_values)+1)
plt.plot(epochs, mae, 'bo', label = 'Training mae')
plt.plot(epochs, val_mae, 'b', label = 'validation mae')
plt.title('Training and validation MAE')
plt.legend()
plt.figure()
plt.plot(epochs, loss_values, 'bo', label = 'Training mse')
plt.plot(epochs, val_loss, 'b', label = 'Validation mse')
plt.title('Training and validation MSE')
plt.legend()
plt.show()