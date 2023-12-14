from typing import List
import pandas
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os

image_size = 120

train_path = "train/"
test_path = "test/"

testLabels = os.listdir(test_path)
trainLabels = os.listdir(train_path)

def dataLoader(aboPath:str, paths:List[str]):
    inputs = []
    targets = []
    for index, i in enumerate(paths):
        sub = os.path.join(aboPath, i)
        imgdir = os.listdir(sub)
        # for i in imgdir:
        for i in imgdir:
            if not "Copy" in i:
                    
                image = keras.utils.load_img(
                    os.path.join(sub, i),
                )
                image = image.resize((image_size, image_size))
                inputs.append(keras.utils.img_to_array(image))
                targets.append(index)
    return (np.asarray(inputs), np.asarray(targets)) 

val_data = dataLoader(test_path, testLabels)
train_data = dataLoader(train_path, trainLabels)

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        # layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)

Input = keras.Input((image_size, image_size, 3))
x = data_augmentation(Input)
x = layers.Conv2D(64, 3, activation = "relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.4)(x)

x = layers.Conv2D(128, 3, activation = "relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.4)(x)

x = layers.Conv2D(256, 3, activation = "relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.4)(x)

x = layers.Flatten()(x)
x = layers.Dense(len(np.unique(val_data[1])), tf.nn.softmax)(x)

model = keras.Model(inputs = Input, outputs = x)
model.summary()

tf.keras.utils.plot_model(model, show_shapes=True)

es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12) 
rLR = tf.keras.callbacks.ReduceLROnPlateau (
    monitor='val_accuracy',
    factor=0.5,
    patience=3,
    min_lr=1e-8,
    verbose=1,
)

model.compile("Adam", loss = tf.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

history = model.fit(train_data[0], train_data[1], batch_size= 10, epochs=200, validation_data=val_data, callbacks=[es, rLR])
valLoss = history.history["val_loss"]
valAccuracy = history.history["val_acc"]
Loss = history.history["loss"]
Accuracy = history.history["acc"]
plt.plot()
model.save("model.keras")