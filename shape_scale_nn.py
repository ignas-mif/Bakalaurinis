import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random as ran

tf.enable_eager_execution()

labels = list()
features = np.random.uniform(1, 100, (100, 3))
for x in range(0, 100):
    l = features[x][0] + 5*features[x][1] + 3*features[x][2]
    k = 2
    labels.append([l, k])

model = tf.keras.Sequential([
layers.Dense(10, activation='relu', input_dim=3),
layers.Dense(64, activation='relu'),
layers.Dense(64, activation='relu'),
layers.Dense(2, activation='softmax')]
)

model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(np.array(features), np.array(labels), epochs=1000, steps_per_epoch=2)
print(history)

prediction = model.predict(np.array([[1,1,1]]))
print(prediction)