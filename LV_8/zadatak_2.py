import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.models import load_model




num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.models.load_model(r"FCN")

x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)

predictions = model.predict(x_test_s)

predictions_model = np.argmax(predictions, axis=1)

y_test_true = np.argmax(y_test_s, axis=1)

for i in range(len(predictions_model)):
    if predictions_model[i] != y_test_true[i]:
        plt.imshow(x_test[i])
        plt.title(
            f'Real value: {y_test[i]}, predicted value: {predictions_model[i]} ')
        plt.show()