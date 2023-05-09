import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns




# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa


# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential()

model.add(layers.Dense(100, activation="relu", input_shape=(784, )))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.summary()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# TODO: definiraj karakteristike procesa ucenja pomocu .compile()

#Reshape podatke slika na 60000 sa dimenzijom 784
x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)


model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy",])


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# TODO: provedi ucenje mreze
history = model.fit(x_train_s, y_train_s, batch_size=64,
                    epochs=10, validation_split=0.1)

score = model.evaluate(x_test_s, y_test_s, verbose=0)
print(score)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# TODO: Prikazi test accuracy i matricu zabune

#predictions je matrica koja ima 10000 redaka i 10 stupaca di je svaki stupac vjerojatnost koji je broj na slici

predictions = model.predict(x_test_s)

#np.argmax uzima najveci broj tj najvecu vjerojatnost iz tih redaka i sad imamo samo 10000 redaka i ,
predictions_model = np.argmax(predictions, axis=1)
y_testt = np.argmax(y_test_s, axis=1)

c = confusion_matrix(y_testt, predictions_model)
print(c)

fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(c, annot=True, fmt='d', ax=ax, cmap='Blues')
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
ax.set_title('Confusion Matrix')

plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# TODO: spremi model

model.save("FCN/")