import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import layers

from keras.models import load_model



data = np.genfromtxt("file.csv", delimiter=",", skip_header=9)

X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Izgradite neuronsku mrežu sa sljedećim karakteristikama:
# - model očekuje ulazne podatke s 8 varijabli
# - prvi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju
# - drugi skriveni sloj ima 8 neurona i koristi relu aktivacijsku funkciju
# - izlasni sloj ima jedan neuron i koristi sigmoid aktivacijsku funkciju.
# Ispišite informacije o mreži u terminal.

model = Sequential()

model.add(layers.Input(shape=(8, )))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Podesite proces treniranja mreže sa sljedećim parametrima:
# - loss argument: cross entropy
# - optimizer: adam
# - metrika: accuracy.

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy",])


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Pokrenite učenje mreže sa proizvoljnim brojem epoha (pokušajte sa 150) i veličinom batch-a 10.

batch_size = 10
epochs = 150
history = model.fit (X_train, y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.1)
predictions = model.predict ( X_test )
score = model.evaluate(X_test, y_test, verbose = 0)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju učitanog modela.

model.save("FCN/")


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Izvršite evaluaciju mreže na testnom skupu podataka.
model = load_model('FCN/')

history = model.fit(x_train_s, y_train_s, batch_size=10, epochs=150, validation_split=0.1)

score = model.evaluate(x_test_s, y_test_s, verbose=0)
print(score)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Izvršite predikciju mreže na skupu podataka za testiranje. Prikažite matricu zabune za skup podataka za testiranje. Komentirajte dobivene rezultate.
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
