import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report



data = np.genfromtxt("D:\\OTHER\PROGRAMMING\\6. semestar\\Osnove strojnog učenja\\Izlazni_ispit\\pima-indians-diabetes.csv", delimiter=",", skip_header=9)

X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Izgradite model logističke regresije pomoću scikit-learn biblioteke na temelju skupa podataka za učenje.

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Provedite klasifikaciju skupa podataka za testiranje pomoću izgrađenog modela logističke regresije.

y_test_predict = log_reg.predict(X_test)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Izračunajte i prikažite matricu zabune na testnim podacima. Komentirajte dobivene rezultate.

cm = confusion_matrix(y_test, y_test_predict)
print("Matrica zabune:\n", cm)
confusion_matrix_display = ConfusionMatrixDisplay(cm)
confusion_matrix_display.plot()
plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Izračunajte točnost, preciznost i odziv na skupu podataka za testiranje. Komentirajte dobivene rezultate.

print('\n\nTočnost:\n', classification_report(y_test, y_test_predict))

