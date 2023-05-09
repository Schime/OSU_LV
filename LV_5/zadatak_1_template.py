import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score





X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)



# Prikaz podataka za train i test
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='Paired', label='train')
plt.scatter(X_test[:,0], X_test[:,1], marker='x', c=y_test, cmap='Paired', label='test')
plt.legend()
plt.show()

# Komentar:
# koristimo funkciju scatter za prikazivanje podataka. Opcija c se koristi za definiranje boje svake klase,
# a opcija cmap za definiranje mape boja. U ovom slučaju koristimo mapu boja "Paired". Opcija label se koristi za dodavanje legende,
# a opcija marker za definiranje markera za testni skup podataka.


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# b) Izgradite model logističke regresije pomoću scikit-learn biblioteke na temelju skupa podataka za učenje

# Izgradnja modela
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

# Evaluacija modela na skupu za testiranje
accuracy = model.score(X_test, y_test)
print("Točnost modela: {:.2f}%".format(accuracy * 100))

# Komentar:
# Nakon inicijalizacije modela, koristimo metodu fit da bismo trenirali model na skupu podataka za učenje.
# Zatim koristimo metodu score da bismo izračunali točnost modela na skupu za testiranje.
# Točnost se izračunava kao omjer broja točno klasificiranih uzoraka i ukupnog broja uzoraka u testnom skupu.


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# c) Pronađite u atributima izgrađenog modela parametre modela. Prikažite granicu odluke naučenog modela u ravnini x1 −x2 zajedno s podacima za učenje.
#    Napomena: granica odluke u ravnini x1−x2 definirana je kao krivulja: θ0+θ1x1+θ2x2 = 0.

print("Koeficijenti modela: ", model.coef_)
print("Slobodni član modela: ", model.intercept_)

# Prikaz podataka za učenje
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')

# Crtanje granice odluke
theta = np.concatenate([model.intercept_, model.coef_.ravel()])
x = np.linspace(-4, 4, 100)
y = -(theta[0] + theta[1]*x) / theta[2]
plt.plot(x, y, color='black', label='Granica odluke')

plt.legend()
plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# d) Provedite klasifikaciju skupa podataka za testiranje pomoću izgrađenog modela logističke regresije.
#    Izračunajte i prikažite matricu zabune na testnim podacima. Izračunate točnost, preciznost i odziv na skupu podataka za testiranje.

# Izvršavanje klasifikacije na skupu za testiranje
y_pred = model.predict(X_test)

# Izračunavanje matrice zabune
cm = confusion_matrix(y_test, y_pred)
print("Matrica zabune:\n", cm)

# Izračunavanje i prikaz točnosti, preciznosti i odziva
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Točnost: ", accuracy)
print("Preciznost: ", precision)
print("Odziv: ", recall)

# Komentar:
# Kod koristi funkciju predict na modelu kako bi izvršio klasifikaciju na skupu za testiranje.
# Zatim se koristi funkcija confusion_matrix iz sklearn.metrics modula da bi se izračunala matrica zabune.
# Nakon toga, kod izračunava i prikazuje točnost, preciznost i odziv pomoću funkcija accuracy_score, precision_score i recall_score iz sklearn.metrics modula.
# Točnost predstavlja omjer točnih predikcija modela i ukupnog broja predikcija. Preciznost predstavlja omjer točnih pozitivnih predikcija i svih pozitivnih predikcija,
# dok odziv predstavlja omjer točnih pozitivnih predikcija i ukupnog broja stvarnih pozitivnih primjera.


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# e) Prikažite skup za testiranje u ravnini x1−x2. Zelenom bojom označite dobro klasificirane primjere dok pogrešno klasificirane primjere označite crnom bojom.

# Prikaz skupa za testiranje u ravnini x1-x2
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm')

# Označavanje dobro klasificiranih primjera zelenom bojom i pogrešno klasificiranih crnom bojom
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        plt.scatter(X_test[i, 0], X_test[i, 1], c='green', marker='o')
    else:
        plt.scatter(X_test[i, 0], X_test[i, 1], c='black', marker='x')

plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# Komentar:
# Kod koristi funkciju scatter za prikaz točaka u ravnini x1-x2.
# Boja točaka odgovara stvarnoj klasi. Zatim se koristi petlja da bi se označili dobro klasificirani primjeri
# zelenom bojom i pogrešno klasificirani primjeri crnom bojom. Prikazuje se graf s označenim primjerima i oznakama osi.