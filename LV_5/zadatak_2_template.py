import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report



labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

y = y[:, 0]

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# a) Pomoću stupčastog dijagrama prikažite koliko primjera postoji za svaku klasu (vrstu pingvina) u skupu podataka za učenje i skupu podataka za testiranje.
#    Koristite numpy funkciju unique.

# Broj primjera za svaku klasu u skupu podataka za učenje
train_classes, train_counts = np.unique(y_train, return_counts=True)
print("Train set:")
for i in range(len(train_classes)):
    print("Class {}: {}".format(labels[train_classes[i]], train_counts[i]))

# Broj primjera za svaku klasu u skupu podataka za testiranje
test_classes, test_counts = np.unique(y_test, return_counts=True)
print("\nTest set:")
for i in range(len(test_classes)):
    print("Class {}: {}".format(labels[test_classes[i]], test_counts[i]))

# Komentar:
# Za svaku klasu se ispisuje koliko primjera postoje u skupu podataka za učenje i skupu podataka za testiranje.
# Koristili smo rječnik "labels" koji smo definirali u početnom dijelu skripte kako bismo dobili nazive klasa umjesto brojeva.


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# b) Izgradite model logističke regresije pomoću scikit-learn biblioteke na temelju skupa podataka za učenje.

# Kreiranje instance modela
log_reg = LogisticRegression(random_state=123)

# Treniranje modela na skupu podataka za učenje
log_reg.fit(X_train, y_train.ravel())

# Komentar:
# U ovom primjeru koristimo funkciju LogisticRegression() za kreiranje instance modela,
# a zatim pozivamo metodu fit() za treniranje modela na skupu podataka za učenje (X_train i y_train).
# Parametar random_state služi za inicijalizaciju generatora slučajnih brojeva za reproducibilnost rezultata.


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# c) Pronađite u atributima izgrađenog modela parametre modela.
print(log_reg.coef_)
print(log_reg.coef_.T)

# Komentar:
# Ovo će ispisati matricu koeficijenata dimenzija (broj klasa) x (broj ulaznih varijabli).


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# d) Pozovite funkciju plot_decision_region pri čemu joj predajte podatke za učenje i izgrađeni model logističke regresije.
#    Kako komentirate dobivene rezultate?

plot_decision_regions(X_train, y_train, log_reg)
plt.show()


# Komentar:
# Iz dobivenog grafa se može vidjeti da logistička regresija dobro razdvaja primjere u tri klase na temelju duljine kljuna i duljine peraja.
# Ipak, postoji nekoliko primjera koji se nalaze blizu granica odluke i koji bi mogli biti pogrešno klasificirani.
# Također, granice odluke nisu uvijek savršeno glatke, što može ukazivati na ograničenja linearne regresije u razdvajanju podataka koji nisu linearno odvojivi.


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# e) Provedite klasifikaciju skupa podataka za testiranje pomoću izgrađenog modela logističke regresije.
#    Izračunajte i prikažite matricu zabune na testnim podacima. Izračunajte točnost.
#    Pomoću classification_report funkcije izračunajte vrijednost četiri glavne metrike na skupu podataka za testiranje.

y_test_predict = log_reg.predict(X_test)

cm = confusion_matrix(y_test, y_test_predict)
print("Matrica zabune:", cm)
confusion_matrix_display = ConfusionMatrixDisplay(cm)
confusion_matrix_display.plot()
plt.show()

print('\n\nTočnost:\n', classification_report(y_test, y_test_predict))


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# f) Dodajte u model još ulaznih veličina. Što se događa s rezultatima klasifikacije na skupu podataka za testiranje?

input_variables = ['bill_length_mm',
                    'flipper_length_mm',
                    'bill_depth_mm',
                    'body_mass_g']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

y_test_p = LogRegression_model.predict(X_test)

print("\n\nTočnost:\n",classification_report(y_test, y_test_p))