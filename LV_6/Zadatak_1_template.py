import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score




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
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine  - osigurava da svi podaci budu na istoj skali
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)            # treniranje modela s podacima za trening

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))           # izračunavanje točnosti train skupa
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))              # izračunavanje točnosti test skupa

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)          # prikazivanje granica odluke modela
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Izradite algoritam KNN na skupu podataka za učenje (uz K=5).
# Izračunajte točnost klasifikacije na skupu podataka za učenje i skupu podataka za testiranje.
# Usporedite dobivene rezultate s rezultatima logističke regresije. Što primjećujete vezano uz dobivenu granicu odluke KNN modela?

# KNN model
KNN_model = KNeighborsClassifier(n_neighbors = 5)
KNN_model.fit(X_train, y_train)

# Evaluacija KNN modela
y_train_p = KNN_model.predict(X_train_n)
y_test_p = KNN_model.predict(X_test_n)

print("KNN: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# Granica odluke pomoću KNN
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Točnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

# Komentar:
# Logistička regresija ima razlike između skupa za učenje i testiranje, što može ukazivati na neku mjeru overfittinga
# Kod KNN-a vidimo da je točnost skupa za testiranje manja od točnosti skupa za treniranje, što ukazuje na overfitting
# Primjećujemo da granica odluke KNN modela nije baš glatka. To može biti posljedica odabira parametra K, koji određuje koliko susjeda se uzima u obzir pri klasifikaciji
# Da bismo približili točnost ta 2 skupa, trebali bismo koristiti tehnike regularizacije; Manhattan ili Euclidean


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Kako izgleda granica odluke kada je K =1 i kada je K = 100?
# Kada je K = 1, granica odluke će biti vrlo "gruba" i prilagođena svakom pojedinačnom primjeru u skupu podataka za učenje,
# što može dovesti do prenaučenosti modela i generalizacijske pogreške.

# S druge strane, kada je K = 100, granica odluke će biti "gladka" i moći će se prilagoditi većim i manjim skupovima podataka za učenje.
# Međutim, model s tako visokom vrijednosti K može imati poteškoća u pronalaženju pravih klasa ako postoje područja u kojima su podaci gusto raspoređeni.

# Kako bi se pronašla optimalna vrijednost K, može se koristiti metoda unakrsne provjere (cross-validation) na skupu podataka za učenje,
# što bi omogućilo pronalaženje vrijednosti K koja daje najbolje performanse na skupu podataka za testiranje.


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Pomoću unakrsne validacije odredite optimalnu vrijednost hiperparametra K algoritma KNN za podatke iz Zadatka 1.

knn_model = KNeighborsClassifier()

# GridSearch za pronalazak optimalnog parametra K
param_grid = {'n_neighbors': range(1, 100)}

grid = GridSearchCV(knn_model, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_n, y_train)

print("Najbolji K parametar: ", grid.best_params_['n_neighbors'])

# Komentar:
# Koristi se GridSearchCV funkcija za pretraživanje po mreži i pronalaženje najbolje vrijednosti parametra n_neighbors.
# U ovom primjeru se koristi 5-struka unakrsna validacija (cv=5) i accuracy kao mjera performansi (scoring='accuracy').


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Na podatke iz Zadatka 1 primijenite SVM model koji koristi RBF kernel funkcijute prikažite dobivenu granicu odluke.
# Mijenjajte vrijednost hiperparametra C i γ. Kako promjenaovih hiperparametara utječe na granicu odluke te pogrešku na skupu podataka za testiranje?
# Mijenjajte tip kernela koji se koristi. Što primjećujete?

SVM_model = svm.SVC(kernel='rbf', gamma=1, C=0.1)
SVM_model.fit(X_train_n, y_train)

# Evaluacija
y_train_p_SVM = SVM_model.predict(X_train_n)
y_test_p_SVM = SVM_model.predict(X_test_n)

train_accuracy_svm = SVM_model.score(X_train_n, y_train)
test_accuracy_svm = SVM_model.score(X_test_n, y_test)

print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_SVM))))

# Granica odluke
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
plt.tight_layout()
plt.show()

# Komentar:
# Vidimo da je točnost na oba skupa puno veća nego do sada



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Pomoću unakrsne validacije odredite optimalnu vrijednost hiperparametra C i γ algoritma SVM za problem iz Zadatka 1.

SVM_model_default = svm.SVC()
param_grid = {'C': [10, 100, 100],
              'gamma': [10, 1, 0.1, 0.01]}
svm_gscv = GridSearchCV(SVM_model_default, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm_gscv.fit(X_train_n, y_train)

print(svm_gscv.best_params_)
print(svm_gscv.best_score_)
print(svm_gscv.cv_results_)

# Komentar:
# n_jobs predstavlja broj procesa koje GridSearchCV koristi za izvršavanje zadataka paralelno.
# Ako je -1, tada će se koristiti svi raspoloživi procesori. Ako je 1, tada se zadaci neće izvršavati paralelno, već u jednom procesu
# Kada bi bilo 1, to može usporiti izvršavanje zadatka ako sustav ima više procesora, jer će samo jedan procesor biti korišten za izvršavanje.
# Dok s druge strane, ako sustav ima samo jednozadaćno okruženje (mobitel) onda može biti 1 kako bi se izbjeglo preopterećivanje procesora