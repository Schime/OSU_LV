import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error




data = pd.read_csv("data_C02_emission.csv")


# a) Odaberite željene numeričke veličine specificiranjem liste s nazivima stupaca. Podijelite podatke na skup za učenje i skup za testiranje u omjeru 80%-20%.

filtered_data = data[["Engine Size (L)", "Cylinders", "Fuel Consumption City (L/100km)", "Fuel Consumption Hwy (L/100km)", "Fuel Consumption Comb (L/100km)", "Fuel Consumption Comb (mpg)", "CO2 Emissions (g/km)"]]

X = filtered_data.iloc[:, : -1]        # svi osim CO2 Emissions (g/kg)
y = filtered_data.iloc[:, -1]           # samo CO2 Emissions (g/kg)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# b) Pomoću matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova o jednoj numeričkoj veličini.
#    Pri tome podatke koji pripadaju skupu za učenje označite plavom bojom, a podatke koji pripadaju skupu za testiranje označite crvenom bojom.

plt.scatter(X_train["Fuel Consumption City (L/100km)"], y_train, color="blue", label="Training data")
plt.scatter(X_test["Fuel Consumption City (L/100km)"], y_test, color="red", label="Testing data")
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.show()

# Komentar:
# Podijelili smo podatke na X i y. X su svi osim CO2 emisije, a y je samo CO2 emisija.
# Prvi scatter će uzeti podskup od X (trening skup), a u drugom scatteru će uzeti drugi podskup od X (testing skup) koje smo podijelili u omjeru 80% : 20%
# Prvi scatter uspoređuje trening skup od X i y (koji su prva 2 argumenta scatter funkcije) te će plotati samo PLAVE kružiće
# Drugi scatter uspoređuje testing skup od X i y (koji su prva 2 argumenta scatter funkcije) te će plotati samo CRVENE kružiće


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# c) Izvršite standardizaciju ulaznih veličina skupa za učenje. Prikažite histogram vrijednosti jedne ulazne veličine prije i nakon skaliranja.
#    Na temelju dobivenih parametara skaliranja transformirajte ulazne veličine skupa podataka za testiranje.

sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)       # računa srednju vrijednost i standardnu devijaciju svake varijable u skupu za u čenje i skalira ih na novi raspon [0, 1]
X_test_n = sc.transform(X_test)             # primjenjuje istu transformaciju na skupu za testiranje, koristeći parametre koji su naučeni iz skupa za učenje

# Biramo jednu ulaznu veličinu: Fuel Consumption City (L/100km)
# Histogram prije skaliranja
plt.hist(X_train["Fuel Consumption City (L/100km)"], bins=20)
plt.title("Prije skaliranja")
plt.xlabel("Vrijednosti")
plt.ylabel("Broj uzoraka")
plt.show()

# Histogram nakon skaliranja
plt.hist(X_train_n[:, 0], bins=20)
plt.title("Nakon skaliranja")
plt.xlabel("Vrijednosti")
plt.ylabel("Broj uzoraka")
plt.show()

# Komentar:
# Koristeći MinMaxScaler skalirali smo podatke na raspon [0, 1]. Znači da su svi podaci skalirani na istu skalu.
# Skaliranjem možemo izbjeći overfitting i underfitting.
# bins=20 označava "debljinu" svake vrijednosti grafa. Što je bin veći to će biti tanji prikaz (raspon vrijednosti podataka će biti podijeljen u 20 jednakih intervala)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# d) Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i povežite ih s izrazom 4.6.

linear_regression_model = lm.LinearRegression()         # inicijalizacija modela
linear_regression_model.fit(X_train_n, y_train)         # treniranje modela na skupu za treniranje

# Ispis
print(linear_regression_model.intercept_)               # presjecište s osi y
print(linear_regression_model.coef_)                    # nagibi regresijskih linija


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# e) Izvršite procjenu izlazne veličine na temelju ulaznih veličina skupa za testiranje.
#    Prikažite pomoći dijagrama raspršenja odnos između stvarnih vrijednosti izlazne veličine i procjene dobivene modelom.

y_test_prediction = linear_regression_model.predict(X_test)         # procjena izlazne veličine na temelju ulaznih veličina skupa za testiranje

# Prikaz odnosa između stvarnih vrijednosti izlazne veličine i procjene dobivene modelom
plt.scatter(y_test, y_test_prediction)
plt.xlabel("Stvarne vrijednosti CO2 emisija (g/km)")
plt.ylabel("Procjene CO2 emisija (g/km)")
plt.title("Dijagram raspršenja za procjene CO2 emisija")
plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# f) Izvršite vrednovanje modela na način da izračunate vrijednosti regresijskih metrika na skupu podataka za testiranje.

# MSE = Mean squared error
MSE = mean_squared_error(y_test,y_test_prediction)

# MAE = Mean absolute error
MAE = mean_absolute_error(y_test,y_test_prediction)

# MAPE = Mean absolute percentage error
MAPE = mean_absolute_percentage_error(y_test,y_test_prediction)

print("Mean squared error: ", MSE)
print("Mean absolute error: ", MAE)
print("Mean absolute percentage error: ", MAPE)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# g) Što se događa s vrijednostima evaluacijskih metrika na testnom skupu kada mijenjate broj ulaznih veličina?

# Odgovor:
# Evaluacijske metrike na testnom skupu se mogu promijeniti ukoliko promijenimo broj ulaznih vrijednosti
# Izvedba modela se može poboljšati dodavanjem više ulaznih varijabli
# Također, mogu postojati varijable koje nisu pogodne za izvedbu modela