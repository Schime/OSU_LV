import numpy as np
import matplotlib.pyplot as plt



filename = "data.csv"
data = np.loadtxt(filename, delimiter = ',', skiprows = 1)

# Komentar:
# pošto radimo sa CSV (comma-seperated file), znamo da su podaci odijeljeni zarezom te ga delimiterom odjeljujemo
# skiprows označava broj redaka od početka čitanja dokumenta (1. redak u file-u ne čitamo)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# a) Na temelju veličine numpy polja data, na koliko osoba su izvršena mjerenja?

print("Broj ljudi: ", data.shape[0])
# gledamo koliko ljudi ima po spolu (muško + žensko), što je nulti stupac u tablici


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# b) Prikažite odnos visine i mase osobe pomoću naredbe matplotlib.pyplot.scatter.

height = data[:, [1]]
weight = data[:, [2]]
# : označava da gledamo vrijednost svih redata, a [1] označava drugi stupac u datoteci
plt.scatter(height, weight, marker = '.', )
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# c) Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici.

height_50 = data[: : 50, [1]]
weight_50 = data[: : 50, [2]]
# do zareza je prva dimenzija, tj. gledamo sve retke, a [1], tj. [2] znači da gledamo 2. i 3. stupac
plt.scatter(height_50, weight_50, marker = '.')
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# d)Izračunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom podatkovnom skupu.

print("Minimalna visina:", height.min())
print("Maksimalna visina:", height.max())
print("Srednja visina:", round(height.mean(), 2))


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# e) Ponovite zadatak pod d), ali samo za muškarce, odnosno žene.
#    Npr. kako biste izdvojili muškarce, stvorite polje koje zadrži bool vrijednosti i njega koristite kao indeks retka. ind = (data[:,0] == 1)

men_index = np.where(data[:, 0] == 1)
women_index = np.where(data[:, 0] == 0)

men_height = data[men_index, 1]
women_height = data[women_index, 1]

print("Minimalna visina za muškarce:", men_height.min())
print("Maksimalna visina za muškarce:", men_height.max())
print("Srednja vrijednost visine za muškarce:", men_height.mean())

print("Minimalna visina za žene:", women_height.min())
print("Maksimalna visina za žene:", women_height.max())
print("Srednja vrijednost visine za žene:", women_height.mean())