import numpy as np
import matplotlib.pyplot as plt


filename = "D:\\OTHER\PROGRAMMING\\6. semestar\\Osnove strojnog učenja\\LV\\LV_2\\data.csv"
data = np.loadtxt(filename, delimiter = ',', skiprows = 1)
# pošto radimo sa CSV (comma-seperated file), znamo da su podaci odijeljeni zarezom te ga delimiterom odjeljujemo
# skiprows označava broj redaka od početka čitanja dokumenta (1. redak u file-u ne čitamo)

# a) Na temelju veličine numpy polja data, na koliko osoba su izvršena mjerenja?
print("Broj ljudi: ", data.shape[0])
# gledamo koliko ljudi ima po spolu (muško + žensko), što je nulti stupac u tablici


# b) Odnos visine i mase osobe
height = data[:, [1]]
weight = data[:, [2]]
# : označava da gledamo vrijednost svih redata, a [1] označava drugi stupac u datoteci
plt.scatter(height, weight, marker = '.', )
plt.xlabel("height")
plt.ylabel("weight")
plt.show()


# c) Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici.
height_50 = data[: : 50, [1]]
weight_50 = data[: : 50, [2]]
# do zareza je prva dimenzija, tj. gledamo sve retke, a [1], tj. [2] znači da gledamo 2. i 3. stupac
plt.scatter(height_50, weight_50, marker = '.', )
plt.xlabel("height")
plt.ylabel("weight")
plt.show()


# d)Izračunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom podatkovnom skupu.
print("Minimalna visina: ", height.min())
print("Maksimalna visina: ", height.max())
print("Srednja visina: ",round(height.mean(),2))