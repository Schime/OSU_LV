import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




data = pd.read_csv("file.csv")



# Za koliko osoba postoje podatci u ovom skupu podataka?

print("Ukupan broj osoba:", len(data))


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Koliko je osoba preživjelo potonuće broda?

survived = data[(data["Survived"]==1)]
print("Preživjelo:", len(survived))


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Pomoću stupčastog dijagrama prikažite postotke preživjelih muškaraca i žena. Dodajte nazive osi i naziv dijagrama.
# Komentirajte korelaciju spola i postotka preživljavanja.

survived_male = data[(data["Sex"] == "male") & data["Survived"] == 1]
survived_female = data[(data["Sex"] == "female") & data["Survived"] == 1]

survived_male_percent = len(survived_male) / len(data["Sex"] == "male") * 100
survived_female_percent = len(survived_female) / len(data["Sex"] == "female") * 100

labels = ["Muškarci", "Žene"]
percentages = [survived_male_percent, survived_female_percent]
plt.bar(labels, percentages)
plt.title("Postotak preživjelih po spolu")
plt.xlabel("Spol")
plt.ylabel("Postotak preživjelih")
plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Kolika je prosječna dob svih preživjelih žena, a kolika je prosječna dob svih preživjelih muškaraca?

survived_male_index = data[(data["Sex"] == "male")].index
survived_female_index = data[(data["Sex"] == "female")].index

average_age_survived_male = data.loc[survived_male_index, "Age"].mean()
print("Prosječna dob svih preživjelih muškaraca:", average_age_survived_male)

average_age_survived_female = data.loc[survived_female_index, "Age"].mean()
print("Prosječna dob svih preživjelih žena:", average_age_survived_female)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Koliko godina ima najmlađi preživjeli muškarac u svakoj od klasa? Komentirajte.






#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# MOJ DODATNI ZADATAK
# Izbriši sve podatke kod kojih ne pišu godine osobe jer ti podaci nisu validni

new_data = data.dropna(subset=["Age"])

# Komentar:
# Funkcija dropna izbacuje sve retke iz tablice koji u stupcu "Age" imaju praznu ćeliju
