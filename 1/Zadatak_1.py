import numpy as np
import matplotlib.pyplot as plt



data = np.genfromtxt("file.csv", delimiter=",", skip_header=9)

# Na temelju veličine numpy polja data, na koliko osoba su izvršena mjerenja?

data_without_header = data[:, 0]

print(data_without_header.size)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Postoje li izostale ili duplicirane vrijednosti u stupcima s mjerenjima dobi i indeksa tjelesne mase (BMI)?
# Obrišite ih ako postoje. Koliko je sada uzoraka mjerenja preostalo?

# MOJA preformulacija:
# Izbaci sve retke u tablici koji imaju bmi = 0

bmi_data = data[:, 5]

zero_bmi_index = np.where(bmi_data == 0)[0]

new_filtered_data = np.delete(data, zero_bmi_index, axis=0)

# Komentar:
# new_filtered_data → izbacili smo dijelove gdje je BMI = 0, jer ne može biti (to su krivi podaci).


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Prikažite odnos dobi i indeksa tjelesne mase (BMI) osobe pomoću scatter dijagrama.
# Dodajte naziv dijagrama i nazive osi s pripadajućim mjernim jedinicama.
# Komentirajte odnos dobi i BMI prikazan dijagramom.

age_data = new_filtered_data[:, 7]
new_bmi_data = new_filtered_data[:, 5]
plt.scatter(age_data, new_bmi_data)
plt.title("Odnos dobi i BMI osobe")
plt.xlabel("Dob")
plt.ylabel("BMI")
plt.xlim(0)
plt.ylim(0)
plt.show()

# Komentar:
# Koristili smo novu tablicu (bez lažnih BMI-a) te iz te nove tablice odabrali stupce age i bmi.
# xlim i ylim određuju odakle kreće graf.


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Izračunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost indeksa tjelesne mase (BMI) u ovom podatkovnom skupu.

print("Minimalan BMI:", new_bmi_data.min())
print("Maksimalan BMI:", new_bmi_data.max())
print("Srednja vrijednost BMI-a:", new_bmi_data.mean())


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Ponovite zadatak pod d), ali posebno za osobe kojima je dijagnosticiran dijabetes i za one kojima nije.
# Kolikom je broju ljudi dijagonosticiran dijabetes? Komentirajte dobivene vrijednosti.

diabetes_positive_index = np.where(new_filtered_data[:, 8] == 1)
diabetes_negative_index = np.where(new_filtered_data[:, 8] == 0)

positive_diabetic_people_bmi = new_filtered_data[diabetes_positive_index, 5]
negative_diabetic_people_bmi = new_filtered_data[diabetes_negative_index, 5]

print("Maksimalan BMI za osobe s dijabetesom:", np.max(positive_diabetic_people_bmi))
print("Minimalan BMI za osobe s dijabetesom:", np.min(positive_diabetic_people_bmi))
print("Srednja vrijednost BMI-a za osobe s dijabetesom:", np.mean(positive_diabetic_people_bmi))

print("Maksimalan BMI za osobe bez dijabetesom:", np.max(negative_diabetic_people_bmi))
print("Minimalan BMI za osobe bez dijabetesom:", np.min(negative_diabetic_people_bmi))
print("Srednja vrijednost BMI-a za osobe bez dijabetesom:", np.mean(negative_diabetic_people_bmi))

# Komentar:
# Prvo smo pronašli INDEKSE osoba koje su dijabetične (imaju 1 u zadnjem stupcu), i koje nisu (imaju 0 u zadnjem stupcu)
# positive_diabetic_people_bmi → spremili smo vrijednosti 6. stupca (BMI) za osobe s dijabetesom (pronašli po indeksu)
# negative_diabetic_people_bmi → spremili smo vrijednosti 6. stupca (BMI) za osobe bez dijabetesom (pronašli po indeksu)
