import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





data = pd.read_csv("D:\\OTHER\PROGRAMMING\\6. semestar\\Osnove strojnog učenja\\Vjezba_za_ispit\\LV_3\\data_C02_emission.csv")

# a) Koliko mjerenja sadrži DataFrame?
    #print(len(data))


#    Kojeg je tipa svaka veličina
    #print(data.dtypes)  ili print(data.info())


#    Postoje li izostale ili duplicirane vrijednosti?
duplicates = data[data.duplicated()]
    #print(len(duplicates))


#    Obrišite izostale ili duplicirane vrijednosti.
data.drop_duplicates()


#    Kategoričke veličine konvertirajte u tip category
kategoricke_velicine = data.select_dtypes(include=["object"]).columns
for column in kategoricke_velicine:
    data[column] = data[column].astype("object")


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# b) Koja tri automobila ima najveću odnosno najmanju gradsku potrošnju?

fuel_consumption_city = data[["Model", "Make", "Fuel Consumption City (L/100km)"]].sort_values("Fuel Consumption City (L/100km)", ascending=False)
    #print("Namnanje: ", fuel_consumption_city.tail(3))
    #print("Najvece: ", fuel_consumption_city.head(3))


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# c) Koliko vozila ima veličinu motora između 2.5 i 3.5 L? Kolika je prosječna C02 emisija plinova za ova vozila?

engine_size = data[(data["Engine Size (L)"] > 2.5) & (data["Engine Size (L)"] < 3.5)]
engine_size_count = engine_size.count().values[0]
    #print(engine_size_count)


#    Kolika je prosječna C02 emisija plinova za ta vozila?
indexes = data[(data["Engine Size (L)"] > 2.5) & (data["Engine Size (L)"] < 3.5)].index
mean_co2 = data.loc[indexes, "CO2 Emissions (g/km)"].mean()
    #print(mean_co2)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# d) Koliko mjerenja se odnosi na vozila proizvođača Audi?

audi = data[data["Make"] == "Audi"]
    #print(len(audi))


#    Kolika je prosječna emisija C02 plinova automobila proizvođača Audi koji imaju 4 cilindara?
filtered_audi_index = data[(data["Make"] == "Audi") & (data["Cylinders"] == 4)].index
mean_co2_audi = data.loc[filtered_audi_index, "CO2 Emissions (g/km)"].mean()
    #print(mean_co2_audi)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# e) Koliko je vozila s 4,6,8. . . cilindara?
four_cylinder = data[(data["Cylinders"] == 4)]
six_cylinder = data[(data["Cylinders"] == 6)]
eight_cylinder = data[(data["Cylinders"] == 8)]
ten_cylinder = data[(data["Cylinders"] == 10)]
twelve_cylinder = data[(data["Cylinders"] == 12)]
sixteen_cylinder = data[(data["Cylinders"] == 16)]
#print("4 cylinders: ", len(four_cylinder))


#    Kolika je prosječna emisija C02 plinova s obzirom na broj cilindara?
four_cylinder_index = data[(data["Cylinders"] == 4)].index
mean_co2_four_cylinder = data.loc[four_cylinder_index, "CO2 Emissions (g/km)"].mean()
    #print("Average CO2 emission for 4 cylinder cars is: ", mean_co2_four_cylinder)

six_cylinder_index = data[(data["Cylinders"] == 6)].index
mean_co2_six_cylinder = data.loc[six_cylinder_index, "CO2 Emissions (g/km)"].mean()
    #print("Average CO2 emission for 6 cylinder cars is: ", mean_co2_six_cylinder)

eight_cylinder_index = data[(data["Cylinders"] == 8)].index
mean_co2_eight_cylinder = data.loc[eight_cylinder_index, "CO2 Emissions (g/km)"].mean()
    #print("Average CO2 emission for 8 cylinder cars is: ", mean_co2_eight_cylinder)

ten_cylinder_index = data[(data["Cylinders"] == 10)].index
mean_co2_ten_cylinder = data.loc[ten_cylinder_index, "CO2 Emissions (g/km)"].mean()
    	#print("Average CO2 emission for 10 cylinder cars is: ", mean_co2_ten_cylinder)

twelve_cylinder_index = data[(data["Cylinders"] == 12)].index
mean_co2_twelve_cylinder = data.loc[twelve_cylinder_index, "CO2 Emissions (g/km)"].mean()
    #print("Average CO2 emission for 12 cylinder cars is: ", mean_co2_twelve_cylinder)

sixteen_cylinder_index = data[(data["Cylinders"] == 16)].index
mean_co2_sixteen_cylinder = data.loc[sixteen_cylinder_index, "CO2 Emissions (g/km)"].mean()
    #print("Average CO2 emission for 16 cylinder cars is: ", mean_co2_sixteen_cylinder)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# f) Kolika je prosječna gradska potrošnja u slučaju vozila koja koriste dizel, a kolika za vozila koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
#    DISCLAIMER: X ćemo označiti kao BENZIN, a Z kao DIZEL → u tablici. Ja odredio jer ne znam koji je koji pa i nije bitno

petrol_index = data[(data["Fuel Type"] == "X")].index
diesel_index = data[(data["Fuel Type"] == "Z")].index
average_city_fuel_consumption_petrol = data.loc[petrol_index, "Fuel Consumption City (L/100km)"].mean()
    #print("Average city fuel consumption for petrol vehicles: ", average_city_fuel_consumption_petrol)
average_city_fuel_consumption_diesel = data.loc[diesel_index, "Fuel Consumption City (L/100km)"].mean()
    #print("Average city fuel consumption for diesel vehicles: ", average_city_fuel_consumption_diesel)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najveću gradsku potrošnju goriva?

four_cylinder_diesel = data[(data["Cylinders"] == 4) & (data["Fuel Type"] == "Z")].sort_values("Fuel Consumption City (L/100km)", ascending=False)
    #print("\n", four_cylinder_diesel.head(1))


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# h) Koliko vozila ima ručni tip mjenjača (bez obzira na broj brzina)?

manual = data[(data["Transmission"]).str.startswith("AM")]
    #print(len(manual))


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# i) Izračunajte korelaciju između numeričkih veličina. Komentirajte dobiveni rezultat.

    #print(data.corr(numeric_only=True))


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# DODATNI ZADACI ZA VJEŽBU → koristeći numpy


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Koliko ima vrijednosti u "Fuel Consumption Comb (L/100km)" koje su veće od 10?
# Ukoliko postoje, obriši ih.

new_data = np.genfromtxt("D:\\OTHER\PROGRAMMING\\6. semestar\\Osnove strojnog učenja\\Vjezba_za_ispit\\LV_3\\data_C02_emission.csv", delimiter=",", skip_header=1)

fuel_consumption_comb = new_data[:, 9]

fuel_consumption_comb_over_10_index = np.where(fuel_consumption_comb > 10)[0]

new_filtered_data = np.delete(new_data, fuel_consumption_comb_over_10_index, axis=0)

# Komentar:
# new_data je isto što i data samo što sam u preko numpyja pristupio new_data umjesto pandasom

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Koristeći funkciju scatter, prikaži odnos "Engine Size (L)" i "Fuel Consumption City (L/100km)" koristeći nove podatke.

engines = new_filtered_data[:, 3]
fuel_cons_city = new_filtered_data[:, 7]
plt.scatter(engines, fuel_cons_city)
plt.title("Odnos veličine motora i gradske potrošnje")
plt.xlabel("Veličina motora")
plt.ylabel("Gradska potrošnja")
plt.xlim(0)
plt.ylim(0)
plt.show()

# Komentar:
# Vidimo da graf doseže veličinu motora 3.5 jer smo koristili nove podatke.
# Kada smo profiltrirali "Fuel Consumption City (L/100km)" u prošlom zadatku, obrisali smo sve one preko 10 pa time i veličine motora preko 3.5 jer veći motori troše puno


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Koristi originalnu tablicu!
# Odredi maksimalu, minimalnu i srednju gradsku potrošnju goriva za automobile koji imaju broj cilindara veći od 6, 

cylinders = new_data[:, 4]
more_than_6_cylinders_index = np.where(cylinders > 6)[0]

more_than_6_cylinders_fuel_consumption_city = new_data[more_than_6_cylinders_index, 7]

print("Maksimalna gradska potrošnja goriva za automobile koji imaju više od 6 cilindara:", np.max(more_than_6_cylinders_fuel_consumption_city))
print("Minimalna gradska potrošnja goriva za automobile koji imaju više od 6 cilindara:", np.min(more_than_6_cylinders_fuel_consumption_city))
print("Srednja vrijednost gradske potrošnje goriva za automobile koji imaju više od 6 cilindara:", np.mean(more_than_6_cylinders_fuel_consumption_city))


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Pomoću stupčastog dijagrama prikažite postotke auta s 4 i 8 cilindara

cylinder_4 = data[(data["Cylinders"] == 4)]
cylinder_8 = data[(data["Cylinders"] == 8)]
cylinder_16 = data[(data["Cylinders"] == 16)]

cylinder_4_percent = len(cylinder_4) / len(data["Cylinders"] == 4) * 100
cylinder_8_percent = len(cylinder_8) / len(data["Cylinders"] == 8) * 100
cylinder_16_percent = len(cylinder_16) / len(data["Cylinders"] == 16) * 100

labels = ["4 cylinder", "8 cylinder", "16 cylinder"]
percentages = [cylinder_4_percent, cylinder_8_percent, cylinder_16_percent]
plt.bar(labels, percentages)
plt.title("Postotak auta s 4 cilindra i auta s 8 cilindra")
plt.xlabel("Cilindri")
plt.ylabel("Postotak auta")
plt.show()
