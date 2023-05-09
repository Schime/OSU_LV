import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





data = pd.read_csv("data_C02_emission.csv")


# a) Pomoću histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz.

plt.figure()
data["CO2 Emissions (g/km)"].plot(kind="hist")
plt.title("Emisija CO2 plinova")
plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# b) Pomoću dijagrama raspršenja prikažite odnos između gradske potrošnje goriva i emisije C02 plinova.
#    Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose između veličina, obojite točkice na dijagramu raspršenja s obzirom na tip goriva.

# Prvo pronalazim koliko različitih tipova goriva imam
unique_fuel_types = data["Fuel Type"].unique()
    #print(unique_fuel_types)       # da vidim koja su slova u pitanju

plt.figure()
colors = {"Z" : "red", "X" : "blue", "D" : "green", "E" : "black"}
x = np.array(data.iloc[:, 7])
y = np.array(data.iloc[:, 11])
def get_color(fuel_type):
    return colors.get(fuel_type)
colors = data["Fuel Type"].apply(get_color)
plt.scatter(x, y, c = colors)
plt.xlabel("Gradska potrošnja (L/100km)")
plt.ylabel("CO2 emisije (g/km)")
plt.show()

# Komentar:
# Ako želim napisati: data.plot.scatter(x="Fuel Consumption City (L/100km)", y="CO2 Emissions (g/km)", c=colors)
# c ne može biti "Fuel Type" jer u tom stupcu imam X i Z, što nisu brojevi. c može raditi s brojevima jer ih on oboja, ali ne može obojati slova, logično


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# c) Pomoću kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip goriva.

grouped_by_cylindedrs = data.groupby("Cylinders")
grouped_by_cylindedrs.boxplot (column = ["CO2 Emissions (g/km)"])
plt.xlabel("Fuel Type")
plt.ylabel("Fuel Consumption Hwy (L/100km)")
plt.title("Distribution of Fuel Consumption Hwy by Fuel Type")
plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# d) Pomoću stupčastog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu groupby.

data.groupby("Fuel Type")["Make"].count().plot(kind="bar")
plt.title("Broj vozila po tipu goriva")
plt.xlabel("Tip goriva")
plt.ylabel("Broj vozila")
plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# e) Pomoću stupčastog grafa prikažite na istoj slici prosječnu C02 emisiju vozila s obzirom na broj cilindara.
average_co2_by_cylinders = data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean()
plt.bar(average_co2_by_cylinders.index, average_co2_by_cylinders.values)
plt.title("Average CO2 Emissions by Number of Cylinders")
plt.xlabel("Number of Cylinders")
plt.ylabel("CO2 Emissions (g/km)")
plt.show()

# Komentar:
# data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean().plot(kind="bar") → NE MOŽE OVAKO