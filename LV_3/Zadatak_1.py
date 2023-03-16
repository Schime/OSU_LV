import numpy as np
import pandas as pd



data = pd.read_csv("data_C02_emission.csv")

# a)
print("a)")
print("Number of rows in the DataFrame:", data.shape[0])    # daje broj redaka (2213 je ukupno no ne računamo prvi redak)

print(data.dtypes)      # daje tip podatka

# traži duplikate
duplicates = data.duplicated()
print("Number of duplicated rows:", duplicates.sum())

# traži izostavljene vrijednosti
missing = data.isnull().sum()
print("Number of missing values in each column:")
print(missing)

# konverta kategoričke veličine u tip 'category'
columns_to_convers = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
for column in columns_to_convers:
    data[column] = data[column].astype('category')

print("Data types of each column:")
print(data.dtypes)



# b)
# Koja tri automobila ima najveću odnosno najmanju gradsku potrošnju? Ispišite u terminal:
# ime proizvođača, model vozila i kolika je gradska potrošnja.
print("\n\nb)")
lowest_city_consumption = data.sort_values(by="Fuel Consumption City (L/100km)", ascending=True)
lowest_city_consumption = lowest_city_consumption[['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(3)

highest_city_consumption = data.sort_values(by="Fuel Consumption City (L/100km)", ascending=False)
highest_city_consumption = highest_city_consumption[['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(3)

print("Three vehicles with the lowest city fuel consumption:")
for i, row in lowest_city_consumption.iterrows():
    make = row['Make']
    model = row['Model']
    consumption = row["Fuel Consumption City (L/100km)"]
    print(f"{i + 1}. {make} {model}: {consumption:.2f} L/100km")

print("\nThree vehicles with the highest city fuel consumption:")
for i, row in highest_city_consumption.iterrows():
    make = row['Make']
    model = row['Model']
    consumption = row['Fuel Consumption City (L/100km)']
    print(f"{i + 1}. {make} {model}: {consumption:.2f} L/100km")

# iterrows() je Pandas funkcija koja omogućuje iteraciju preko redaka DataFramea kao parova (indeks, serija)
# vraća iterator koji daje vrijednost indeksa i vrijednost retka kao Pandas Series 



# c)
# Koliko vozila ima veličinu motora između 2.5 i 3.5 L? Kolika je prosječna C02 emisija plinova za ova vozila?
print("\n\nc)")
filtered_data = data[(data["Engine Size (L)"] >= 2.5) & (data["Engine Size (L)"] <= 3.5)]
number_of_filtered_vehicles = len(filtered_data)

avg_co2_emission = filtered_data['CO2 Emissions (g/km)'].mean()

print(f"Number of vehicles with engine size between 2.5 and 3.5 L: {number_of_filtered_vehicles}")
print(f"Average CO2 emission for these vehicles: {avg_co2_emission:.2f} g/km")



# d)
# Koliko mjerenja se odnosi na vozila proizvočača Audi?
# Kolika je prosječna emisija C02 plinova automobila proizvođača Audi koji imaju 4 cilindara?
print("\n\nd)")
audi_data = data[data['Make'] == 'Audi']
num_measurements_audi = len(audi_data)
avg_co2_emission_audi = audi_data[audi_data['Cylinders'] == 4]['CO2 Emissions (g/km)'].mean()

print(f"Number of measurements related to Audi vehicles: {num_measurements_audi}")
print(f"Average CO2 emission of Audi vehicles with 4 cylinders: {avg_co2_emission_audi:.2f} g/km")



# e)
# Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosječna emisija C02 plinova s obzirom na broj cilindara?
print("\n\ne)")
grouped_by_cylinders = data.groupby('Cylinders')
vehicle_count_cylinders = grouped_by_cylinders.size()
avg_co2_emission_cylinder = grouped_by_cylinders['CO2 Emissions (g/km)'].mean()

print("Vehicle counts by number of cylinders:")
print(vehicle_count_cylinders)
print("\nAverage CO2 emissions by number of cylinders:")
print(avg_co2_emission_cylinder)



# f)
# Kolika je prosječna gradska potrošnja u slučaju vozila koja koriste dizel, a kolika za vozila
# koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
print("\n\nf)")
print("Average city fuel consumption for diesel vehicles:", round((data['Fuel Consumption City (L/100km)'].where(data['Fuel Type'] == 'D')).mean(), 2))
print("Average city fuel consumption for gasoline vehicles:", round((data['Fuel Consumption City (L/100km)'].where(data['Fuel Type'] == 'X')).mean(), 2))

dieselMotors = data[data['Fuel Type'] == 'D']
gasolineMotors = data[data['Fuel Type'] == 'X']

dieselMotorsAmount = len(dieselMotors)
gasolineMotorsAmount = len(gasolineMotors)

dieselMotors = dieselMotors.sort_values('Fuel Consumption City (L/100km)',ascending=False)
gasolineMotors = gasolineMotors.sort_values('Fuel Consumption City (L/100km)',ascending=False)

if dieselMotorsAmount%2 == 0:
    print("Diesel median: ",(dieselMotors['Fuel Consumption City (L/100km)'].iloc[int(dieselMotorsAmount/2)]+dieselMotors['Fuel Consumption City (L/100km)'].iloc[int(dieselMotorsAmount/2) + 1])/2)
else:
    print("Diesel median: ",dieselMotors['Fuel Consumption City (L/100km)'].iloc[dieselMotorsAmount/2])

if gasolineMotorsAmount%2 == 0:
    print("Diesel median: ",(gasolineMotors['Fuel Consumption City (L/100km)'].iloc[int(gasolineMotorsAmount/2)]+gasolineMotors['Fuel Consumption City (L/100km)'].iloc[int(gasolineMotorsAmount/2) + 1])/2)
else:
    print("Diesel median: ",gasolineMotors['Fuel Consumption City (L/100km)'].iloc[gasolineMotorsAmount/2])



# g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najveću gradsku potrošnju goriva?
print("\n\ng)")
print("Dizel vozilo koje najvise trosi: \n", dieselMotors[dieselMotors['Cylinders'] == 4].head(1))



# h) Koliko ima vozila ima ručni tip mjenjača (bez obzira na broj brzina)?
print("\n\nh)")
print("Vozila sa rucnim mjenjacem (M): ", len(data[data['Transmission'].str.contains('M')]))



# i) Izračunajte korelaciju između numeričkih veličina. Komentirajte dobiveni rezultat.
print("Numericka korelacija: ", data.corr(numeric_only=True))