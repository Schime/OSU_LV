import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv("data_C02_emission.csv")


make = np.array(data.iloc[:, 0])       # nulti stupac u tablici uzima
cylinder = np.array(data.iloc[:, 4])

acura = make[make == "Acura"]

sorted_cylinders = cylinder[cylinder > 4]

#print(data[(data["Cylinders"] > 8)].Make)

data.plot.scatter(x="Cylinders", y="Make", c="Engine Size (L)")
plt.show()