import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.0, 2.0, 3.0, 3.0, 1.0])     # prvo pogledam gdje imam točke na X-osi (1, 2, dvije na 3, 1 (moramo se vratiti odakle smo počeli))
y = np.array([1.0, 2.0, 2.0, 1.0, 1.0])     # zatim po y osi spajam te točke ovisno o tome gdje se nalaze kada pogledamo x i y os

plt.plot(x, y, 'b', linewidth = 2 , marker = ".", markersize = 5 )
                                    # za promjenu boje u crvenu: 'r'
plt.axis([0.0, 4.0, 0.0, 4.0])      # limita x os na elemente od 0 do 4, a y-os isto na 0 do 4
                                    # to su dimenzije cijelog figure-a
plt.xlabel('x os')
plt.ylabel('y os')
plt.title('Primjer')
plt.show()