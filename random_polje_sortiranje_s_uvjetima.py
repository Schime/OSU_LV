# Napravi polje od 10 random elemenata, sortiraj polje te napravi novo polje u koje idu elementi koji su veci odo 0.5 iz prvog polja. Ispisi novo polje

import numpy as np

np.random.seed(100)

a = np.random.rand(10)
print(np.sort(a))

b = a[a > 0.5]
print(b)