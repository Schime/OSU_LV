import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans




# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Koliko je različitih boja prisutno u ovoj slici?

colors = np.unique(img_array_aprox, axis=0)
print("Broj različitih boja na slici:", len(colors))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Primijenite algoritam K srednjih vrijednosti koji će pronaći grupe u RGB vrijednostima elemenata originalne slike.

km = KMeans(n_clusters=5, init="random", n_init=5, random_state=0)
km.fit(img_array_aprox)
labels = km.predict(img_array_aprox)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Vrijednost svakog elementa slike originalne slike zamijeni s njemu pripadajućim centrom.

newimg = km.cluster_centers_[labels]
newimg = np.reshape(newimg, (img.shape))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Usporedite dobivenu sliku s originalnom. Mijenjate broj grupa K. Komentirajte dobivene rezultate.

f, axarr = plt.subplots(2, 1)
axarr[0].imshow(img)
axarr[1].imshow(newimg)
plt.show()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Primijenite postupak i na ostale dostupne slike.
for i in range (2, 7):
    img = Image.imread(f"imgs\\test_{i}.jpg")        
    img = img.astype(np.float64) / 255
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))
    img_array_aprox = img_array.copy()
    km = KMeans(n_clusters=4, init='random', n_init=5, random_state=0)
    km.fit(img_array_aprox)
    labels = km.predict(img_array_aprox)
    newimg = km.cluster_centers_[labels]
    newimg = np.reshape(newimg, (img.shape))
    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(img)
    axarr[1].imshow(newimg)
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Grafički prikažite ovisnost J o broju grupa K. Koristite atribut inertia objekta klase KMeans. Možete li uočiti lakat koji upućuje na optimalni broj grupa?

img = Image.imread("imgs\\test_1.jpg")
img = img.astype(np.float64) / 255
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
img_array_aprox = img_array.copy()
Ks = range(1, 11)
Js = []
for i in Ks:
    km = KMeans(n_clusters=i, init='random', n_init=5, random_state=0)
    km.fit(img_array_aprox)
    Js.append(km.inertia_)
plt.plot(Ks, Js)
plt.show()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Elemente slike koji pripadaju jednoj grupi prikažite kao zasebnu binarnu sliku. Što primjećujete?

unique_labels = np.unique(labels)
print(unique_labels)

f, axarr = plt.subplots(2, 2)

for i in range(len(unique_labels)):
    bit_values = labels==[i]
    bit_img = np.reshape(bit_values, (img.shape[0:2]))
    bit_img = bit_img*1
    x=int(i/2)
    y=i%2
    axarr[x, y].imshow(bit_img)

plt.show()