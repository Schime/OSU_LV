import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering



def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# generiranje podatkovnih primjera
X = generate_data(500, 3)

# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Primijenite metodu K srednjih vrijednosti te ponovo prikažite primjere, ali svaki primjer obojite ovisno o njegovoj pripadnosti pojedinoj grupi.
# Nekoliko puta pokrenite programski kod. Mijenjate broj K. Što primjećujete?

kmeans = KMeans(n_clusters=3, init="random", n_init=5, random_state=0)
kmeans.fit(X)
plt.figure()
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()

# Komentar:
# n_clusters = parametar K → određuje broj grupa
# Primjećujemo da različiti brojevi grupa mogu rezultirati različitim grupiranjem podataka i različitim vizualizacijama.


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# MOJ DODATNI ZADATAK:
# Pronađi centre grupa te ih označi crnim znakom X

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=100, linewidths=2, color='black')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Mijenjajte način definiranja umjetnih primjera te promatrajte rezultate grupiranja podataka (koristite optimalni broj grupa).
# Kako komentirate dobivene rezultate?

# Pronalazimo optimalni broj grupa pomoću LAKAT METODE (engl. elbow metghod)
ssd = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    ssd.append(kmeans.inertia_)

# Plot
plt.plot(K, ssd, 'bx-')
plt.xlabel('Broj grupa')
plt.ylabel('SSD')
plt.title('Lakat metoda')
plt.show()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# MOJ DODATNI ZADATAK:
# U konzolu ispiši optimalan broj grupa te označi crvenom točkicom LAKAT

# kreiraj prazne liste za spremanje inercija i broja grupa
inertias = []
num_clusters = range(1, 11)

# izvrši KMeans za različit broj grupa i izračunaj inerciju za svaki model
for k in num_clusters:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# ispiši optimalan broj grupa koristeći lakat metodu
diff = np.diff(inertias)
diff_r = diff[1:] / diff[:-1]
k_opt = num_clusters[np.argmin(diff_r) + 1]

print("Optimalan broj grupa K:", k_opt)

# prikazi graf inercije
plt.plot(num_clusters, inertias, marker='o')
plt.xlabel('Broj grupa')
plt.ylabel('Inercija')
plt.title('Metoda lakta')
plt.plot(k_opt, inertias[k_opt-1], 'ro')
plt.show()