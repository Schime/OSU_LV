# Ulazni sloj (Input): očekuje slike dimenzija 32x32 piksela i 3 kanala (RGB)
# Konvolucijski sloj (Conv2D) sa 32 filtera dimenzija 3x3, aktivacijskom funkcijom ReLU i "same" paddingom.
# Sloj za agregiranje (MaxPooling2D) dimenzija 2x2 koji izvlači najznačajnije značajke iz konvoluiranih slika.
# Konvolucijski sloj (Conv2D) sa 64 filtera dimenzija 3x3, aktivacijskom funkcijom ReLU i "same" paddingom.
# Sloj za agregiranje (MaxPooling2D) dimenzija 2x2.
# Konvolucijski sloj (Conv2D) sa 128 filtera dimenzija 3x3, aktivacijskom funkcijom ReLU i "same" paddingom.
# Sloj za agregiranje (MaxPooling2D) dimenzija 2x2.
# Sloj za izravnavanje (Flatten) koji transformira 3D tenzor u 1D vektor prije ulaska u potpuno povezane slojeve.
# Potpuno povezani sloj (Dense) sa 500 neurona i aktivacijskom funkcijom ReLU.
# Izlazni sloj (Dense) sa 10 neurona i aktivacijskom funkcijom Softmax.





# Proučite krivulje koje prikazuju točnost klasifikacije i prosječnu vrijednost funkcije gubitka 
# na skupu podataka za učenje i skupu podataka za validaciju. Što se dogodilo tijekom učenja 
# mreže? Zapišite točnost koju ste postigli na skupu podataka za testiranje.

# Na krivuljama koje prikazuju točnost klasifikacije i funkciju gubitka vidljivo je da se točnost na skupu podataka za učenje i
# validaciju povećava s povećanjem broja epoha, no nakon nekoliko epoha točnost na validacijskom skupu prestaje se poboljšavati.
# Ovo ukazuje na to da mreža počinje prenaučivati na skupu podataka za učenje, što se vidi po razdvojenju krivulja točnosti
# na skupu za učenje i validaciji nakon određene epohe.
# Na kraju učenja, postignuta je točnost od 76.61% na skupu podataka za testiranje.
# Ovo je prilično dobra točnost s obzirom na relativno jednostavnu arhitekturu mreže i relativno malu veličinu skupa podataka.