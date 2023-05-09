# Što se događa s procesom ucčnja:
# 1. ako se koristi jako velika ili jako mala veličina serije?
# 2. ako koristite jako malu ili jako veliku vrijednost stope učenja?
# 3. ako izbacite određene slojeve iz mreže kako biste dobili manju mrežu?
# 4. ako za 50% smanjite veličinu skupa za učenje?





# Jako velika veličina serije (batch size) može dovesti do bržeg učenja, ali i do značajnog gubitka memorije i usporavanja procesa
# zbog povećanog opterećenja na GPU-u. S druge strane, jako mala veličina serije može usporiti proces učenja, ali i dovesti do veće stabilnosti i generalizacije modela.

# Jako velika vrijednost stope učenja (learning rate) može dovesti do prebrzog učenja i prevelikih koraka u optimizaciji,
# što može dovesti do oscilacija i nekonvergencije. S druge strane, jako mala vrijednost stope učenja može dovesti do spore
# konvergencije ili do toga da model zaglavi u lokalnom minimumu.

# Ako izbacite određene slojeve iz mreže kako biste dobili manju mrežu, može se dogoditi da gubite neke važne značajke koje su prethodni slojevi naučili.
# To može dovesti do smanjenja performansi i generalizacije modela. Međutim, ako su izbačeni slojevi nepotrebni ili ne doprinose puno performansi,
# to može dovesti do smanjenja broja parametara i ubrzati proces učenja.

#Smanjenje veličine skupa za učenje (training set) za 50% može dovesti do smanjenja kvalitete i raznolikosti podataka koje model uči,
# što može dovesti do lošije generalizacije i performansi. Međutim, ako je originalni skup za učenje vrlo velik, smanjenje veličine skupa
# za učenje može ubrzati proces učenja bez velikog gubitka performansi.