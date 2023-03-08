import numpy as np
import matplotlib.pyplot as plt

image = plt.imread("D:\\OTHER\\PROGRAMMING\\6. semestar\\Osnove strojnog učenja\\LV\\LV_2\\road.jpg")

# a) Posvijetli sliku
brightness_increase_index = 50
brightness_increased_image = np.clip(image.astype(np.int32) + brightness_increase_index, 0, 255).astype(np.uint8)
# image.astype(np.int32) → konverta sliku u numpy array od 32-bitna integera.
#                        → sprječava overflow kada dodajemo brightness_increase_index
# + brightness_increase_index → dodaje brightness_increase_index za svaki piksel slike
# np.clip(..., 0, 255)   → istječe svaki piksel slike osiguravajući da je svaki piksel u intervalu od 0 do 255
# .astype(np.uint8) → konverta numpy array natrag u 8-bitni unsigned int array što uzrokuje distorziju slike


# b) Prikaži samo drugu četvrtinu slike po širini
height, width, unused = image.shape
cropped_image = image[:, width // 4: width // 2, :]
# : → selecta sve retke slike
# width // 4:width // 2 → selecta podskup stupaca, posebno one koji se nalaze između četvrtine i i polovine širine slike
#                       → operator // dijeli integere (osigurava da je rezultat cijeli broj) 


# c) Zarotiraj sliku za 90 stupnjeva u smjeru kazaljke na satu
rotated_image = np.rot90(image, k = 3)
# rot90 → rotira sliku 90 stupnjeva SUPROTNO od smjera kazaljke na satu
# k → broj ponavljanja rotacije
# slika je zarotirana 3 puta suprotno od kazaljke na satu (tj. jednom u smjeru kazaljke na satu)


# d) Zrcali sliku
flipped_image = np.fliplr(image)


# grafički prikaz
plt.imshow(image)
plt.show()
plt.imshow(brightness_increased_image)
plt.show()
plt.imshow(cropped_image)
plt.show()
plt.imshow(rotated_image)
plt.show()
plt.imshow(flipped_image)
plt.show()