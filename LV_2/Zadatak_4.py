import numpy as np
import matplotlib.pyplot as plt



black_square = np.zeros((50, 50), dtype = np.uint8)
# zeros daje matricu nula dimenzije 50×50

white_square = np.ones((50, 50), dtype = np.uint8) * 255
# ones daje matricu jedinica dimenzije 50×50
# *255 postavlja sve vrijednosti na *255 što daje bijelu

top_squares = np.hstack((black_square, white_square))
# funkcija hstack postavi određuje koje će boje biti prvi kvadrat (u našem slučaju crni)
bottom_squares = np.hstack((white_square, black_square))

checkerboard = np.vstack((top_squares, bottom_squares))
# prvo postavi gornji pa onda donji red i vraća 2D matricu
# vstack → funkcija koja stacka polje vertikalno
# argumenti funkcije predstavljaju gornu i donju polovicu slike


plt.imshow(checkerboard, cmap='gray')
plt.show()