import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.image as Image



# Učitavanje modela
model = tf.keras.models.load_model(r"FCN")

# Učitavanje slike
img = Image.imread(r"test.png")[:, :, 0]

# Normalizacija (ja msm od 0-1 da bude)
normalized_img = tf.keras.utils.normalize(img, axis=1)

# Shape slike je (28,28) a pretavaramo je u (1,784), jedna slika i pixeli se množe
normalized_img = normalized_img.reshape(1, 784)
plt.imshow(img, cmap='gray')
plt.show()

predictions = model.predict(normalized_img)
print(f"Predicted: {np.argmax(predictions)}")