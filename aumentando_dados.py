# Load the required packages
from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
#from keras.preprocessing.image import load_img
import numpy as np

#Let's define the input dir path & Output dir path where generated image will store
#src_path = './treino/plantas/th (2).jpg'
des_pat = 'C:/Users/Usuario/Desktop/treino/pedras'
des_path = 'C:/Users/Usuario/Desktop/treino/plantas'

# Let's load input image
#image = load_img(src_path)
#image = img_to_array(image)
#image = np.expand_dims(image, axis=0)

# Let's define ImageDataGenerator class
aug = ImageDataGenerator(
 rotation_range=30,
 zoom_range=0.15,
 width_shift_range=0.2,
 height_shift_range=0.2,
 shear_range=0.15,
 horizontal_flip=True,
 fill_mode="nearest")

# Let's apply ImageDataGenerator to input image
#imageGen = aug.flow(image, batch_size=1, save_to_dir=des_path,save_prefix="image", save_format="jpg")

# Define number of augmented image which you want to download and iterate through loop
total_image = 5
j = 1
for j in range(6):
    # Let's load input image
    scr_path = './treino/plantas/th (' + str(j + 1) + ').jpg'
    image = load_img(scr_path)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    imageGen = aug.flow(image, batch_size=1, save_to_dir=des_path,save_prefix="image", save_format="jpg")
    i = 0
    for e in imageGen:
        if (i == total_image):
            break
        i = i + 1

