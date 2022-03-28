"""# Partie 1 : Création du dataset"""

import os
from keras.preprocessing import image

folders = os.listdir('dataset')

image_data = []
labels = []
count = 0

for ix in folders:
    path = os.path.join("/dataset", ix)
    print(path, count)
    if(path!="/content/drive/MyDrive/Dataset/.ipynb_checkpoints"):
      for im in os.listdir(path):
          try:
              img = image.load_img(os.path.join(path,im), target_size = (224,224))
              img_array = image.img_to_array(img)
              image_data.append(img_array)
              labels.append(count)
          except:
              pass
      count += 1

#Ici on melange le dataset
import random

combined_dataset = list(zip(image_data, labels))
random.shuffle(combined_dataset)
image_data[:], labels[:] = zip(*combined_dataset)

# Convertissement du dataset en array
import numpy as np
from keras.utils import np_utils
X_train = np.array(image_data)
Y_train = np.array(labels)
Y_train = np_utils.to_categorical(Y_train)

"""# Partie 2 : modele ResNet50"""

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np
print("Imported Successfully!")

model = ResNet50(include_top = False, weights = 'imagenet', input_shape = (224,224,3)) #False car on donne au model notre propre classificateur
print(model.summary())

"""Création de notre propre classificateur"""

av1 = GlobalAveragePooling2D()(model.output)
fc1 = Dense(256, activation = 'relu')(av1) #Dense layer de 256 neurones et relu comme fonction d'activation
d1 = Dropout(0.5)(fc1) #Dropout de 0.5 veut dire que certain neurones serons aleatoirment non activés afin d'eviter le surapprentissage
fc2 = Dense(10, activation = 'softmax')(d1) #nb de classe d'images / nb de Pokemon

"""Connection entre ResNet50 et le classificateur"""

model_new = Model(inputs = model.input, outputs = fc2) # Model() est une API de Keras, afin de definir les input et output  du model
model_new.summary()

"""# Partie 3 : Prediction"""

from tensorflow.keras.applications.resnet50 import preprocess_input

image_path = 'salameche2.jpg'
img = image.load_img(image_path,target_size = (224,224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)

pred = model_new.predict(x)
print(np.argmax(pred))

adam = Adam(lr = 0.00003)
model_new.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

for ix in range(len(model_new.layers)):
    print(ix, model_new.layers[ix])

for ix in range(169):
    model_new.layers[ix].trainable = False
print(model_new.summary())

hist = model_new.fit(X_train, Y_train, shuffle = True, batch_size = 16, epochs = 8, validation_split = 0.20) #entrainement du nouveau modele

image_path = 'carapuce.png'
img = image.load_img(image_path,target_size = (224,224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)

pred = model_new.predict(x)
print(np.argmax(pred))