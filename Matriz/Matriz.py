from re import X
import joblib 
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers,Sequential,optimizers
import numpy as np
import matplotlib as plt
import os
from keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


MUSIC_PATH = "./AB_normalizado.csv"
MUSIC_PATH_TRANS = "yandy_data1.csv"

def load_music_data(music_path = MUSIC_PATH):
    csv_path = os.path.join(music_path)
    return pd.read_csv(csv_path,index_col=0)


#artistas = load_music_data()
#dataframe= pd.DataFrame(artistas)
##print(dataframe)
#
##dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('^Unnamed')]
##print(dataframe)
##dataframe.to_csv("normaliada_rank.csv")
#
##dft = dataframe.transpose()
##print(dft)
##dft.to_csv("normalizada_rank_trans.csv")
#
#t = artistas.drop(columns="Name")
#t.to_csv("A.csv")
#A = t.to_numpy()
#print(A)
#
#
#
#trans = load_music_datat(MUSIC_PATH_TRANS)
#dataframe = pd.DataFrame(trans)
#dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('^Unnamed')]
#print(dataframe)
#dataframe = dataframe.drop(0)
#print(dataframe)
#dataframe.to_csv("B.csv")
#B = dataframe.to_numpy()
#print(B)
#
AB = load_music_data()
AB = pd.DataFrame(AB)
AB = AB.to_numpy()
dt = load_music_data(MUSIC_PATH_TRANS)
dt = pd.DataFrame(dt)
dt = dt.drop(columns="Artist")

dt = dt.fillna(0)
#dt.to_csv("result.csv")
#print(dt)
H = dt.to_numpy()

AB = AB.flatten()
H = H.flatten()

X_train, X_test, Y_train, Y_test = train_test_split(AB, H, test_size=0.9)
train_gen = DataGenerator(X_train, Y_train, 32)
test_gen = DataGenerator(X_test, Y_test, 32)
capa = layers.Dense(units=1,input_shape=[1])
salida = layers.Dense(units=1)
modelo = Sequential(
    [capa,layers.Activation('sigmoid')]
)
modelo.compile(
    optimizer="adam",
    loss="",
     metrics=['accuracy']
)
print("Entrenado Modelo")
history = modelo.fit(train_gen,
                    epochs=10,
                    validation_data=test_gen)
print("Modelo Entrenado")
modelo.save("modelo_capa.h5")
loss , acurracy = modelo.evaluate(X_test,Y_test)
print(acurracy)


#modelo.save("modelo_0.9_10.h5")
#kk = joblib.load("modelo_entrenado1.pkl")

#modelo.evaluate(X_test,Y_test,batch_size=32)

#odelo
#kk = joblib.load("modelo_entrenado.pkl")




