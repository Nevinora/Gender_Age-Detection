import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from numpy import ndarray
import keras
from sklearn.metrics import confusion_matrix
import torch
import torch.optim as optim
import torch.nn as nn
import os
import CNN
from keras import datasets, layers, models

ageXTrain = None
ageYTrain = None
ageXTest = None
ageYTest = None


class NN_CNN_age():



        def readData(self):
            #global genderXTest
            

            #self.splitTrainTest()
            self.loadageTrainingData()
            self.loadageTestData()
        
            self.sequentialage_CNN()

        def splitTrainTest(self):
            df = pd.read_csv('C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\age\\output_age.csv')

            msk = np.random.rand(len(df)) <= 0.7

            train = df[msk]
            test = df[~msk]
            train.to_csv('age_train.csv', index=False)
            test.to_csv('age_test.csv', index=False)



        def loadageTrainingData(self):
            global ageXTrain, ageYTrain
            df: ndarray = pd.read_csv('C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\age_train.csv').values
            ageXTrain = df[:, :13]
            ageYTrain = np.int_(df[:, 13])

        def loadageTestData(self):
            global ageXTest, ageYTest
            df = pd.read_csv('C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\age_test.csv').values
            ageXTest = df[:, :13]
            ageYTest = np.int_(df[:, 13])


        def sequentialage_CNN(self):
            model = models.Sequential()
            model.add(layers.Conv1D(512, kernel_size=5, strides=1,
                                    padding="same", activation="relu",
                                    input_shape=(ageXTrain.shape[1], 1)))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPool1D(pool_size=5, strides=2, padding="same"))

            model.add(layers.Conv1D(512, kernel_size=5, strides=1,
                                    padding="same", activation="relu"))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPool1D(pool_size=5, strides=2, padding="same"))

            model.add(layers.Conv1D(256, kernel_size=5, strides=1,
                                    padding="same", activation="relu"))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPool1D(pool_size=5, strides=2, padding="same"))
            layers.Dropout(0.5)
            model.add(layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

            model.add(layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
            layers.Dropout(0.2)
            model.add(layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling1D(pool_size=3, strides = 2, padding = 'same'))

            model.add(layers.Flatten())
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(3, activation="softmax"))

            model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            model.fit(ageXTrain, ageYTrain, epochs=5, validation_split=0.2, batch_size=16)
            callbacks=keras.callbacks.EarlyStopping(verbose=1, patience=2),

            # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,             #
            # 													patience=5, min_lr=0.001)
            
            model.summary()


                #model.evaluate(genderXTest, genderYTest)
            model.evaluate(ageXTest, ageYTest)
            # y_pred = model.predict(ageXTest)
