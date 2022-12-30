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
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

genderXTrain = None
genderYTrain = None
genderXTest = None
genderYTest = None


class NN_CNN():



        def readData(self):
            #global genderXTest
            

            #self.splitTrainTest()
            self.loadGenderTrainingData()
            self.loadGenderTestData()
        
            self.sequentialGender_CNN()

        def splitTrainTest(self):
            df = pd.read_csv('C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\VoxCeleb_gender\\deneme\\neutral\\output_females.csv')

            msk = np.random.rand(len(df)) <= 0.7

            train = df[msk]
            test = df[~msk]
            train.to_csv('gender_train.csv', index=False)
            test.to_csv('gender_test.csv', index=False)



        def loadGenderTrainingData(self):
            global genderXTrain, genderYTrain
            df: ndarray = pd.read_csv('C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\gender_train.csv').values
            genderXTrain = df[:, :13]
            genderYTrain = np.int_(df[:, 13])

        def loadGenderTestData(self):
            global genderXTest, genderYTest
            df = pd.read_csv('C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\gender_test.csv').values
            genderXTest = df[:, :13]
            genderYTest = np.int_(df[:, 13])


        def sequentialGender_CNN(self):
            model = models.Sequential()
            model.add(layers.Conv1D(512, kernel_size=5, strides=1,
                                    padding="same", activation="relu",
                                    input_shape=(genderXTrain.shape[1], 1)))
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
            model.add(layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

            model.add(layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling1D(pool_size=3, strides = 2, padding = 'same'))

            model.add(layers.Flatten())
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dense(1, activation="sigmoid"))

            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            model.fit(genderXTrain, genderYTrain, epochs=5, validation_split=0.2, batch_size=32)
            callbacks=keras.callbacks.EarlyStopping(verbose=1, patience=2),

            # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,             #
            # 													patience=5, min_lr=0.001)
            
            model.summary()


                #model.evaluate(genderXTest, genderYTest)
            model.evaluate(genderXTest, genderYTest)
            y_pred = model.predict(genderXTest)
            confusion_matrix = metrics.confusion_matrix(genderYTest, y_pred.round())
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
            cm_display.plot()
            plt.show()
