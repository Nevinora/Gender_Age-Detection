# import modules
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
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

genderXTrain = None
genderYTrain = None
genderXTest = None
genderYTest = None


class NN():



	def readData(self):
		#global genderXTest
		

		#self.splitTrainTest()
		self.loadGenderTrainingData()
		self.loadGenderTestData()
	
		self.sequentialGender()

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



		
	def sequentialGender(self):
		print(genderYTrain)
		print(genderYTest)
		if not os.path.exists("C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\models\\saved_model.pb"):
			print("model yok")
			model = keras.Sequential([keras.layers.InputLayer(input_shape=genderXTrain.shape[1:]),
										keras.layers.Dense(32, activation='relu'),
										keras.layers.Dense(64, activation='relu'),
										keras.layers.Dense(64, activation='relu'),
										keras.layers.Dense(32, activation='relu'),
										keras.layers.Dropout(0.2),                                  #
										keras.layers.Dense(16, activation='relu'),
										# keras.layers.Dropout(0.2),
										keras.layers.Dense(8, activation='relu'),
										# keras.layers.Dense(11, activation='relu'),
										keras.layers.Dense(1, activation='sigmoid')
										])
			model.save("C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\models")
		else:
			print("model var")
			model = keras.models.load_model("C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\models")


		model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,             #
														patience=5, min_lr=0.001)
		model.summary()

		model.fit(genderXTrain, genderYTrain, epochs=50, validation_split=0.2, batch_size=64)
		#model.evaluate(genderXTest, genderYTest)
		model.evaluate(genderXTest, genderYTest)

		y_pred = model.predict(genderXTest)

		confusion_matrix = metrics.confusion_matrix(genderYTest, y_pred.round())
		cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

		cm_display.plot()
		plt.show()
        

