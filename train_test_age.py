# import modules
import pandas as pd
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

ageXTrain = None
ageYTrain = None
ageXTest = None
ageYTest = None

class NNAge():



	def readData(self):
	
		
		# self.splitTrainTest()
		self.loadageTrainingData()
		self.loadageTestData()
		self.sequentialage()

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



		
	def sequentialage(self):
		print(ageYTrain)
		print(ageYTest)
		if not os.path.exists("C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\models_age\\saved_model.pb"):
			print("model not exıst")
			model = keras.Sequential([keras.layers.InputLayer(input_shape=ageXTrain.shape[1:]),
										keras.layers.Dense(32, activation='relu'),
										keras.layers.Dense(64, activation='relu'),
										keras.layers.Dense(64, activation='relu'),
										keras.layers.Dense(32, activation='relu'),
										keras.layers.Dropout(0.2),                                  #
										keras.layers.Dense(16, activation='relu'),
										# keras.layers.Dropout(0.2),
										keras.layers.Dense(8, activation='relu'),
										# keras.layers.Dense(11, activation='relu'),
										keras.layers.Dense(3, activation='softmax')
										])
			model.save("C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\models_age")
		else:
			print("model exıst")
			model = keras.models.load_model("C:\\Users\\Elif\\Documents\\Python_Projects\\muhtas3\\models_age")


		model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,             #
														patience=5, min_lr=0.001)
		model.summary()

		model.fit(ageXTrain, ageYTrain, epochs=50, validation_split=0.2, batch_size=64)
		#model.evaluate(genderXTest, genderYTest)  sparse_
        
		model.evaluate(ageXTest, ageYTest)

		y_pred = model.predict(ageXTest)
		confusion_matrix = metrics.confusion_matrix(ageYTest, y_pred.round())

        




