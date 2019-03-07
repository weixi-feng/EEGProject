import numpy as np

def get_train():
	X_train_valid = np.load('./data/X_train_valid.npy')
	y_train_valid = np.load('./data/y_train_valid.npy')
	person_train_valid = np.load('./data/person_train_valid.npy')
	return X_train_valid, y_train_valid, person_train_valid

def get_test():
	X_test = np.load('./data/X_test.npy')
	y_test = np.load('./data/y_test.npy')
	person_test = np.load('./data/person_test.npy')
	return X_test, y_test, person_test

get_train()