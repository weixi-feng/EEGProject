import numpy as np

def get_train(path):
	X_train_valid = np.load(path + 'X_train_valid.npy')
	y_train_valid = np.load(path + 'y_train_valid.npy')
	person_train_valid = np.load(path + 'person_train_valid.npy')
	return X_train_valid, y_train_valid, person_train_valid

def get_test(path):
	X_test = np.load(path + 'X_test.npy')
	y_test = np.load(path + 'y_test.npy')
	person_test = np.load(path + 'person_test.npy')
	return X_test, y_test, person_test


def label_encode(y):
    category = np.unique(y)
    cls = 0
    for cat in sorted(category):
        y[y==cat] = int(cls)
        cls += 1
    return y


def shuffle_data(inputs):
    shapes = []
    for i in range(len(inputs)):
        shapes.append(inputs[i].shape)
        inputs[i] = np.reshape(inputs[i], (inputs[i].shape[0], -1))
    concat = np.concatenate(inputs, axis=1)
    np.random.shuffle(concat)
    start = 0
    end = 0
    for i in range(len(inputs)):
        if i == 0:
            end = start + np.prod(shapes[i][1:])
            inputs[i] = np.reshape(concat[:, start:end], shapes[i])
        else:
            start += np.prod(shapes[i-1][1:])
            end += np.prod(shapes[i][1:])
            inputs[i] = np.reshape(concat[:, start:end], shapes[i])

    return inputs


def get_data(path):
    X_train, y_train, person_train = get_train(path)
    X_test, y_test, person_test = get_test(path)

    # First 22 are EGG
    X_train, X_test = X_train[:, np.newaxis, :22, :], X_test[:, np.newaxis, :22, :]
    # y_train: 769, 770, 771, 772
    y_train = np.reshape(label_encode(y_train), (-1, 1))
    y_test = np.reshape(label_encode(y_test), (-1, 1))

    return ([X_train, y_train, person_train],[X_test, y_test, person_test])
