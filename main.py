from read_data import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
from model.model import *


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


def get_data():
    X_train, y_train, person_train = get_train()
    X_test, y_test, person_test = get_test()

    # First 22 are EGG
    X_train, X_test = X_train[:, np.newaxis, :22, :], X_test[:, np.newaxis, :22, :]
    # y_train: 769, 770, 771, 772
    y_train = np.reshape(label_encode(y_train), (-1, 1))
    y_test = np.reshape(label_encode(y_test), (-1, 1))

    return ([X_train, y_train, person_train],[X_test, y_test, person_test])


# get data and shuffle
train_data, test_data = get_data()
X_train, y_train, person_train = shuffle_data(train_data)
X_test, y_test, person_test = shuffle_data(test_data)


# define deivce
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using %s" % device)

# define batch size and training batch
batch_size = 64
batch_num = math.ceil(X_train.shape[0]/batch_size)

# define network
basenet = SimpleNet()
basenet = basenet.to(device)

# define loss function and optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(basenet.parameters(), lr=0.001)

def train(epoch):
    basenet.train()
    train_loss = 0.0
    for i in range(batch_num):
        try:
            inputs = X_train[i*batch_size:(i+1)*batch_size]
            targets = y_train[i*batch_size:(i+1)*batch_size]
        except:
            inputs = X_train[i*batch_size:]
            targets = y_train[i*batch_size:]

        inputs = torch.tensor(inputs, dtype=torch.float,requires_grad=True)
        targets = torch.tensor(np.squeeze(targets, 1), dtype=torch.long, requires_grad=False)
        inputs, targets = inputs.to(device), targets.to(device)
        # clear gradient
        optimizer.zero_grad()
        outputs = basenet(inputs)
        loss = criterion(outputs, targets)

        # gradient descent
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print('Epoch %d, training loss: %f' % (epoch+1, train_loss/batch_num))

def test():
    test_batch_num = math.ceil(X_test.shape[0]/batch_size)
    basenet.eval()
    corrected = 0
    with torch.no_grad():
        for i in range(test_batch_num):
            try:
                inputs = X_test[i*batch_size:(i+1)*batch_size]
                targets = y_test[i*batch_size:(i+1)*batch_size]
            except:
                inputs = X_test[i*batch_size:]
                targets = y_test[i*batch_size:]

            inputs = torch.tensor(inputs, dtype=torch.float, requires_grad=True)
            targets = torch.tensor(np.squeeze(targets, 1), dtype=torch.long, requires_grad=False)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = basenet(inputs)
            _, predict = outputs.max(1)
            corrected += int(torch.sum(predict==targets).item())
    test_acc = 100*corrected/(X_test.shape[0])
    print('Test accuracy: %f' % test_acc)

for epoch in range(20):
    train(epoch)
    test()