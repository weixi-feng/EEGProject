from utils.read_data import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import math
from model.model import *
import argparse
import pickle as pkl
import os

torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description='EGG classification')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate during training')
parser.add_argument('--epoch',  default=10, type=int, help='training epoch')
parser.add_argument('--optim', default='adam', type=str, help='select optimizer, either sgd or adam')
parser.add_argument('--model', default='base', type=str, help='Select model')
parser.add_argument('--load', action='store_true', help='Load the best stored model')
args = parser.parse_args()



# Hyperparameters
loss_history = []
epoch = args.epoch
epoch_old = 0
lr = args.lr
batch_size = 64
# define deivce
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using %s" % device)


# get data
abs_path = os.path.abspath('.')
abs_path = abs_path + '/data/'
train_data, test_data = get_data(abs_path)
# shuffle
X_train, y_train, person_train = shuffle_data(train_data)
X_test, y_test, person_test = shuffle_data(test_data)

X_train, y_train, X_val, y_val = crop_data(X_train, y_train, train=True)
X_test, y_test = crop_data(X_test, y_test)
X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)
val = data_utils.TensorDataset(X_val.type(torch.FloatTensor), y_val.type(torch.LongTensor))
valloader = data_utils.DataLoader(val, batch_size=batch_size, shuffle=False)

# Create dataloader
X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)

train = data_utils.TensorDataset(X_train.type(torch.FloatTensor), y_train.type(torch.LongTensor))
trainloader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
test = data_utils.TensorDataset(X_test.type(torch.FloatTensor), y_test.type(torch.LongTensor))
testloader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=False)

batch_num = math.ceil(X_train.shape[0]/batch_size)



# define network
if args.model == 'simple':
    basenet = SimpleNet()
elif args.model == 'base':
    basenet = BaseNet()
elif args.model == 'kernel':
    basenet = KernelNet()
elif args.model == 'simplev2':
    basenet = SimpleNet_v2()
else:
    raise RuntimeError('Wrong model!')
basenet = basenet.to(device)

# define loss function and optim
criterion = nn.CrossEntropyLoss()
if args.optim == 'sgd':
    optimizer = optim.SGD(basenet.parameters(), lr=lr)
elif args.optim == 'adam':
    optimizer = optim.Adam(basenet.parameters(), lr=lr)
else:
    raise RuntimeError('Optimizer should be sgd or adam')

if args.load == True:
    print('loading model!')
    checkpoint = torch.load('./model/'+args.model+'*.t7')
    basenet.load_state_dict(checkpoint['model_state_dict'])
    epoch_old = checkpoint['epoch']
    loss = checkpoint['loss']


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
## basenet.apply(weights_init)


def train(epoch):
    basenet.train()
    train_loss = 0.0
    for idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # clear gradient
        optimizer.zero_grad()
        outputs = basenet(inputs)
        loss = criterion(outputs, targets.view(-1))

        # gradient descent
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print('Epoch %d, training loss: %f' % (epoch+1, train_loss/batch_num))
    loss_history.append(train_loss/batch_num)

def test(val=False):
    if val:
        loader = valloader
        num_samples = X_val.shape[0]
    else:
        loader = testloader
        num_samples = X_test.shape[0]
    test_batch_num = math.ceil(X_test.shape[0]/batch_size)
    basenet.eval()
    corrected = 0
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = basenet(inputs)
            _, predict = outputs.max(1)
            corrected += int(torch.sum(predict==targets.view(-1)).item())
    test_acc = 100*corrected/(num_samples)
    if val:
        print('Validation accuracy: %f' % test_acc)
    else:
        print('Test accuracy: %f' % test_acc)
    return test_acc

# Training
test_accuracy = 0
val_accuracy = 0
for epo in range(args.epoch):
    train(epo)
    val_accuracy = test(val=True)
    if val_accuracy >= 90:
        break

test_accuracy = test()

with open('./model/model_acc.p', 'rb') as file:
    model_acc = pkl.load(file)


if test_accuracy >= model_acc[args.model]:
    model_acc[args.model] = test_accuracy
    torch.save({
        'epoch': epoch+epoch_old,
        'model_state_dict': basenet.state_dict(),
        'loss': loss_history,
    }, './model/%s_%s_%d_%f_%.5f.t7'%(args.model, args.optim, epoch+epoch_old, args.lr, test_accuracy))

with open('./model/model_acc.p', 'wb') as file:
    pkl.dump(model_acc, file, protocol=pkl.HIGHEST_PROTOCOL)

