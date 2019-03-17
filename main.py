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
parser.add_argument('--crop', default='naive', type=str, help='data crop, naive/neighbour')
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
step = 3
fold = 4

# get data
abs_path = os.path.abspath('.')
abs_path = abs_path + '/data/'
train_data, test_data = get_data(abs_path)

# Shuffle data
X_train, y_train, person_train = shuffle_data(train_data)
X_test, y_test, person_test = shuffle_data(test_data)


if args.crop == 'naive':
    X_train, y_train, X_val, y_val, _ = naive_crop(X_train, y_train, train=True)
    X_test_c, y_test_c, N = naive_crop(X_test, y_test)
elif args.crop == 'neighbour':
    X_train, y_train, X_val, y_val, N = neighbour_crop(X_train, y_train, train=True, step=step, start=200, end=710)
    X_test_c, y_test_c, N = neighbour_crop(X_test, y_test, step=step, start=200, end=710)
elif args.crop == 'neighbor':
    raise RuntimeError('Please type in British Spelling!')
elif args.crop == 'fancy':
    X_train, y_train, X_val, y_val, N = fancy_crop(X_train, y_train, train=True, length=2, fold=fold)
    X_test_c, y_test_c, N = fancy_crop(X_test, y_test, length=2, fold=fold)
else:
    raise RuntimeError('Wrong input, naive or neighbour!')

X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
train = data_utils.TensorDataset(X_train.type(torch.FloatTensor), y_train.type(torch.LongTensor))
trainloader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=False)

X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)
val = data_utils.TensorDataset(X_val.type(torch.FloatTensor), y_val.type(torch.LongTensor))
valloader = data_utils.DataLoader(val, batch_size=batch_size, shuffle=False)

X_test_list = np.split(X_test_c, N, axis=0)
y_test_list = np.split(y_test_c, N, axis=0)
testloader_list = []
for i in range(N):
    X, y = torch.from_numpy(X_test_list[i]), torch.from_numpy(y_test_list[i])
    test = data_utils.TensorDataset(X.type(torch.FloatTensor), y.type(torch.LongTensor))
    testloader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=False)
    testloader_list.append(testloader)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

batch_num = math.ceil(X_train.shape[0]/batch_size)



# define network
if args.model == 'simple':
    basenet = SimpleNet()
elif args.model == 'base':
    basenet = BaseNet()
elif args.model == 'inception':
    basenet = Inception()
elif args.model == 'simple_v2':
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


def val():
    num_samples = X_val.shape[0]
    basenet.eval()
    corrected = 0
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = basenet(inputs)
            _, predict = outputs.max(1)
            corrected += int(torch.sum(predict==targets.view(-1)).item())
    test_acc = 100*corrected/(num_samples)
    print('Validation accuracy: %f' % test_acc)
    return test_acc


def test(y_test):
    num_samples = X_test.shape[0]
    scores = torch.zeros((num_samples, 4))
    scores = scores.to(device)
    for i in range(N):
        score = 0
        loader = testloader_list[i]
        basenet.eval()
        corrected = 0
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = basenet(inputs)
                score = outputs if idx==0 else torch.cat((score, outputs), dim=0)
        scores += score
    scores /= N 
    _, predict = scores.max(1)
    y_test = y_test.to(device)
    corrected = int(torch.sum(predict==y_test.view(-1)).item())
    test_acc = 100*corrected/(num_samples)
    print('Test accuracy: %f' % test_acc)
    return test_acc

# Training
test_accuracy = 0
val_accuracy = 0
print("Start training!")
for epo in range(args.epoch):
    train(epo)
    val_accuracy = val()
    if val_accuracy >= 95 or epo>=70:
        break

test_accuracy = test(y_test)

with open('./model/model_acc.p', 'rb') as file:
    model_acc = pkl.load(file)

try:
    max_acc = model_acc[args.model]
except:
    max_acc = 0

if test_accuracy >= max_acc:
    model_acc[args.model] = test_accuracy
    torch.save({
        'epoch': epoch+epoch_old,
        'model_state_dict': basenet.state_dict(),
        'loss': loss_history,
    }, './model/%s_%s_%d_%f_%.5f.t7'%(args.model, args.optim, epoch+epoch_old, args.lr, test_accuracy))

with open('./model/model_acc.p', 'wb') as file:
    pkl.dump(model_acc, file, protocol=pkl.HIGHEST_PROTOCOL)

