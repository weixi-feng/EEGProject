import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNet(nn.Module):

    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1, 25), padding=(0, 12))
        self.conv2 = nn.Conv2d(25, 25, kernel_size=(22, 1))
        self.elu = nn.ELU(alpha=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 10)),
            nn.ELU(alpha=1),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 10)),
            nn.ELU(alpha=1),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 10), padding=(0, 1)),
            nn.ELU(alpha=1),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )
        self.fc = nn.Sequential(
            nn.Linear(400, 4),
            nn.Softmax(1)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = torch.unsqueeze(x, 1)
        x = self.conv2(x) # x.size() = (N, 25, 1, 1, 1000)
        x = torch.squeeze(x, 2)
        x = self.maxpool1(self.elu(x)) # x.size() = (N, 25, 1, 334)
        x = torch.unsqueeze(torch.squeeze(x, 2), 1) # (N, 1, 25, 334)
        out = self.block2(x)
        out = torch.unsqueeze(torch.squeeze(out, 2), 1)
        out = self.block3(out)
        out = torch.unsqueeze(torch.squeeze(out, 2), 1)
        out = self.block4(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, kernel_size=(1, 25), stride=1, padding=0)
        self.conv2 = nn.Conv2d(40, 1, kernel_size=(22, 1))
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.linear = nn.Linear(2440, 4)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out = torch.pow(out, 2)
        out = torch.log(self.pool(out))
        out = out.view(out.size(0), -1)
        out = F.softmax(self.linear(out))
        return out


class SimpleNet_v2(nn.Module):
    def __init__(self):
        super(SimpleNet_v2, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, kernel_size=(1, 25), stride=1, padding=0) # Nx40x22x476
        self.conv2 = nn.Conv2d(40, 40, kernel_size=(22, 1)) # Nx40x1x476
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)) # Nx40x1x27
        self.linear = nn.Linear(1080, 4)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out = torch.pow(out, 2)
        out = torch.log(self.pool(out))
        out = out.view(out.size(0), -1)
        out = F.softmax(self.linear(out), dim=1)
        return out

class SimpleNet_v3(nn.Module):
    def __init__(self, ksize):
        super(SimpleNet_v3, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, kernel_size=(1, ksize), stride=1, padding=(0, (ksize-1)/2)) # Nx40x22x500
        self.conv2 = nn.Conv2d(40, 40, kernel_size=(22, 1)) # Nx40x1x500
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)) # Nx40x1x29

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out = torch.pow(out, 2)
        out = torch.log(self.pool(out))
        return out


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.block1 = SimpleNet_v3(ksize=25)
        self.block2 = SimpleNet_v3(ksize=35)
        self.block3 = SimpleNet_v3(ksize=15)
        self.conv = nn.Conv2d(120, 60, kernel_size=1, stride=1)
        self.linear = nn.Linear(1740, 4)

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(x)
        b3 = self.block3(x)
        out = torch.cat((b1, b2, b3), 1)
        out = self.conv(out)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.linear(out), 1)
        return out
        