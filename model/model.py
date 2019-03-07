import torch
import torch.nn as nn

class BaseNet(nn.Module):

    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1, 25), padding=(0, 12))
        self.conv2 = nn.Conv3d(1, 25, kernel_size=(25, 22, 1))
        self.elu = nn.ELU(alpha=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 1))
        self.block2 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(25, 10)),
            nn.ELU(alpha=1),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 1)),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=(50, 13)),
            nn.ELU(alpha=1),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 1)),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(1, 200, kernel_size=(100, 10)),
            nn.ELU(alpha=1),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(1600, 4),
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
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(1, 7), stride=(1, 5), padding=(0, 3)),
            nn.Conv2d(16, 32, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(8800, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Linear(1000, 4),
            nn.Softmax(),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out