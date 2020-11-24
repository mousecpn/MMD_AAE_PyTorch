import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class Encoder(nn.Module):
    def __init__(self,input_shape,hidden_layer = 2000):
        super(Encoder,self).__init__()
        self.fc = nn.Linear(input_shape,hidden_layer)
        return

    def forward(self,x):
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_shape, hidden_layer=2000):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(hidden_layer,input_shape)
        return

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Taskout(nn.Module):
    def __init__(self, n_class, hidden_layer=2000):
        super(Taskout, self).__init__()
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(hidden_layer,hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, n_class)
        return

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x,dim=1)
        return x

class Adversarial(nn.Module):
    def __init__(self, hidden_layer=2000):
        super(Adversarial, self).__init__()
        self.fc1 = nn.Linear(hidden_layer, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 1)
        self.sigmoid = nn.Sigmoid()
        return

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class MMD_AAE(nn.Module):
    def __init__(self, input_shape, nClass):
        super(MMD_AAE,self).__init__()
        self.input_shape = input_shape
        self.E = Encoder(input_shape = input_shape, hidden_layer=2000)
        self.D = Decoder(input_shape = input_shape, hidden_layer=2000)
        self.T = Taskout(n_class = nClass, hidden_layer=2000)
        return

    def forward(self,x):
        e = self.E(x)
        d = self.D(e)
        t = self.T(e)

        return e,d,t
