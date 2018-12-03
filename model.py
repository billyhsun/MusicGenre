import torch.tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import tensor
import torch

num_processed = 1
num_ks_1 = 50
k_size_1 = 500
num_ks_2 = 30
k_size_2 = 100
pool_1 = 100

nn_input_size = 56880  # NOTE - this is hard coded. Will need to be different if song length != 200000
hid_layers_1_num = 100
hid_layers_2_num = 10
num_genres = 3

# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvClassifier1D(nn.Module):
    def __init__(self, batch_size, num_of_genres):
        '''Model Archetecture: 2 1D conv layers with 1 maxpooling in between'''
        super(ConvClassifier1D, self).__init__()
        self.num_processed = 1
        self.num_ks_1 = 50
        self.k_size_1 = 500
        self.num_ks_2 = 30
        self.k_size_2 = 100
        self.pool_1 = 100
        self.hid_layers_1_num = 100
        self.hid_layers_2_num = 10
        self.num_genres = num_of_genres

        self.conv1 = nn.Conv1d(1, 50, 500).float()
        self.pool1 = nn.MaxPool1d(100)
        self.conv2 = nn.Conv1d(50, 30, 100).float()
        self.fc1 = nn.Linear(3360*batch_size, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 4)
        self.batch_size = batch_size

    def forward(self, x):
        # dummy_layer = nn.Linear(nn_input_size, num_genres).float()
        # x = dummy_layer(x.float())
        # assert (x.shape[1] == 200000)  # Just in case we forget to change nn_input_size
        x = (x.unsqueeze(1)).float()
        x = self.conv1(x)
        #print(x.shape)
        x = self.pool1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = x.view(self.batch_size, -1)  # Convert feature maps for each song in batch into 1-d array
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.softmax(x)
        #print(x.shape)
        return x


class ConvClassifier2D(nn.Module):
    def __init__(self, batch_size, num_of_genres, input_dimensions):
        '''Model Archetecture: 2 2D conv layers with 1 maxpooling in between'''
        super(ConvClassifier2D, self).__init__()
        self.num_processed = 1
        self.num_ks = [10, 10, 10, 10]
        self.k_size = [(1, 2000), (20, 1), (5, 1), (5, 1)]
        self.pool = [(1, 1), (1, 1), (1, 1), (1, 1)]
        self.hid_layers = [50, 10, num_of_genres]
        self.batch_size = batch_size
        self.input_dimensions = input_dimensions

        a = input_dimensions[0]  #Dimension size of all FT coeffs
        b = input_dimensions[1]  #Dimension of #time samples
        for i in range(len(self.num_ks)):
            a = int((a - self.k_size[i][0] + 1) / self.pool[i][0])
            b = int((b - self.k_size[i][1] + 1) / self.pool[i][1])

        self.input_size = 200 #a * b * self.num_ks[-1]

        self.conv1 = nn.Conv2d(self.num_processed, self.num_ks[0], self.k_size[0], stride=5).float()
        self.pool1 = nn.MaxPool2d(self.pool[0])
        self.conv2 = nn.Conv2d(self.num_ks[0], self.num_ks[1], self.k_size[1], stride=2).float()
        self.pool2 = nn.MaxPool2d(self.pool[1])
        self.conv3 = nn.Conv2d(self.num_ks[1], self.num_ks[2], self.k_size[2], stride=2).float()
        self.pool3 = nn.MaxPool2d(self.pool[2])
        self.conv4 = nn.Conv2d(self.num_ks[2], self.num_ks[3], self.k_size[3], stride=2).float()
        self.fc1 = nn.Linear(self.input_size, self.hid_layers[0])
        self.fc2 = nn.Linear(self.hid_layers[0], self.hid_layers[1])
        self.fc3 = nn.Linear(self.hid_layers[1], self.hid_layers[2])

    def forward(self, x):
        # dummy_layer = nn.Linear(nn_input_size, num_genres).float()
        # x = dummy_layer(x.float())
        x = (x.unsqueeze(1)).float()
        #print(x.shape)
        x = self.conv1(x)
        #print(torch.mean(x, 0), torch.var(x, 0))
        #x = F.batch_norm(x, torch.mean(x, 0), torch.var(x, 0))
        x = F.dropout(x)
        #print(x.shape)
        #x = self.pool1(x)
        #print(x.shape)
        x = self.conv2(x)
        #x = self.conv3(x)
        #x = self.conv4(x)
        #print(x.shape)
        x = x.view(self.batch_size, -1)  # Convert feature maps for each song in batch into 1-d array
        #print(x.shape)
        #x = self.fc1(x)
        #x = self.fc2(x)
        x = self.fc3(x)
        x = F.softmax(x)
        #print(x.shape)
        return x

#model = ConvClassifier2D().to(device)

class RNNClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_genres):
        super(RNNClassifier, self).__init__()

        self.num_layers = 1
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=self.num_layers)
        self.linear = nn.Linear(hidden_dim, num_genres)

    def forward(self, x):
        x = x.float()
        output, h_n = self.gru(x) # or x.transpose(0, 1)
        x = h_n[0].float()
        #print(x.shape)
        x = self.linear(x)
        #print(x.shape)
        x = F.softmax(x)
        return x

