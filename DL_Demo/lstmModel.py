# Demo file: get data from Twitter posts and donation amount
# Cheng Shen
# chs091@ucsd.edu
# 2019 Jan. 28th

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as torch_init
from torch.autograd import Variable
import numpy as np
import os
import pandas as pd
import csv
#from nltk.translate import bleu_score

def process_train_data ( pd_frame ):
    ''' Return all the data in the file in a numpy matrix'''
    x = df.values[1:]
    y = df.values[:-1]

    # Reshape for input of lstm
    y = y[:, -1]
    return x.reshape(x.shape[0], 1, x.shape[1]), y.reshape(y.shape[0], 1, 1)

class lstmPredictor(nn.Module):
    def __init__(self, config):
        super(lstmPredictor, self).__init__()

        self.h = None
        self.c = None
        self.lstm = nn.LSTM(config['input_dim'], config['hidden_dim'], config['layers'])

        self.fc = nn.Linear(in_features=config['hidden_dim'], out_features=config['output_dim'])
        torch_init.xavier_normal_(self.fc.weight)

    def forward(self, sequence):
        # Takes in the sequence of the form (seq_length x input_dim)
        # and returns the output of form(seq_length x output_dim)

        if self.h is None:
            output, (hn, cn) = self.lstm(sequence)
        else:
            output, (hn, cn) = self.lstm(sequence, (self.h, self.c))

        self.h = hn
        self.c = cn

        # Connect the output of LSTM to a fully connected Linear layer
        output = nn.functional.relu(output)
        output = self.fc(output)

        return output


# input vector: input date, # of tweets of input date, output date
# output vector: # of tweets of output date
config = {'input_dim': 3, 'hidden_dim': 10, 'output_dim': 1, 'layers': 2, 'learning_rate':0.1, \
    'epochs':50, 'filename':'data.csv', 'cuda':True}

if config['cuda']:
    computing_device = torch.device("cuda")
else:
    computing_device = torch.device("cpu")

# Training process
model = lstmPredictor(config)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
model.to(computing_device)

df = pd.read_csv(config['filename'])
print("Starting training: {} epochs".format(config['epochs']))
for epoch in range(config['epochs']):
    # Converting DataFrame to numpy array
    train_data, train_labels = process_train_data(df)

    # Convert data to pytorch tensor
    X_train = torch.tensor(train_data).type(torch.FloatTensor).to(computing_device)
    y_train = torch.tensor(train_labels).type(torch.FloatTensor).to(computing_device)

    model.zero_grad()
    model.h = None
    model.c = None

    y_actual = model(X_train)
    loss = loss_function(y_actual, y_train)
    loss.backward()
    optimizer.step()

    print("Loss: {} at epoch {}".format(loss, epoch))

# Try with custom input
input_matrix = np.ndarray((10, 1, 3))
for i in range(10):
	input_matrix[i, 0, 0] = i
	input_matrix[i, 0, 1] = 10*i

input_matrix = torch.tensor(input_matrix).type(torch.FloatTensor).to(computing_device)
# Output to file
output = model(input_matrix)
for i in range(output.shape[0]):
	print("Expected donation at day {}: {}".format(i, output[i,0,0]))
