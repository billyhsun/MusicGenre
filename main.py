import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import SongDataset
from model import *

import matplotlib.pyplot as plt
import scipy.signal as sig

#For using GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_accuracy_vs_stepnum(step_list, data_list, label, sig_fact_1, sig_fact_2):
    plt.figure()
    plt.title(label + " Accuracy vs Number of Steps")
    smoothed_data_list = sig.savgol_filter(data_list, sig_fact_1, sig_fact_2)
    plt.plot(step_list, smoothed_data_list, label=label)
    plt.xlabel("Number of batches")
    plt.ylabel("Accuracy")
    plt.savefig("{}_accuracy.png".format(label))


def find_num_correct(predictions, label):
    '''Takes in an array of predictions and of labels. Returns the number of these predicitons that are correct'''
    predictions = np.argmax((predictions.cpu().detach().numpy()), axis=1)  # Both prediction and label 1xnum_genre vectors
    label = np.argmax((label.cpu().detach().numpy()), axis=1)
    b = (predictions == label)
    corr_num = int(b.sum())
    return corr_num


def evaluate(model, val_loader):
    total_corr = 0
    for i, batch in enumerate(val_loader):
        feats, labels = batch
        feats = feats.reshape((batch_size, 100, 52)).permute(1, 0, 2)
        feats = feats.to(device)
        labels = labels.to(device)
        #print(feats.shape)
        predictions = model.forward(feats)
        total_corr += find_num_correct(predictions, labels)
        #print(total_corr)
    print(total_corr)
    #print(val_loader.dataset)
    return float(total_corr)/len(val_loader.dataset)


data_filepath = "./final_data"

# HYPERPARAMETERS
batch_size = 20
learn_rate = 0.001  # Decreases with epoch
MaxEpochs = 10
eval_every = 10
num_genres = 4
input_dimensions = (100, 13, 4)  # Reshape to (13, 4, 100)
embedding_dim = 52
rnn_hidden_dim = 100

train_feats = os.path.join(data_filepath, "train_data.npy")
val_feats = os.path.join(data_filepath, "val_data.npy")

train_data = SongDataset(train_feats, os.path.join(data_filepath, "train_labels.npy"))
val_data = SongDataset(val_feats, os.path.join(data_filepath, "val_labels.npy"))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

#model = ConvClassifier2D(batch_size, num_genres, input_dimensions)  # ConvClassifier1D() for raw audio, ConvClassifier2D() for Fourier transformed data

model = RNNClassifier(embedding_dim, rnn_hidden_dim, num_genres)
model = model.to(device)
loss_fnc = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

leftover = 0
step_list = []
train_data_list = []
val_data_list = []
data = []
best_val_acc = 0

tot_corr = 0
for counter, epoch in enumerate(range(MaxEpochs)):
    #optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    #learn_rate = learn_rate / 1.1 # Decrease every epoch
    for i, batch in enumerate(train_loader):
        feats, labels = batch
        feats = (feats.reshape((batch_size, 100, 52))).permute(1, 0, 2)
        #feats = feats.reshape((batch_size, 52, 100))
        #print(feats.shape)
        feats = feats.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        predictions = model.forward(feats)
        batch_loss = loss_fnc(input=predictions.float(), target=labels.float())
        batch_loss.backward()
        optimizer.step()

        tot_corr += find_num_correct(predictions, labels)

        # Evaluate and log losses and accuracies for plotting
        if ((i + leftover) % eval_every == 0) and ((i + leftover) != 0):  # Leftover makes sure that even if batch size goes over, you graph every eval_evry steps
            val_acc = evaluate(model, val_loader)
            #print(tot_corr)
            train_acc = float(tot_corr / (eval_every * batch_size))
            if train_acc > 1:
                train_acc = train_acc/2
            print("Batch", i, ": Total correct in last", eval_every, "batches is", tot_corr,
                  "out of ", eval_every * batch_size)
            print("Total training accurracy over last batches is ", train_acc)
            print("Total validation accurracy over last batches is ", val_acc, "\n")

            # Record relevant values
            if len(step_list) == 0:
                step_list.append(0)
            else:
                step_list.append(step_list[-1] + eval_every)

            train_data_list.append(train_acc)
            val_data_list.append(val_acc)
            
            data.append(np.array([counter, i, train_acc, val_acc]))

            tot_corr = 0

        # SAVE RESULTS
        np.savetxt("results.txt", np.array(data), delimiter = ',')

    leftover = ((len(train_loader) % eval_every) * (counter + 1)) % eval_every
    
    if len(val_data_list) > 0:
        if val_data_list[-1] > best_val_acc:  # Check model performance and save if best
            best_val_acc = val_data_list[-1]
            torch.save(model, "best_model")
    print("Epoch ", counter, " complete")

plot_accuracy_vs_stepnum(step_list, train_data_list, "Train", 5, 3)  # Make plots of accuacuers
plot_accuracy_vs_stepnum(step_list, val_data_list, "Validation", 11, 5)

