import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import SongDataset
from model import *

import matplotlib.pyplot as plt
import scipy.signal as sig


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
    predictions = np.argmax((predictions.detach().numpy()), axis=1)  # Both prediction and label 1xnum_genre vectors
    label = np.argmax((label.detach().numpy()), axis=1)
    b = (predictions == label)
    corr_num = int(b.sum())
    return corr_num


def evaluate(model, val_loader):
    total_corr = 0
    for i, batch in enumerate(val_loader):
        feats, labels = batch
        predictions = model.forward(feats)
        total_corr += find_num_correct(predictions, labels)
    return float(total_corr)/len(val_loader.dataset)


data_filepath = "./final_data"

# HYPERPARAMETERS
batch_size = 3
learn_rate = 0.8
MaxEpochs = 3
eval_every = 100
num_genres = 3
input_dimensions = (1000, 200)

train_data = SongDataset(os.path.join(data_filepath, "train_data.npy"), os.path.join(data_filepath, "train_labels.npy"))
val_data = SongDataset(os.path.join(data_filepath, "val_data.npy"), os.path.join(data_filepath, "val_labels.npy"))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

model = ConvClassifier2D(batch_size, num_genres, input_dimensions)  # ConvClassifier1D() for raw audio, ConvClassifier2D() for Fourier transformed data
loss_fnc = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

leftover = 0
step_list = []
train_data_list = []
val_data_list = []
best_val_acc = 0

tot_corr = 0
for counter, epoch in enumerate(range(MaxEpochs)):
    for i, batch in enumerate(train_loader):
        feats, labels = batch
        optimizer.zero_grad()
        predictions = model.forward(feats)
        batch_loss = loss_fnc(input=predictions.float(), target=labels.float())
        batch_loss.backward()
        optimizer.step()

        tot_corr += find_num_correct(predictions, labels)
        print(1)

        # Evaluate and log losses and accuracies for plotting
        if ((i + leftover) % eval_every == 0) and ((i + leftover) != 0):  # Leftover makes sure that even if batch size goes over, you graph every eval_evry steps
            val_acc = evaluate(model, val_loader)
            train_acc = float(tot_corr / (eval_every * batch_size))
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
            tot_corr = 0

    leftover = ((len(train_loader) % eval_every) * (counter + 1)) % eval_every
    if val_data_list[-1] > best_val_acc:  # Check model performance and save if best
        best_val_acc = val_data_list[-1]
        torch.save(model, "best_model")
    print("Epoch ", counter, " complete")

plot_accuracy_vs_stepnum(step_list, train_data_list, "Train", 5, 3)  # Make plots of accuacuers
plot_accuracy_vs_stepnum(step_list, val_data_list, "Validation", 11, 5)
