import os
import numpy as np
import json
import torch
from dataset import SongDataset
from torch.utils.data import DataLoader

logistics_path = './logistics'

def test_find_num_correct(predictions, label):
    '''Takes in an array of predictions and of labels. Returns the number of these predicitons that are correct'''
    predictions = np.argmax((predictions.cpu().detach().numpy()), axis=1)  # Both prediction and label 1xnum_genre vectors
    label = np.argmax((label.cpu().detach().numpy()), axis=1)
    b = (predictions == label)
    corr_num = int(b.sum())
    return corr_num

def test_evaluate(model, batch_size, val_loader):
    total_corr = 0
    for i, batch in enumerate(val_loader):
        feats, labels = batch
        feats = feats.reshape((batch_size, 100, 52)).permute(1, 0, 2)
        #print(feats.shape)
        predictions = model.forward(feats)
        total_corr += test_find_num_correct(predictions, labels)
    return float(total_corr)/len(val_loader.dataset)

def generate_confusion_matrix(model, batch_size, loader, res_2_gen):
    total_corr = 0
    conf_mat = {'classical': {'classical': 0, 'jazz': 0, 'rap': 0, 'rock': 0},
                'jazz': {'classical': 0, 'jazz': 0, 'rap': 0, 'rock': 0},
                'rap': {'classical': 0, 'jazz': 0, 'rap': 0, 'rock': 0},
                'rock': {'classical': 0, 'jazz': 0, 'rap': 0, 'rock': 0}}

    for i, batch in enumerate(loader):
        feats, labels = batch
        feats = feats.reshape((batch_size, 100, 52)).permute(1, 0, 2)
        predictions = model.forward(feats).detach().numpy()
        predictions = np.argmax(predictions, axis=1)
        labels = np.argmax(labels.detach().numpy(), axis=1)

        #Populate confusion dict
        for i in range (0, len(labels)):
            conf_mat[res_2_gen[str(labels[i])]][res_2_gen[str(predictions[i])]] += 1
        return conf_mat

if __name__ == '__main__':
    features_path = './final_data/testset_all_songs_mfcc.npy'
    labels_path = './final_data/testset_all_labels.npy'
    test_data = SongDataset(features_path, labels_path)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    a = os.path.join(logistics_path, "best_model.pt")
    model = torch.load(os.path.join(logistics_path, "best_model.pt"), map_location='cpu')
    label_dict = json.loads(open(os.path.join(logistics_path, "num_to_label.json")).read())
    print(test_evaluate(model, 640, test_loader))
    print(generate_confusion_matrix(model, 640, test_loader, label_dict))
