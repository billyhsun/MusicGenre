import torch.utils.data as data
import numpy as np


class SongDataset(data.Dataset):
    '''Instantiate a Dataset object for data'''
    def __init__(self, songs_path, labels_path):
        """Seperate data into features and labels
        :param songs_path: filepath to np array of song data where every row is one song's audio.
        :param labels_path: filepath to np array of song genres where every row is one song's genre.
        """
        self.features = np.load(songs_path)
        self.labels = np.load(labels_path)

    def __len__(self):  # return number of different songs that are in dataset
        return self.features.shape[0]

    def __getitem__(self, index):  # return audio encoding of ith song in dataset and label
        ######

        # 3.1 YOUR CODE HERE
        return self.features[index, :], self.labels[index]

        ######
