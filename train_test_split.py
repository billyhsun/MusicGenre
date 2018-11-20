import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


def concat_data_and_gen_labels(dir_path):
    '''Take songs saved in multiple .npy files and merge data into one array. Also produce label for each song in
    same order but seperate array

    :param Input - Abs path to directory where the normalized numpy files are saved
    '''
    for genre_data_file in os.listdir(dir_path):  # Concat all data and create label list
        genre_name = (genre_data_file.split("_"))[0]  # assumes genre name is first phrase in path
        song_data = np.load(os.path.join(dir_path, genre_data_file))
        try:
            all_song_data = np.concatenate((all_song_data, song_data), axis=0)
        except NameError:
            all_song_data = song_data
        try:
            labels = np.concatenate((labels, np.repeat(genre_name, song_data.shape[0], axis=0)), axis=0)
        except NameError:
            labels = np.repeat(genre_name, song_data.shape[0], axis=0)

    # Turn label list into one hot encoding
    label_encoder = LabelEncoder()
    oneh_encoder = OneHotEncoder(sparse=False)
    labels = label_encoder.fit_transform(labels)
    labels = oneh_encoder.fit_transform(labels.reshape(-1, 1))

    np.save("./final_data/all_songs.npy", all_song_data)
    # np.save("./final_data/all_labels.npy", labels)


def split_data(feature_file, label_file):
    '''Split data into test, validation and training sets'''
    seed = 42
    features = np.load(feature_file)
    labels = np.load(label_file)
    train_data, val_data, train_labels, val_labels = \
        train_test_split(features, labels, test_size=0.20, random_state=seed)

    # print(train_labels)
    np.save("./final_data/train_data.npy", train_data)
    np.save("./final_data/train_labels.npy", train_labels)
    np.save("./final_data/val_data.npy", val_data)
    np.save("./final_data/val_labels.npy", val_labels)


if __name__ == "__main__":
    concat_data_and_gen_labels("data/norm_data")
    split_data("./final_data/all_songs.npy", "./final_data/all_labels.npy")
