''' Preprocessing of mp3 audio of songs '''

from pydub import AudioSegment
import numpy as np
import librosa
import glob
import os
import random
from train_test_split import *
from get_audio import download_all_songs


# Rename mp3 files into 1.mp3, 2.mp3, ..., n.mp3 in each folder
def rename_file(directory, pattern):
    i = 1
    for pathAndFilename in glob.iglob(os.path.join(directory, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        os.rename(pathAndFilename, os.path.join(directory, str(i) + ext))
        i += 1


def find_cummu_avg(all_data):
    '''
    Find average value over each row in all_data

    return all_data sized array: averages that should be subtracted:
    '''
    normalizers = all_data.sum(axis=1)
    normalizers = np.divide(normalizers, all_data.shape[1])
    normalizers = np.expand_dims(normalizers, axis=1)
    # normalizers = np.expand_dims(normalizers, 2)
    # normalizers = np.repeat(normalizers, 100, axis=2)
    # normalizers = np.moveaxis(normalizers, 2, 0)
    return normalizers


def find_cummu_std(all_data, avg):
    '''
    Find std over each row in all_data
    '''
    cummu_var = (np.power(all_data - avg, 2)).sum(axis=1)
    std = np.power(np.divide(cummu_var, all_data.shape[1]), 0.5)
    std = np.expand_dims(std, 1)
    return std


def normalize_data(data_file):
    ''' Normalize each row in data array by subtracting mean and dividing my std.'''
    all_data = np.load(data_file)
    all_data = np.transpose(all_data)
    avg = find_cummu_avg(all_data)
    std = find_cummu_std(all_data, avg)
    norm = np.divide((all_data - avg), np.repeat(std, all_data.shape[1], axis=1))

    file_name = (data_file.split("."))[0]
    print(norm.shape)
    np.save(file_name + "_normalized", norm)  # Saves without time col


# Cut out a sample from a song represented in audio format
def cut_audio(filepathtosong, length_in_seconds):
    sound = AudioSegment.from_mp3(filepathtosong)

    # len() and slicing are in milliseconds
    start = len(sound) / 3
    end = start + length_in_seconds*100
    sample = sound[start:end]

    # Concatenation is just adding
    # second_half_3_times = second_half + second_half + second_half

    sample.export("cut.mp3", format="mp3")


# Cut out a sample from a song represented in numpy array format
def cut_audio_array(filepathtosong):
    array = np.load(filepathtosong)
    newfile = filepathtosong.strip(".npy") + "_stripped.npy"
    length = array.shape[1]
    start = int(length/3)
    end = start + 200000    # Number may be adjusted
    newarray = array[:, start:end]
    np.save(newfile, newarray)


# Represent a song audio track into a numpy array, and also cuts it into 3 samples
def decode_audio_toarray(songdir_path, saveasfilename):
    i = 1
    samples_array = []
    files = os.listdir(songdir_path) 
    no_of_files = len(files)
    for song_path in files:
        # song_path = os.path.join(songdir_path, str(i) + '.mp3')
        y, sr = librosa.load(songdir_path+song_path)

        # Take 2 samples of length 200,000 (roughly 10 seconds) from each song, spaced evenly apart
        start = int(1 * y.shape[0] / 2)
        end = start + 200000
        samples_array.append(y[start:end])

        i += 1
        if i % 50 == 0:
            print(i)

    if no_of_files <= 1000:
        for song_path in files:
            y, sr = librosa.load(songdir_path+song_path)
            start = 10
            end = start + 200000
            samples_array.append(y[start:end])
            i += 1
            if i % 50 == 0:
                print(i)
            if i > 1100:
                break

    array = np.array(samples_array)
    print(array.shape)
    np.save(saveasfilename, array)


def pick_samples(dir_to_norm_array, no_of_samples):
    songs = np.load(dir_to_norm_array)
    samples = random.sample(range(songs.shape[1]), no_of_samples)
    new = []
    for i in samples:
        new.append(songs[:, i])
    new = np.array(new)
    np.save("data/norm_data/"+dir_to_norm_array.strip("data/").strip(".npy")+"_samples.npy", new)


def merge_samples(list_of_array_files, list_of_labels):
    feats = []
    label = []
    a = 0
    for i in list_of_array_files:
        arr = np.load(i)
        print(arr.shape)
        arr = np.transpose(arr)
        for j in range(arr.shape[1]):
            feats.append(arr[:, j])
            label.append(list_of_labels[a])
        a += 1
    feats = np.array(feats)
    label = np.array(label)
    print(feats.shape)
    print(label.shape)
    np.save("data/all_songs.npy", feats)
    np.save("data/all_labels.npy", label)


def fourier_transform(filename, no_save=False):
    if no_save:
        arr = filename
    else:
        arr = np.load(filename)
    print(arr.shape)
    sr = 22500  # Data points per second
    new_arr = []
    mfcc_arr = []
    for i in range(arr.shape[0]):
        n = 0
        fft = []
        mfcc_a = []
        for j in range(100):
            a = arr[i, n:n+2000]
            
            # Mel-scaled power (energy-squared) spectrogram 
            S = librosa.feature.melspectrogram(a, sr=sr, n_mels=128)
            
            # Convert to log scale (dB); use peak power as reference
            log_S = librosa.amplitude_to_db(S, ref=np.max)

            # Compute mfcc
            mfcc = librosa.feature.mfcc(S=log_S, sr=sr, n_mfcc=13)
            mfcc_a.append(mfcc)

            # Compute Fourier Transform
            #np.fft.fft(a, n=None, axis=-1, norm=None)
            #fft.append(a)
            n += 2000

        mfcc_arr.append(np.array(mfcc_a))
        new_arr.append(np.array(fft))
        if i % 100 == 0:
            print(i)
    mfcc_arr = np.array(mfcc_arr)
    #new_arr = np.array(new_arr)
    print(mfcc_arr.shape)
    #print(new_arr.shape)
    if no_save:
        return mfcc_arr
    else:
        np.save("mfcc_feats.npy", mfcc_arr)
    #np.save(filename.strip(".npy")+"_fft.npy", new_arr)


if __name__ == "__main__":
    #download_all_songs()  #Download all mp3 files from the server

    #Cut all audio samples into two and then save into array
    #decode_audio_toarray('songs/classical/', 'data/classical_songs.npy')
    #normalize_data("data/classical_songs.npy")
    #print(1)
    #decode_audio_toarray('songs/jazz/', 'data/jazz_songs.npy')
    #normalize_data("data/jazz_songs.npy")
    #print(1)
    #decode_audio_toarray('songs/pop/', 'data/pop_songs.npy')
    #normalize_data("data/pop_songs.npy")
    #print(1)
    #decode_audio_toarray('songs/rap/', 'data/rap_songs.npy')
    #normalize_data("data/rap_songs.npy")
    #print(1)
    #decode_audio_toarray('songs/rock/', 'data/rock_songs.npy')
    #normalize_data("data/rock_songs.npy")

    # Make sure there are same number of samples for each of rock, pop, and rap songs - put into data folder
    # pick_samples("data/rock_songs_normalized.npy", 30)
    # pick_samples("data/rap_songs_normalized.npy", 30)
    # pick_samples("data/pop_songs_normalized.npy", 30)
    # pick_samples("data/jazz_songs_normalized.npy", 30)
    # pick_samples("data/classical_songs_normalized.npy", 30)

    #concat_data_and_gen_labels("data/norm_data")
    #fourier_transform("final_data/all_songs.npy")
    split_data("./final_data/mfcc_feats.npy", "./final_data/all_labels.npy")
