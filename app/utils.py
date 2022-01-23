from __future__ import unicode_literals

import os
import torch
import librosa
import json
import numpy as np

import youtube_dl
import sounddevice as sp
from apscheduler.schedulers.background import BackgroundScheduler
sched = BackgroundScheduler(daemon=True)

logistics_path = './static/logistics'
song_path = './static/demo_songs'


def download_youtube_song(url):
    ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


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


def get_cut_sample(song_path):
    '''Take in path to song return numpy array of sample'''
    y, sr = librosa.load(song_path)

    # Take 2 samples of length 200,000 (roughly 10 seconds) from each song, spaced evenly apart
    start = int(1 * y.shape[0] / 2)
    end = start + 200000
    return y[start:end]


def mmfc_transform(song_array):
    song_array = np.expand_dims(song_array, 0)
    val = fourier_transform(song_array, True)
    val = (val.reshape((1, 100, 52)))
    val = np.swapaxes(val, 0, 1)
    return val


def run_demo():
    '''Launch command line demo that prompts user to input song, plays it back, and then predicts'''

    # while True:
    # print(os.getcwd())
    for filename in os.listdir(song_path):
        if filename == '.DS_Store':
            os.remove(os.path.join(song_path, filename))
        else:
            os.rename(os.path.join(song_path, filename), os.path.join(song_path, "curr_song.mp3"))

    mp3_path = os.path.join(song_path, "curr_song.mp3")
    # print(mp3_path)
    song_array = get_cut_sample(mp3_path)
    sp.play(song_array, 22050)
    transformed_song_array = mmfc_transform(song_array)

    # input('This is the song sample we used for prediction. Continue? \n')

    model = torch.load(os.path.join(logistics_path, "best_model.pt"), map_location='cpu')
    results = model.forward(torch.tensor(transformed_song_array))

    genres = ["Classical", "Jazz", "Rap", "Rock"]
    r = {"Classical": 0.0, "Jazz": 0.0, "Rap": 0.00, "Rock": 0.00}

    confidence = []
    for i in range(0, len(results[0])):
        r[genres[i]] = str(round(float(results[0][i]) * 100, 2))
        confidence.append(round(float(results[0][i]), 4))

    with open('./static/logistics/results.json', 'w') as fp:
        json.dump(r, fp)

    for i in range(0, len(results[0])):
        print(genres[i], ":", confidence[i], " -- ")
    print('\n')

    for filename in os.listdir(song_path):
        os.remove(os.path.join(song_path, filename))

    return r


def run_from_youtube(mp3_path):
    # print(os.getcwd())
    # print(os.path.exists(song_path))
    if ~os.path.exists(song_path):
        os.chdir(song_path)
    download_youtube_song(mp3_path)
    os.chdir('..')
    os.chdir('..')

    conf = run_demo()
    return conf


def abc():
    return {"status": 200, "Message": "App running"}


# Run pipeline daily at 1 pm (13:00)
# sched.add_job(pipeline, trigger='cron', day='*', hour=13)
sched.add_job(abc, trigger='cron', day='*', hour=13)
sched.start()


if __name__ == '__main__':
    conf = run_demo()
