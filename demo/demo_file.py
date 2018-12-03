from __future__ import unicode_literals

import os
import torch
import librosa
import json
import numpy as np

import youtube_dl
import sounddevice as sp
from preprocess import fourier_transform

logistics_path = '/Users/RobertAdragna/Documents/Third\ Year/Fall\ Term/MIE\ 324\ -\ Introduction\ to\ Machine\ Intelligence/mie324/mie324_final_project/logisitcs'

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

    while True:
        mp3_path = input('Enter the youtube URL to your song \n')

        os.chdir('./demo_songs')
        download_youtube_song(mp3_path)
        os.chdir('..')

        for filename in os.listdir('./demo_songs'):
            if filename == '.DS_Store':
                os.remove(os.path.join('./demo_songs', filename))
            else:
                os.rename(os.path.join('./demo_songs', filename), os.path.join('./demo_songs', "curr_song.mp3"))

        mp3_path = os.path.join("demo_songs", "curr_song.mp3")

        song_array = get_cut_sample(mp3_path)
        sp.play(song_array, 22050)
        transformed_song_array = mmfc_transform(song_array)

        input('This is the song sample we used for prediction. Continue? \n')

        model = torch.load(os.path.join(logistics_path, "best_model.pt"), map_location='cpu')
        results = model.forward(torch.tensor(transformed_song_array))

        num_to_label = json.loads(open(os.path.join(logistics_path, "num_to_label.json")).read())
        confidence = []
        for i in range (0, len(results[0])):
            confidence.append(float(results[0][i]))

        for i in range(0, len(results[0])):
            print(str(num_to_label[str(i)]), " :", confidence[i], " -- ")
        print ('\n')

        for filename in os.listdir('./demo_songs'):
            os.remove(os.path.join('./demo_songs', filename))

if __name__ == '__main__':
    run_demo()