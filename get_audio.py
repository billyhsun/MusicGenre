'''Use spotify API to download 30s previews and song informations'''
import os
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import requests


def download_mp3_from_url(url, dir, name):
    '''Downloads mp3 file located at given url into given directory'''
    r = requests.get(url)
    open(os.path.join(dir, name + ".mp3"), 'wb').write(r.content)

existing_songs_path = "./processed_data/song_metadata/curr_songs_include.json"  #DICTIONARY OF DICTIONARIES

username = "leafsmaple"
client_id = "76a157b70b5c4e21b1d1c1e8af629db8"
client_secret = "621a656e04e94a6d932ad646fef2f485"

#token = util.prompt_for_user_token(username)
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
cache_token = client_credentials_manager.get_access_token()

playlist_urls = {'rap': "3LFHhtxgV1bpsJV0SLYfPl",
                 'rock': "4n5pu9jboAXAKZFEQvQEgY",
                 'classical': "4ELj5bXytIqM0oPgkBUsAn",
                 'jazz': '2nGsPJmPuRnF9bn0YgqtEK',
                 'pop': '5V3Z7jR8935yf5P752sHNn'}

if cache_token:
    sp = spotipy.Spotify(cache_token)
    sp.trace = False
    try:
        metadata_all_songs = json.loads(existing_songs_path)
    except:
        metadata_all_songs = {}
    for plist in playlist_urls:
        playlist_tracks = sp.user_playlist_tracks(username, playlist_urls[plist], fields="items(track(name,href,album(name,href), preview_url, id))")  #The track we want to deal with has preview link at external_urls field
        metadata = {}
        for i, track in enumerate(playlist_tracks['items']):
            if track['track']['id'] not in metadata_all_songs.keys():
                if track['track']['preview_url'] is not None:
                    metadata[track['track']['id']] = {}  #update metadata
                    metadata[track['track']['id']]["name"] = track['track']['name']
                    metadata[track['track']['id']]["album"] = track['track']['album']
                    metadata[track['track']['id']]["prev"] = track['track']['preview_url']
                    metadata[track['track']['id']]["genre"] = plist
                    download_mp3_from_url(metadata[track['track']['id']]["prev"], os.path.join("./songs", plist), track['track']['id'])
                else:
                    pass
                    #sp.user_playlist_remove_specific_occurrences_of_tracks(username, playlist_urls[plist], list({"uri": track['track']['id'], "positions":[i]}))

    metadata_all_songs.update(metadata)  #Add update metadata of all songs
    json.dump(metadata_all_songs, open("./data/curr_songs_include.json", "w"))

else:
    print("Can't get token for", username)
