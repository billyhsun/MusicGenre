'''Use spotify API to download 30s previews and song informations'''
import os
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import requests

def get_playlist_tracks(sp, username, playlist_id, playlist_length):
    results = sp.user_playlist_tracks(username, playlist_id, fields="items(track(name,href,album(name,href), preview_url, id))")
    tracks = results['items']

    while len(tracks) < playlist_length:
        limit = min(playlist_length - len(tracks), 100)
        offset = len(tracks)
        results = sp.user_playlist_tracks(username, playlist_id, fields="items(track(name,href,album(name,href), preview_url, id))",
                                          limit=limit, offset=offset)
        tracks.extend(results['items'])
    return tracks

def download_mp3_from_url(url, dir, name):
    '''Downloads mp3 file located at given url into given directory'''
    r = requests.get(url)
    open(os.path.join(dir, name + ".mp3"), 'wb').write(r.content)



def download_all_songs(Test=False) :
    existing_songs_path = "./processed_data/song_metadata/curr_songs_include.json"  #DICTIONARY OF DICTIONARIES

    username = "adragon-a"
    client_id = "2caf6b9b00e347c1b86222c8de646f3d"
    client_secret = "1fed4e90fd55414b890ad87f7f60cfcc"

    # token = util.prompt_for_user_token(username)
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    cache_token = client_credentials_manager.get_access_token()

    if Test:
        playlist_urls = {'rap': "0aZ1erj2W90khnZRPWFpfs",
                         'rock': "3WpckgZxmWpb9Ch25mM8pW",
                         'classical': "0gVKUpD8dGWFUseamhm3i4",
                         'jazz': '7oJkkK1ZrElmH9U3UCuQTD'}
        playlist_lengths = {'rap': 350,
                            'rock': 276,
                            'classical': 143,
                            'jazz': 386}
    else:
        playlist_urls = {'rap': "2nY0lFbuPIMIOcPxz8IoZa",
                         'rock': "6mjHhCsUwFL6b0Fmh9F3lz",
                         'classical': "4266TiRPmZ9zezVWsOmgNI",
                         'jazz': '0rfHlXDoyOxv5ovkIs8aIC',
                         'pop': '0CxzTl3OZ7wRZXgMCsGv9o'}
        playlist_lengths = {'rap': 1701,
                         'rock': 1691,
                         'classical': 1453,
                         'jazz': 1343,
                         'pop': 1431}

    test = 0
    if cache_token:
        sp = spotipy.Spotify(cache_token)
        sp.trace = False
        try:
            metadata_all_songs = json.loads(existing_songs_path)
        except:
            metadata_all_songs = {}
        for plist in playlist_urls:
            playlist_tracks = get_playlist_tracks(sp, username, playlist_urls[plist], playlist_lengths[plist])
            #playlist_tracks = sp.user_playlist_tracks(username, playlist_urls[plist], fields="items(track(name,href,album(name,href), preview_url, id))")  #The track we want to deal with has preview link at external_urls field
            metadata = {}
            for i, track in enumerate(playlist_tracks):
                if track['track']['id'] not in metadata_all_songs.keys() and j<80:
                    if track['track']['preview_url'] is not None:
                        metadata[track['track']['id']] = {}  #update metadata
                        metadata[track['track']['id']]["name"] = track['track']['name']
                        metadata[track['track']['id']]["album"] = track['track']['album']
                        metadata[track['track']['id']]["prev"] = track['track']['preview_url']
                        metadata[track['track']['id']]["genre"] = plist
                        download_mp3_from_url(metadata[track['track']['id']]["prev"], os.path.join("./songs", plist), track['track']['id'])
                    else:
                        test = test + 1
                        print(test)
                        pass
                        # sp.user_playlist_remove_specific_occurrences_of_tracks(username, playlist_urls[plist], list({"uri": track['track']['id'], "positions":[i]}))

        metadata_all_songs.update(metadata)  #Add update metadata of all songs
        json.dump(metadata_all_songs, open("./processed_data/song_metadata/curr_songs_include.json", "w"))

        metadata_all_songs.update(metadata)  #Add update metadata of all songs
        json.dump(metadata_all_songs, open("./processed_data/song_metadata/curr_songs_include.json", "w"))

    else:
        print("Can't get token for", username)
