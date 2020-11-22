from flask import Flask

input_dir = './static/demo_songs'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = input_dir

from app import routes