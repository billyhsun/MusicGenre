from flask import Flask, render_template, request, redirect, url_for
import os
import app.utils as ut
import shutil
from app import app


# Route for home or index
@app.route('/')
def home():
    # if os.path.exists('./static/demo_songs'):
    #     shutil.rmtree('./static/demo_songs')
    # if ~os.path.exists('./static/demo_songs'):
    #     os.mkdir('./static/demo_songs')
    return render_template('home.html')


@app.route('/uploader', methods=['GET', 'POST'])
def from_youtube():
    conf = {}
    if request.method == 'POST':
        mp3_path = request.form['text']
        conf = ut.run_from_youtube(mp3_path)
    return render_template('test.html', percentages=conf)


@app.route('/uploader1', methods=['GET', 'POST'])
def upload_file():
    conf = {}
    if request.method == 'POST':
        f = request.files['file']
        print(os.getcwd())
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        conf = ut.run_demo()
    return render_template('test.html', percentages=conf)
