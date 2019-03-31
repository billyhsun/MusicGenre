from flask import Flask, render_template, request, redirect, url_for

import os
import demo
import shutil

input_dir = './static/demo_songs'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = input_dir


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
        conf = demo.run_from_youtube(mp3_path)
    return render_template('test.html', percentages=conf)


@app.route('/uploader1', methods=['GET', 'POST'])
def upload_file():
    conf = {}
    if request.method == 'POST':
        f = request.files['file']
        print(os.getcwd())
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        conf = demo.run_demo()
    return render_template('test.html', percentages=conf)


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
