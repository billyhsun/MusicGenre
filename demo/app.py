from flask import Flask, render_template, request, redirect

import os
import demo
import shutil

input_dir = './static/demo_songs'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = input_dir


# Route for home or index
@app.route('/')
def home():
    shutil.rmtree('./static/demo_songs')
    os.mkdir('./static/demo_songs')
    return render_template('home.html')


@app.route('/uploader', methods=['GET', 'POST'])
def from_youtube():
    if request.method == 'POST':
        mp3_path = request.form['text']
        demo.run_from_youtube(mp3_path)
    return redirect('http://localhost:5000/test')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        demo.run_demo()
    return redirect('http://localhost:5000/test')


@app.route('/test', methods=['GET'])
def test():
    return render_template('test.html')


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
