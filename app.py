from flask import Flask, request, render_template, redirect, url_for
import os

app = Flask(__name__, template_folder='app/templates')

UPLOAD_FOLDER = 'data/adult'  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/index.html')
def predictautoencoder():
    return render_template('index.html')

@app.route('/output.html')
def about():
    return render_template('output.html')

@app.route('/input.html')
def upload_file():
    return render_template('input.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return redirect(url_for('about'))  # Redirect to 'about' function
    


if __name__ == '__main__':
    app.run(debug=True)
