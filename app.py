from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS  # Import CORS from flask_cors
import os
from anonymize import Anonymizer
import BayesianOpt

app = Flask(__name__, template_folder='app/templates')
CORS(app, methods=["GET", "POST"])

UPLOAD_FOLDER = 'data/adult'  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/anon')
def anon():
    filename = request.args.get('filename')
    script_path = 'BayesianOpt.py'
    filename = filename[0:len(filename)-4]
    cmd = 'python ' + script_path + ' ' + filename
    # if 'file' not in request.files:
    #     return 'No file part'
    # file = request.files['file']
    # process = subprocess.Popen(['python', script_path, filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # output, error = process.communicate()
    os.system(cmd)
    # output = BayesianOpt.main(filename)    
    # anonymizer = Anonymizer(['mondrian', 10, filename])
    return render_template('anonymize.html')

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
        return redirect(url_for('anon', filename=file.filename))  # Redirect to 'anon' function

if __name__ == '__main__':
    app.run(debug=True)
