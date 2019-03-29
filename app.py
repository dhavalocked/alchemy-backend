# IMPORTS
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
import flask
import io
import cv2
import numpy as np
from base64 import b64encode
from os import makedirs
from os.path import join, basename
import os
from sys import argv
import json
from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
from scannable_paper import getResponseFromImage, evaluateOmrQuestion
from werkzeug.utils import secure_filename
import os

# CONFIG
photos = UploadSet('photos', IMAGES)

# CONFIG
app = Flask(__name__, instance_relative_config=True)

#from waitress import serve 


app.config['UPLOADED_PHOTOS_DEST'] = 'static/'
configure_uploads(app, photos)

app.config.from_object(os.environ['APP_SETTINGS'])


from tools import upload_file_to_s3,upload_filename_to_s3
from ocr import ocr_prediction

ALLOWED_EXTENSIONS = app.config["ALLOWED_EXTENSIONS"]


# ROUTES
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # There is no file selected to upload
        if "user_file" not in request.files:
            return "No user_file key in request.files"

        file = request.files["user_file"]

        # There is no file selected to upload
        if file.filename == "":
            return "Please select a file"

        # File is selected, upload to S3 and show S3 URL
        if file and allowed_file(file.filename):
            file.filename = secure_filename(file.filename)
            output = upload_file_to_s3(file, app.config["S3_BUCKET"])
            return str(output)
    else:
        return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    responses = []
    q_types = ["ocr","ocr", "ocr", "omr","omr"]
    idx_char_omr = { 1 : "A", 2 : "B", 3 : "C", 4: "D"}
    print(request.files)
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST" and 'photo' in request.files:
            filename = photos.save(request.files['photo'])
            success , answers, question_array_names= getResponseFromImage(filename)
            if success == True:
                for i in range(len(answers)):
                    q_img = "answers"+str(i+1)+".png"
                    if q_types[i] == "omr":
                        img = cv2.imread(os.path.join('./answers',q_img)) 
                        detected_omr_ans = evaluateOmrQuestion(img);
                        responses.append(idx_char_omr[detected_omr_ans[0] ] )

                    if q_types[i] =="ocr":
                        img = cv2.imread(os.path.join('./answers',q_img))
                        responses.append(ocr_prediction(img))
                questions = []
                for i in range(len(question_array_names)):
                    q_img = question_array_names[i]
                    file = os.path.join('questions',q_img) 
                    output = upload_filename_to_s3(file,q_img, app.config["S3_BUCKET"])
                    questions.append(str(output))

                data["predictions"] = responses
                data["questions"] = questions


                # indicate that the request was a success
                data["success"] = True
                #print(question_array_names)
            else :
                data["message"] = "Not able to detect regions properly"
                print(data)
    # return the data dictionary as a JSON response
    return flask.jsonify(data)