
import os
from flask import Flask, flash, request, redirect, url_for , render_template
from werkzeug.utils import secure_filename
from flask import jsonify
# from flask_mysqldb import MySQL
from load_embeddings import scan_known_people
import PIL.Image
import dlib
import difflib
import sys
from jinja2 import Template
from flask import send_from_directory
import json
import click
import os
import math
import difflib

import csv
import numpy as np
from scipy.spatial import distance
import face_recognition.api as face_recognition

app = Flask(__name__)



UPLOAD_FOLDER = 'saved_pictures'
ORIGINAL = 'original'
input_file = 'database.csv'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS\


@app.route('/')
def upload_file():
    return render_template('index.html')



@app.route("/top_matches", methods=['GET', 'POST'])
def request_matches():

    if request.method == 'POST':
        # if request.form['submit_button'] == 'upload':
             # check if the post request has the file part
        if 'file' not in request.files:
            print("something")
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

# !!!!!!!!!!!! SAVING FILE TILL HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        unknown_encodings = query_image_embeddings(filename)
        known_names = known_names_and_encodings.keys()
        known_face_encodings = [known_names_and_encodings[x] for x in known_names]

        ##calculate distances from faces in database
        distances = face_distance_3(known_face_encodings, unknown_encodings)
        look_alike_distances, real_names_distance_dict, same_person_distances = distances_same_person_and_look_alike(
            distances, known_names)

        look_alike_names = get_look_alike_names(look_alike_distances, real_names_distance_dict)

        companies_look_alike, persons_look_alike = get_look_alike_person__and_company_name(look_alike_names)

        # print(companies_look_alike)

        same_person_name = same_person_names(real_names_distance_dict, same_person_distances)

        companies_same, persons_same = get_same_person_name_and_company(same_person_name)

        all_files_name = os.listdir('static/look_alikes')

        names_look_alike, names_same = get_names_with_extension(all_files_name, look_alike_names, same_person_name)


        hists1 = ['look_alikes/' + file for file in names_same]

        hists2 = ['look_alikes/' + file for file in names_look_alike]

    return {'top_matches':{'hists':hists1  , 'confidence':same_person_distances , 'name':persons_same , 'company':companies_same} , 'look_alike':{'hists':hists2  , 'confidence':look_alike_distances , 'name':persons_look_alike , 'company':companies_look_alike},'status':True}

def get_names_with_extension(all_files_name, look_alike_names, same_person_name):
    names_same = []
    names_look_alike = []
    for matched_name in same_person_name:

        closest_match_name = difflib.get_close_matches(matched_name.lower(), all_files_name, 1)
        if len(closest_match_name) == 0:
            print("##")  # return "no such name in database"
        else:
            string_name = ''.join(closest_match_name)
            print("matched_string ", string_name)
            names_same.append(string_name)
            print("length :", len(names_same))
    print("look_alike_names ", len(look_alike_names))
    for matched_name in look_alike_names:
        print("entered into look alike")
        closest_match_name2 = difflib.get_close_matches(matched_name.lower(), all_files_name, 1)
        print("closest match name ", closest_match_name2)
        if len(closest_match_name2) == 0:
            print("closest match name zero")
            # return "no such name in database"
        else:
            string_name = ''.join(closest_match_name2)
            print("string_look_alike :", string_name)
            names_look_alike.append(string_name)
    return names_look_alike, names_same


def get_same_person_name_and_company(same_person_name):
    persons_same = []
    companies_same = []
    for name in same_person_name:
        person, company = get_company_name(name)
        persons_same.append(person)
        companies_same.append(company)
    return companies_same, persons_same


def same_person_names(real_names_distance_dict, same_person_distances):
    same_person_name = []
    for i in real_names_distance_dict:  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if i[1] in same_person_distances:
            same_person_name.append(i[0])
    print(same_person_name)
    return same_person_name


def get_look_alike_person__and_company_name(look_alike_names):
    persons_look_alike = []
    companies_look_alike = []
    for name in look_alike_names:
        person, company = get_company_name(name)
        persons_look_alike.append(person)
        companies_look_alike.append(company)
    return companies_look_alike, persons_look_alike


def get_look_alike_names(look_alike_distances, real_names_distance_dict):
    look_alike_names = []
    for i in real_names_distance_dict:  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if i[1] in look_alike_distances:
            print("look_alike", i[0])
            look_alike_names.append(i[0])
    # print(look_alike_names)
    return look_alike_names


def distances_same_person_and_look_alike(distances, known_names):
    distances = [round(x) for x in distances]
    # make dictionary out of names and distances
    real_names_distance_dict = dict(zip(known_names, distances))
    # print(len(real_names_distance_dict))
    # sort the dictionary
    real_names_distance_dict = (sorted(real_names_distance_dict.items(), key=lambda x: x[1], reverse=True))
    # top5 = real_names_distance_dict[:5]
    # names = [v[0] for v in top5]
    distances_sorted = [v[1] for v in real_names_distance_dict]
    same_person_distances = [dist for dist in distances_sorted if dist in range(90, 101)]
    look_alike_distances = [dist for dist in distances_sorted if dist in range(85, 90)]
    return look_alike_distances, real_names_distance_dict, same_person_distances


def query_image_embeddings(filename):
    image_to_check = os.path.join('saved_pictures/', filename)
    unknown_image = face_recognition.load_image_file(image_to_check)
    # Scale down image if it's giant so things run a little faster
    if max(unknown_image.shape) > 1600:
        pil_img = PIL.Image.fromarray(unknown_image)
        pil_img.thumbnail((1600, 1600), PIL.Image.LANCZOS)
        unknown_image = np.array(pil_img)
    # unknown face encodings
    basename = os.path.splitext(os.path.basename(image_to_check))[0]
    unknown_encodings = face_recognition.face_encodings(unknown_image)
    if len(unknown_encodings) > 1:
        click.echo("WARNING: More than one face found in {}. Only considering the first face.".format(basename))
    if len(unknown_encodings) == 0:
        click.echo("WARNING: No faces found in {}. Ignoring file.".format(basename))
        sys.exit(" Exiting since no faces found")
    return unknown_encodings


@app.route('/uploads/<filename>')
def send_file(filename):
    print("@@@@@@@ ",send_from_directory(ORIGINAL, filename))
    return send_from_directory(ORIGINAL, filename)




####################################### function to upload ###############################################

def face_distance_3(face_encodings, face_to_compare):
    distances = []
    result = []
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    # return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]
    if len(face_encodings) == 0:

        return np.empty((0))
    else:
        for face_encoding in face_encodings:
            result.append(np.linalg.norm(face_encoding - face_to_compare))

    for x in result:
        # print(x)
        distances.append((1 - math.pow(x, 2) / 2) * 100)
    distances = np.asarray(distances)
    return distances



def get_company_name(filename):

    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        print(reader)
        for i, line in enumerate(reader):
            if i > 0:
                name = line[1]
                image_path = line[3]
                image_name = image_path.split(".")[0]
                company_name = line[5]

                if (image_name == filename):
                    return name,company_name
                else:
                    continue
        return "no matching name " , "path "


########################################################################################################################
if __name__ == '__main__':
    ## get all encodings from database
    known_people_folder = 'static/look_alikes'
    known_names_and_encodings = scan_known_people("static/look_alikes")
    app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(debug=True)