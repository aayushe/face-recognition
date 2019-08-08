import os
import re
import click
import pickle
import face_recognition.api as face_recognition


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def scan_known_people(known_people_folder):
    model_filename = 'all_embeddings.sav'
    known_names = []
    known_face_encodings = []

    if not os.path.isfile(os.path.join(os.getcwd(), model_filename)):

        for file in image_files_in_folder(known_people_folder):
            basename = os.path.splitext(os.path.basename(file))[0]
            img = face_recognition.load_image_file(file)
            encodings = face_recognition.face_encodings(img)

            if len(encodings) > 1:
                click.echo("WARNING: More than one face found in {}. Only considering the first face.".format(file))

            if len(encodings) == 0:
                click.echo("WARNING: No faces found in {}. Ignoring file.".format(file))
            else:
                known_names.append(basename)
                known_face_encodings.append(encodings[0])

        names_and_encodings = dict(zip(known_names,known_face_encodings))
        pickle.dump(names_and_encodings, open(model_filename, 'wb'))


    names_and_encodings = pickle.load(open(model_filename, 'rb'))

    return (names_and_encodings)

