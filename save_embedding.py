import face_recognition
import pickle
import os
import glob
from PIL import Image

all_face_encodings = {}



images = glob.glob("known_people_folder/*.jpg")
for image in images:
	print("filename ",image)
	img1 = face_recognition.load_image_file(image)
	all_face_encodings[image] = face_recognition.face_encodings(img1)[0]
	# print("all_face_encodings",all_face_encodings)

with open('dataset_facesALL.dat', 'wb') as f:
	pickle.dump(all_face_encodings, f)