from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

ap = argparse.ArgumentParser(description="""
Ex:
❯ python encode_faces.py --dataset dataset/fernando --encodings enconding_fernando.pickle
""")
ap.add_argument("-i", "--dataset", required=True, help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn", help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())
# Grab paths to input images from dataset.
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
# Initialize list of known encodings and known names.
knownEncodings = []
knownNames = []
# Loop over image paths.
for (i, imagePath) in enumerate(imagePaths):
    # Extract person's name from image path.
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    # Load input image. Convert from BGR openCV ordering to dlib ordering RGB.
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Detect the x, y coordinates of bounding boxes from each face in input image.
    boxes = face_recognition.face_locations(
        rgb, model=args["detection_method"])
    # Compute facial embedding for face.
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        # Add each encoding + name to set of known names and encodings.
        knownEncodings.append(encoding)
        knownNames.append(name)
# Dump facial encodings + names to disk.
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()