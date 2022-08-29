import face_recognition
import argparse
import pickle
import cv2
# import pandas as pd

ap = argparse.ArgumentParser(description="""
Ex:
❯ python recognize_faces_image.py --encodings encodings_jd.pickle --image examples/Jenny/jd09.jpg
""")
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())
"""
Ex:
❯ python recognize_faces_image.py --encodings encodings_jd.pickle --image examples/Jenny/jd09.jpg
"""
# load the known faces
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
# df = pd.DataFrame(data)
# df.to_csv(r'from_pickle_to_csv_file')
# print("data: ", len(data), data)
# load input image and convert from BGR to RGB
image = cv2.imread(args["image"])
# print("image: ", len(image), image)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# print("rgb: ", len(rgb), rgb)
# detect the x, y coordinates of the bounding boxes for each face in the input image. Compute facial embeddings for each face
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
# print("boxes: ", len(boxes),boxes)
encodings = face_recognition.face_encodings(rgb, boxes)
# print("encodings: ", len(encodings), encodings)
# initialize list of names for each face detected
names = []

# loop over the facial embeddings
for encoding in encodings:
    print("encodings: ", len(encodings), encodings)
    # print("encoding: ", len(encoding), encoding)
    # attempt to match each face in the input image to known encodings
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"
    # check to see if match found
    if True in matches:
        # find indices of all matched faces. initialize dictionary to count total number of times each face was matched
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        # loop over matched indices, keep count for each recognized face
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1
        # determine recognized face with the largest number of votes. It tie, will select first entry
        name = max(counts, key=counts.get)
    # update list of names
    names.append(name)


# recognized faces loop
for ((top, right, bottom, left), name) in zip(boxes, names):
    # draw the predicted face name on the image
    cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 0), 3)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0), 3)

# output image
cv2.imshow("Image", image)
cv2.waitKey(0)
