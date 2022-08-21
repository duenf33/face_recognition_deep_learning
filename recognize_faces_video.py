from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str, help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1, help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn", help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())
# Load known faces and embeddings.
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
# Initialize video stream and pointer to putput video file. 
# Allow camera sensor to warm up.
print("[INFO] starting video stream...")
vs = VideoStream(src=2).start()
writer = None
time.sleep(2.0)
# Loop over frames from video file stream.
while True:
    # Grab frame from threaded video stream.
    frame = vs.read()
    # Convert input frame from BGR to RGB.
    # Resize it to 750px width to speedup processing.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])
    # Detect (x, y)-coordinates of bounding boxes corresponding to each face in the input frame
    # Compute facial embeddings for each face.
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    # Loop over facial embeddings.
    for encoding in encodings:
        # Attempt to match each face in input image to known encodings.
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
    # Check to see if match found.
    if True in matches:
        # Find the indices of all matched faces then initialize dictionary to count total number of times each face was matched.
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        # Loop over matched indices and maintain a count for for each recognized face face.
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1
        # Determine recognized face with largest number of votes if tie python will select first entry in Dict.
        name = max(counts, key=counts.get)
    # Update list of names.
    names.append(name)
    # Loop over recognized faces.
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # Rescale face coordinates.
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)
        # Draw predicted face name on the image.
        cv2.rectangle(frame, (left, top), (right, bottom),
            (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2)
    # If video writer None and if need to write output video to disk initialize the writer.
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 20, (frame.shape[1], frame.shape[0]), True)
    # If the writer is not None, write frame with recognized faces to disk.
    if writer is not None:
        writer.write(frame)
    # Check if need to display the output frame to screen.
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # If `q` key pressed, break from loop.
        if key == ord("q"):
            break
cv2.destroyAllWindows()
vs.stop()
if writer is not None:
    writer.release()