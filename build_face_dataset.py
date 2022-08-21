from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help = "path to where the face cascade resides")
ap.add_argument("-o", "--output", required=True, help="path to output directory")
args = vars(ap.parse_args())
# Load openCV's Haarcascade for face detection from disk
detector = cv2.CascadeClassifier(args["cascade"])
# Initialize video stream.
# Initialize total number of example faces written to disk.
print("[INFO] starting video stream...")
vs = VideoStream(src=2).start()
# Let camera sensor warm up.
time.sleep(2.0)
total = 0
# Loop over frames from video stream
while True:
    # Grab frame from threaded video stream.
    frame = vs.read()
    # Clone it if needed to write to disk.
    orig = frame.copy()
    # Resize frame to apply face detection faster.
    frame = imutils.resize(frame, width=400)
    # Detect faces in the grayscale frame
    rects = detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Loop over face detections and draw them on the frame.
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Show output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # If `k` pressed write original frame to disk to use for face recognition.
    if key == ord("k"):
        p = os.path.sep.join([args["output"], "{}.png".format(str(total).zfill(5))])
        cv2.imwrite(p, orig)
        total += 1
    # If `q` pressed break from loop.
    elif key == ord("q"):
        break
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()   
