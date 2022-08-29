DESCRIPTION/STATEMENT OF PURPOSE:

Face-clustering.- is similar to `Face Recognition`, the only difference is that `Face Clustering` has no name labels. Its job is to identify each person's face and also keeping a count of it.

	- What does your program do?

		This program will give the ability to identify a person's face from a dataset of pictures just by giving one sample picture. 

	- Who is the potential user of your program? 

		This program can be useful for law enforcement or just for fun to search and find a person's face in any picture.
	
	- In what form will your program exist? (CLI/Website/script/app)

		CLI

	- What problem does my project/program solve? 

		- it helps to search and identify (for law enforcement) any suspect in the run.
		- it helps sort out a dataset of pictures by names.
		- it helps organize pictures.

Requirements:
	- Which technologies does it use? (python, 3rd party libraries etc.)

        In order to be able to perform the face recognition with python and OpenCV we need to install the following libraries: 
        - face_recognition 
        and
        - dlib

        "face_recognition" library wraps around dlibs facial recognition functionality.

		python, 3rd party libraries such as:
			* os
			* requests
			* argparse
			* pickle
			* cv2
			* face_recognition
			* imutils:
            ( to access your computer camera i used the VideoStream class from imutils. 
            This allows you to connect to any internal an external camera just by identifying it at the src=2.
            I'm ussing src=2 for my computer camera.)

			* ( dlib ) => the last and main library needs 4 primary pre-requisites: 
				. cmake
				. boost
				. boost-python3
				. `XQuartz window manager` => downloading `.dmg`
			       * dlib python bindings for image processing:
				~ numpy
				~ scipy
				~ scimitar-image
			* dlib

	- Does it need any api access etc? 

		Yes, it does. It's:
		URL = "https://api.bing.microsoft.com/v7.0/images/search"


User Stories/Functional Requirements:

User story: 

	User is trying to get their child's pictures from a collection of pictures saved in a folder 	in their computer. The user then is able to sort out all the child's pictures with success in a different folder.

Functional Requirements:

	- The system will accept a input image
	- The system will allow to collect all the pictures and quantify them
	

> search_bing_api.py
❯ python search_bing_api.py --query "will smithwill" --output dataset/will_smith

or 

> build_face_dataset.py
❯ python build_face_dataset.py --cascade haarcascade_frontalface_default.xml --output dataset/Dom

> encode_faces.py
❯ python encode_faces.py --dataset dataset/Dom --encodings enconding_dom.pickle

> recognize_faces_image.py

> recognize_faces_video.py
❯ python recognize_faces_video.py --encodings encodings_dom.pickle --output output/first_video_3.avi --display 1
