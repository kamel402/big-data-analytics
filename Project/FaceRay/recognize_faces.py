# import the necessary packages
import face_recognition
import argparse
from camera import VideoCam
import pickle
import cv2
import ray
import imutils
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-r", "--resize", type=int, default=480,
	help="resize the image")
ap.add_argument("-v", "--video", default=argparse.SUPPRESS, type=str,
	help="path to the (optional) video file")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

DATABASE = None

def detection(image):
	# detect the (x, y)-coordinates of the bounding boxes corresponding
	# to each face in the input image, then compute the facial embeddings
	# for each face
	boxes = face_recognition.face_locations(image,
		model=args["detection_method"])
	
	encodings = face_recognition.face_encodings(image, boxes)

	return boxes, encodings

@ray.remote
def recognize(encoding):

	matches = face_recognition.compare_faces(DATABASE["encodings"],
		encoding)
	name = "Unknown"

		# check to see if we have found a match
	if True in matches:
		# find the indexes of all matched faces then initialize a
		# dictionary to count the total number of times each face
		# was matched
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}
		# loop over the matched indexes and maintain a count for
		# each recognized face face
		for i in matchedIdxs:
			name = DATABASE["names"][i]
			counts[name] = counts.get(name, 0) + 1
		# determine the recognized face with the largest number of
		# votes (note: in the event of an unlikely tie Python will
		# select first entry in the dictionary)
		name = max(counts, key=counts.get)
	else:
		return None

	return name


def draw(image, boxes, names):
	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image
		cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)
	return image


if __name__ == '__main__':
    
	ray.init()

	# load the known faces and embeddings
	print("[INFO] loading encodings...")
	DATABASE = pickle.loads(open(args["encodings"], "rb").read())


	# Get the video
	source = args["video"]

	(H, W) = (None, None)

	# Infernce 
	cap = VideoCam(source)
	cap.check_camera(cap.cap)
	print("[INFO] Start recognizing")

	ct = 0
	start_time = time.time()

	while True:
		ct += 1
		try:
			ret = cap.cap.grab()
			
			ret, frame = cap.get_frame()
			if not ret:
				break

			if W is None or H is None:
				(H, W) = frame.shape[:2]
			
			frame = imutils.resize(frame, width=args["resize"])

			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			boxes, encodings = detection(rgb)

			names = []

	
			futures = [recognize.remote(encoding) for encoding in encodings]
			names = ray.get(futures)

			names = [i for i in names if i]

			print("[INFO] Recognized {0} in Frame {1}".format(names, ct))

		except KeyboardInterrupt:
			cap.close_cam()
			exit(0)

	print('[INFO] This video took: '+str(time.time() - start_time)+' sec')

	cap.close_cam()
	exit(0)