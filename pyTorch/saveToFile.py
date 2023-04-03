from torchvision.models import detection
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import torch
import time
import cv2

#uncomment to allow cl arguments, otherwise use default variables
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-m", "--model", type=str, default="mobilenet",
	choices=["resnet", "mobilenet", "retinanet"],
	help="name of the object detection model")
ap.add_argument("-l", "--labels", type=str, default="resources/coco_classes.pickle",
	help="path to file containing list of categories in COCO dataset")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-co", "--codec", type=str, default="MJPG",
	help="codec of output video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
args = vars(ap.parse_args())



# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load the list of categories in the COCO dataset and then generate a
# set of bounding box colors for each class

# uncomment below for cl arguments
CLASSES = pickle.loads(open(args["labels"], "rb").read())
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# initialize a dictionary containing model name and its corresponding 
# torchvision function call
MODELS = {
	"resnet": detection.fasterrcnn_resnet50_fpn,
	"mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
	"retinanet": detection.retinanet_resnet50_fpn
}

model = MODELS[args["model"]](pretrained=True, progress=True,
	num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model.eval()

# initialize the video stream, allow the camera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")

if args.get("video", None) is None:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])
fps = FPS().start()

fourcc = cv2.VideoWriter_fourcc(*args["codec"])
writer = None
(h, w) = (None, None)
zeros = None
f= int(vs.get(cv2.CAP_PROP_FPS))

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = frame if args.get("video", None) is None else frame[1]
	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if frame is None:
		break

	frame = imutils.resize(frame, width=400)
	orig = frame.copy()
	# convert the frame from BGR to RGB channel ordering and change
	# the frame from channels last to channels first ordering
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = frame.transpose((2, 0, 1))
	# add a batch dimension, scale the raw pixel intensities to the
	# range [0, 1], and convert the frame to a floating point tensor
	frame = np.expand_dims(frame, axis=0)
	frame = frame / 255.0
	frame = torch.FloatTensor(frame)
	# send the input to the device and pass the it through the
	# network to get the detections and predictions
	frame = frame.to(DEVICE)
	detections = model(frame)[0]
	
    	# loop over the detections
	for i in range(0, len(detections["boxes"])):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections["scores"][i]
		if confidence > args["confidence"]:	
			# extract the index of the class label from the
			# detections, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections["labels"][i])
			box = detections["boxes"][i].detach().cpu().numpy()
			(startX, startY, endX, endY) = box.astype("int")
			# draw the bounding box and label on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			cv2.rectangle(orig, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(orig, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
				# show the output frame

	cv2.imshow("Frame", orig)
	# check if the writer is None
	if writer is None:
		# store the image dimensions, initialize the video writer,
		# and construct the zeros array
		
		(h, w) = (orig.shape[:2])
		writer = cv2.VideoWriter(args["output"], fourcc, f,
			(w, h), True)
		zeros = np.zeros((h, w), dtype="uint8")
			# break the image into its RGB components, then construct the
	# RGB representation of each frame individually
	
	output = np.zeros((h,w,3), dtype="uint8")
	output[0:h, 0:w] = orig
	# write the output frame to file
	writer.write(output)


	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key was pressed, break from the loop
	if key == ord("q"):
		break
	# update the FPS counter
	fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()