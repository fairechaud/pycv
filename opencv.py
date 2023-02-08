# import the necessary packages
import imutils
import cv2
# load the input image and show its dimensions, keeping in mind that
# images are represented as a multi-dimensional NumPy array with
# shape no. rows (height) x no. columns (width) x no. channels (depth)
image = cv2.imread("jp.png")
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))
# display the image to our screen -- we will need to click the window
# open by OpenCV and press a key on our keyboard to continue execution
cv2.imshow("Image", image)
#cv2.waitKey(0)

(B,G,R) = image[100,60]
print("R={}, G={}, B={}".format(R,G,B))

roi = image[60:160, 320:420]
cv2.imshow("ROI", roi)
#cv2.waitKey(0)


""" 
#new aspect ratio based on width = 300 px
r = 300.0/w
newDim = (300, int(h*r))
resized = cv2.resize(image, newDim)
cv2.imshow("Aspect ratio new size", resized)
cv2.waitKey(0) 
"""

resized = imutils.resize(image, width = 300)
cv2.imshow("Imutils resize", resized)
cv2.waitKey(0)
