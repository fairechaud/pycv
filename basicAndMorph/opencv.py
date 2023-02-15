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
cv2.waitKey(0)

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

# OpenCV doesn't "care" if our rotated image is clipped after rotation
# so we can instead use another imutils convenience function to help
# us out
rotated = imutils.rotate_bound(image, 45)
cv2.imshow("Imutils Bound Rotation", rotated)
cv2.waitKey(0)

# apply a Gaussian blur with a 11x11 kernel to the image to smooth it,
# useful when reducing high frequency noise
"""
Larger kernels would yield a more blurry image. Smaller kernels will create less blurry images. See https://pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
"""
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

# draw a 2px thick red rectangle surrounding the face
output = image.copy()
cv2.rectangle(output, (320, 60), (420, 160), (255, 0, 0), 2)
cv2.imshow("Rectangle", output)
cv2.waitKey(0)

# draw a blue 20px (filled in) circle on the image centered at
# x=300,y=150
output = image.copy()
cv2.circle(output, (300, 150), 20, (0, 255, 0), -1)
cv2.imshow("Circle", output)
cv2.waitKey(0)

# draw a 5px thick red line from x=60,y=20 to x=400,y=200
output = image.copy()
cv2.line(output, (60, 20), (400, 200), (0, 0, 255), 5)
cv2.imshow("Line", output)
cv2.waitKey(0)

# draw green text on the image
output = image.copy()
cv2.putText(output, "OpenCV + Jurassic Park!!!", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Text", output)
cv2.waitKey(0)
#tutorial from https://pyimagesearch.com/2018/07/19/opencv-tutorial-a-guide-to-learn-opencv/