import cv2
from picamera import PiCamera
from time import sleep
from picamera.array import PiRGBArray
import numpy as np
import copy

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8) 
    return (line_img, lines)

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, a=0.8, b=1., g=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * a + img * b + hg
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, b, g)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
def distancePoint(x1,y1, x2, y2):
	diff1 = x1 - x2
	if (0 > diff1):
		diff1 = diff1 * (-1)

	diff2 = y1 - y2
	if (0 > diff2):
		diff2 = diff2 * (-1)

	return  (diff1 + diff2) / 2

def distanceLines(line1, line2):
	distance = distancePoint(line1[0], line1[1], line2[0], line2[1]) + distancePoint(line1[2], line1[3], line2[2], line2[3])
	return distance / 2

def averageLines(lines):
	threshold = 60
	i = 0
	k = 0
	newList=[]	
	alreadyDetected=[]
	while (k < len(lines)):
		alreadyDetected.insert(k,0)
		k+=1
	while (i < len(lines)):
		j = i + 1
		while (j < len(lines)):
			if (threshold > distanceLines(lines[i][0], lines[j][0])):
				alreadyDetected[j]=1
			j = j +1
		i = i +1
	i = 0
	while (i < len(lines)):
		if(0== alreadyDetected[i]):
			newList.insert(len(newList), lines[i])
		i+=1
	return newList
def getCarMargins(image):
	#width in procent
	width = 50
	height = 10
	imgShape=image.shape
	return [(imgShape[1]/2 - (imgShape[1] * width /100 / 2),imgShape[0]), (imgShape[1]/2 + (imgShape[1] * width /100 / 2),imgShape[0]-(imgShape[0] * height/100))]
def drawCarDirection(image, margins):
	#width in procent
	width = 50
	imgShape=image.shape
	cv2.rectangle(image, margins[0], margins[1], [0,255,0], 2)
def getOrientation(image, lines, margins):
	center = [(margins[0][0] + margins[1][0]) / 2, (margins[0][1] + margins[1][1]) / 2]
	laneEnd = []

	if (1 < len(lines)):
		i = 0
		while (2 > i):
			closePoint = lines[i][0]
			secondPoint = copy.copy(lines[i][0])
			#print closePoint, secondPoint, closePoint,distancePoint(secondPoint[2], secondPoint[3], center[0], center[1]),distancePoint(closePoint[0], closePoint[1], center[0], center[1])
			if (distancePoint(closePoint[0], closePoint[1], center[0], center[1]) > distancePoint(secondPoint[2], secondPoint[3], center[0], center[1])):
				closePoint[0] = secondPoint[2]
				closePoint[1] = secondPoint[3]
				closePoint[2] = secondPoint[0]
				closePoint[3] = secondPoint[1]
			i = i + 1

			cv2.circle(image, (closePoint[0], closePoint[1]), 20,[0,0,255], 2)
			laneEnd.insert(len(laneEnd), [closePoint[0], closePoint[1]])

	cv2.line(image, (center[0], center[1]), (center[0], 100), [0,255,0], 2)
	laneLocLeft = laneEnd[0]
	laneLocRight = laneEnd[1]
	if (laneLocLeft[0] > laneLocRight[0]):
		laneLocRight = laneLocLeft
		laneLocLeft = laneEnd[1]
	print "lane left=",laneLocLeft," lane right=", laneLocRight
	
	distanceLeftLaneCenter = center[0] - laneLocLeft[0]
	distanceRightLaneCenter = laneLocRight[0] - center[0]
	distanceMedium = (distanceLeftLaneCenter + distanceRightLaneCenter) / 2
	distanceLeftLaneProcent = (distanceLeftLaneCenter * 100) / distanceMedium
	distanceRightLaneProcent = (distanceRightLaneCenter * 100) / distanceMedium

	print "Left procent=", distanceLeftLaneProcent, "Right procent=", distanceRightLaneProcent

	cv2.line(image, (laneLocLeft[0], laneLocLeft[1]), (center[0], center[1]),[0,0,255],2)
	cv2.putText(image, str(distanceLeftLaneCenter)+" "+str(distanceLeftLaneProcent)+"%", (laneLocLeft[0] + 20,laneLocLeft[1]-10), 0, 2, [0,0,255],2,cv2.LINE_AA)
	
	cv2.line(image, (laneLocRight[0], laneLocRight[1]), (center[0], center[1]),[0,0,255],2)
	cv2.putText(image, str(distanceRightLaneCenter) + " "+ str(distanceRightLaneProcent)+"%", (laneLocRight[0] - 300,laneLocRight[1]-10), 0, 2, [0,0,255],2,cv2.LINE_AA)

	print distanceMedium

# Set up PiCamera and let it warm up
camera = PiCamera()
raw = PiRGBArray(camera)

camera.capture(raw, format="bgr")
image = raw.array


cv2.imwrite("org.jpg", image)



gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
img_hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

lower_yellow = np.array([20, 100, 100], dtype = "uint8")
upper_yellow = np.array([30, 255, 255], dtype="uint8")
mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
mask_white = cv2.inRange(gray, 200, 255)
mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
mask_yw_image = cv2.bitwise_and(gray, mask_yw)

kernel_size = 5
gauss_gray = cv2.GaussianBlur(mask_yw_image,(kernel_size,kernel_size), 0)

low_threshold = 50
high_threshold = 150
canny_edges = cv2.Canny(gauss_gray,low_threshold,high_threshold)

imshape = image.shape
lower_left = [imshape[1]/9,imshape[0]]
lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/5]
top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/5]

lower_left=[0,imshape[0] - 20]
lower_right=[imshape[1],imshape[0] - 20]
top_right=[imshape[1],imshape[0]*1/4]
top_left=[imshape[1]/5,imshape[0]*1/4]


vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
roi_image = region_of_interest(canny_edges, vertices)

#rho and theta are the distance and angular resolution of the grid in Hough space
#same values as quiz
rho = 2
theta = np.pi/180
#threshold is minimum number of intersections in a grid for candidate line to go to output
threshold = 150
min_line_len = 50
max_line_gap = 300
houghResult = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)
line_image = houghResult[0]
line_image_avg = copy.copy(line_image)
lines = houghResult[1]
draw_lines(line_image, lines)

averageLines = averageLines(lines)
draw_lines(line_image_avg, averageLines)


margins = getCarMargins(line_image_avg)
print margins
drawCarDirection(line_image_avg, margins)
print 'get orientation'
getOrientation(line_image_avg, averageLines, margins)


result = weighted_img(line_image, image, a=0.8, b=1., g=0.)



cv2.imwrite("a.jpg", gray)
cv2.imwrite("b.jpg", img_hsv)
cv2.imwrite("c.jpg", mask_yw_image)
cv2.imwrite("d.jpg", gauss_gray)
cv2.imwrite("e.jpg", canny_edges)
cv2.imwrite("f.jpg", roi_image)
cv2.imwrite("g.jpg", line_image)
cv2.imwrite("h.jpg", line_image_avg)
cv2.imwrite("j.jpg", result)
