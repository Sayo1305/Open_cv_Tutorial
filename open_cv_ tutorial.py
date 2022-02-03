#open cv tutorial 
import hmac
import os
import sys
from turtle import color
import cv2
import numpy as np
import matplotlib as mp
from numpy.lib.function_base import hamming
from numpy.lib.twodim_base import diagflat

"""
# to import an image into the file [cv2.imread()]
img  = cv2.imread("C:\\Users\\Hp\\Downloads\\wp4124575.jpg");
# img is the name of the image

# to show the image [cv2.imshow]
cv2.imshow("output",img);
# output is the window name of the display
# img is the varriable name of the image 

cv2.waitKey(0);
# waitkey is for the wait for the exit of the program otherwise the program will run for 1 milisec only.
# 0 for infinite and 1000 for 1 second
"""

"""
# to read the video in a varriable called vid 
vid = cv2.VideoCapture("C:\\Users\\Hp\\Documents\\iFun Screen Recorder\\Outputs\\20220101_201800.mp4");

# the video is continous loop of from so we have to use the while till the last frame and run it
while True:
  sucess, img = vid.read()
  cv2.imshow("video",img)
  # the below condition is for if the time is 1 msec and if we press q then it quit
  if(cv2.waitKey(1) & 0XFF == ord('q')):
    break
"""
"""
# to run the webcam we use same method as video reading

vid = cv2.VideoCapture(0)  # for your local webcam id = 0
# to costomize the video
vid.set(3,1280) # 3 is the id for width
vid.set(4,720) # 4 is the id for hieght
vid.set(10,100) # 10 is the id for brightness
# loop for running the webcam and show the each frame
while(True):
  success, img = vid.read()
  cv2.imshow("video",img)
  # the below condition is for if the time is 1 msec and if we press q then it quit
  if(cv2.waitKey(1) & 0XFF == ord('q')):
    break
"""

# chapter 2 (basic function) ------------------------------------------------------------------------------------------------------------------------------------
"""
# convert RGB to GRAY
# covert to blur
img1 = cv2.imread("C:\\Sayo\\IMG_20210813_211255_mfnr.jpg")

# to resize the image 
img = cv2.resize(img1,(600,720))
# cv2.cvtColor is used to change the color format of the image like RGB To GRAY  etc 
grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# to blur the image we use gaussian blur
# note  = in ksize the value should be (odd,odd) highher the ksize more is the blur
blur = cv2.GaussianBlur(img,(9,9),0)

# to get the edge of the image we use canny
# the thresshold 1 and 2 in canny is the edge acuracy lower the value higher is the cauracy
imgedge = cv2.Canny(img,50,50)

# sometimes the edges are so fine that the line can't be drawn so we use the dialation for edges
# we have to define the kernal as an array of one so we use numpy
kernal = np.ones((5,5),np.uint8)
dilate = cv2.dilate(imgedge,kernal,iterations=1) # on increasing the number of iteration the thickness of edges increase

# the opposite of dilation is eroded
eroimg = cv2.erode(dilate,kernal,iterations=1)

# to diplay the size of the image 
#print(img.shape) #(720,600,3) 720 is hight and 600 with and 3 is RBG 

#cv2.imshow("original",img)
#cv2.imshow("GRAY",grayimg)
#cv2.imshow("BLUR",blur)
cv2.imshow("edge",imgedge)
cv2.imshow("dilate",dilate)
cv2.imshow("erord",eroimg)
cv2.waitKey(0)
"""
# chapter 3 ( resizing and croping)- --------------------------------------------------------------------------------------------------------------
"""
# how to resize the image 
img = cv2.imread("C:\\Users\\Hp\\Downloads\\R (6).jfif")
print(img.shape)
# print the dimension of the image in (x,y,z) format x is height, y is width, and z is the number of color format like RGB

imgshort = cv2.resize(img,(640,480))
# resize the image into width, height
# cv2.imshow("Original",img)
# cv2.imshow("resize",imgshort)

# to make a crop image

cropimg = img[0:200,100:300]
cv2.imshow("Original",img)
cv2.imshow("cropped",cropimg)
cv2.waitKey(0)
"""
# chapter 4 (shapes and text) --------------------------------------------------------------------------------------------------------------------
"""
# to make a screen of black window of size n x n
img = np.zeros((640,480,3),np.uint8)
print(img.shape)
# to make the area described as green
#img[0:320,0:240] = 0,255,0
#cv2.imshow("ori",img)
# to draw a line using cv2.line of(diffrent thickness)

#cv2.line(img,(0,300),(480,300),(0,255,0),3)
#cv2.line(img,(0,200),(480,200),(255,255,0),2)
#cv2.line(img,(0,500),(480,500),(0,255,255),1)
#cv2.line(img,(0,350),(480,350),(200,255,255),5)

# to draw the ractangle
cv2.rectangle(img,(0,0),(300,300),(0,255,0),3)

# to amke a rectangle of filled coloe
cv2.rectangle(img,(300,300),(img.shape[1],img.shape[0]),(255,0,0),cv2.FILLED)

# to make a circle 
#(filed)
cv2.circle(img,(340,250),100,(0,0,255),cv2.FILLED)

# without filled
cv2.circle(img,(340,250),100,(0,255,255),3)

# to write text on screen
# here scale is the size of the text and thickness is the bolderness of the text
cv2.putText(img,"Unnat Das",(0,300),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),1)

cv2.imshow("ori",img)
cv2.waitKey(0)
"""
# chapter 5 (perspective wraping of the image)-------------------------------------------------------------------------------
"""
# means: to take a particular component of the image and show it on the other side
img = cv2.imread("C:\\Users\\Hp\\Downloads\\tomandjerry.jpg")
wi = 250
hi = 300
cv2.imshow("output",img)

pt1 = np.float32([[70,120],[400,120],[400,530],[70,530]])
pt2 = np.float32([[0,0],[wi,0],[0,hi],[wi,hi]])

maxtrix = cv2.getPerspectiveTransform(pt1,pt2)
new_img = cv2.warpPerspective(img,maxtrix,(wi,hi))
cv2.imshow("output",img)
cv2.imshow("output2",new_img)
cv2.waitKey(0)
"""
# chapter 6 (joining image)
"""
# for joining the two image we can use hroizontal or vertical stack to implement this 
img = cv2.imread("C:\\Users\\Hp\\Downloads\\tomandjerry.jpg")
img = cv2.resize(img,(600,400))

imghstack = np.hstack([img,img,img])
imgvstack = np.vstack([img,img,img])

cv2.imshow("djd",imghstack)
cv2.imshow("dj",imgvstack)
cv2.waitKey(0)
"""


# to make track bar in the image-----------------------------------------------------------------------------------------------
"""

def fun(a):
  pass


# create trackbar
cv2.namedWindow("trackbar")
cv2.resizeWindow("trackbar",400,300)
# making diffrent trackbar
cv2.createTrackbar("Hue Min","trackbar",0,180,fun)
cv2.createTrackbar("Hue Max","trackbar",16,180,fun)
cv2.createTrackbar("Sat Min","trackbar",135,255,fun)
cv2.createTrackbar("Sat Max","trackbar",208,255,fun)
cv2.createTrackbar("Val Min","trackbar",83,255,fun)
cv2.createTrackbar("Val Max","trackbar",255,255,fun)




while True:
  img = cv2.imread("C:\\Users\\Hp\\Downloads\\tomandjerry.jpg")
  img = cv2.resize(img,(500,480))
  # convert to hsv format
  hsvimg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
  # create varriable to track the value of the trackbar
  h_min = cv2.getTrackbarPos("Hue Min","trackbar")
  h_max = cv2.getTrackbarPos("Hue Max","trackbar")
  s_min = cv2.getTrackbarPos("Sat Min","trackbar")
  s_max = cv2.getTrackbarPos("Sat Max","trackbar")
  v_min = cv2.getTrackbarPos("Val Min","trackbar")
  v_max = cv2.getTrackbarPos("Val Max","trackbar")
  print(h_min,h_max,s_min,s_max,v_min,v_max)
  low = np.array([h_min,s_min,v_min])
  high = np.array([h_max,s_max,v_max])
  mask = cv2.inRange(hsvimg,low,high)
  imgres = cv2.bitwise_and(img,img,mask=mask)
  imgstack = np.hstack([imgres,img,hsvimg])
  cv2.imshow("result",imgstack)
  if(0XFF == ord('q')):
    break
  cv2.waitKey(1) 
"""


"""
# shape detection using contors
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

# main contour function
# contour: a curve joining all the continuous points (along the boundary), having same color or intensity.

def get_contour(img):
    coutour, hirechy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # to get the item or elemet in a image
    for cnt in coutour:
        area = cv2.contourArea(cnt) # to get the area of the item 
        #print(area)
        cv2.drawContours(imgcopy,cnt,-1,(255,0,0),5) # to draw the counter point of the element in the image
        peri = cv2.arcLength(cnt,True) # to get the perimeter of the items
        #print(peri)
        aprox = cv2.approxPolyDP(cnt,0.02*peri,True)
        #print(len(aprox)) print the number corner in the element
        x, y, w, h = cv2.boundingRect(aprox)
        obji = len(aprox)
        if(obji == 3):
            objtype = "triangle"
        elif(obji == 4):
            objtype = "rectangle"
        elif(obji == 8):
            objtype = "circle"
        else:
            objtype="NA"

        cv2.rectangle(imgcopy,(x,y),(x+w,y+h),(0,255,0),4)
        cv2.putText(imgcopy,objtype,(x+(w//2)-5,y+(h//2)-5),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,0),2)


img = cv2.imread("C:\\Users\\Hp\\Documents\\shape.png")
imgcopy = img.copy()
imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgblur = cv2.GaussianBlur(imggray,(7,7),1)
imgcanny = cv2.Canny(imgblur,100,100)
get_contour(imgcanny)
stack = stackImages(0.6,([img,imggray,imgblur],[imgcanny,imgcopy,img]))
cv2.imshow("result",stack)

cv2.waitKey(0)
"""