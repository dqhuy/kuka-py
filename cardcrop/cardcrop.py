"""
KUKA LIB - ID dectecing and croping module
Input: image containt ID document like Nationa ID Card, Student card, Bank card...
Output: 
    Kuka try to detect 4 borders of document, 
    if getting borders success then trying to crop and do affine transform
    return a croped & affine transform image
"""

import cv2
import numpy as np
import os;
import sys

#common function

def findIntersection (params1, params2):
    """
    @params1,@paramas2: input tuple (a,b,c) of a line
    """
    x = -1
    y = -1
    det = params1[0] * params2[1] - params2[0] * params1[1]
    if (det < 0.5 and det > -0.5): #lines are approximately parallel
        return (-1, -1)
    else:
        x = (params2[1] * -params1[2] - params1[1] * -params2[2]) / det;
        y = (params1[0] * -params2[2] - params2[0] * -params1[2]) / det;
    return (int(x), int(y))

#line's equation computation
def calcParams(p1,p2):
    """
    Parameters
    @p1,@p1: two point of line p1=(x1,y1) p2=(x2,y2) 
    """
    if p2[1] - p1[1] ==0:
        a = 0.0
        b = -1.0
    elif p2[0] - p1[0] ==0:
        a = -1.0
        b =0.0
    else:
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = -1.0
    c = (-a * p1[0]) - b * p1[1]
    return a,b,c

def order_points(pts):
    """
    Rearrange coordinates to order:
      top-left, top-right, bottom-right, bottom-left
      
    Parameters
    -
        @pts: list of point
    """

    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]
 
    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # Return the ordered coordinates.
    return rect.astype('int').tolist()

def find_dest(pts):
    """
    Find destination corner for preparing cropping and making affine transform
    """
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
 
    return order_points(destination_corners)
    
def getLineAngle(p1, p2):
    x1,y1=p1
    x2,y2=p2
    xDiff = x2 - x1
    yDiff = y2 - y1
    return np.arctan2(yDiff, xDiff)*(180 / np.pi)

def getLineLength(p1, p2):
    x1,y1=p1
    x2,y2=p2
    xDiff = x2 - x1
    yDiff = y2 - y1
    return np.sqrt((xDiff**2)+(yDiff**2))

def cardCrop(src,debug=False):
    """
    try to detect 4 borders of id card in image 
    then making croping and perspective transforming
    by using hough technic
    
    Parameter
    -
        @src: color image containt id card needed to crop
    
    Return
    -
        Return croped image if detect card borders success 
        otherwise return empty
    """
    cropedImg=src.copy()
    lineImg=src.copy()

    ## Step 1: try to extract background
    ### using Morph_close
    kernel=np.ones((7,7),np.uint8)
    dilectImg=cv2.morphologyEx(src,cv2.MORPH_CLOSE,kernel,iterations=2)
    
    gray=cv2.cvtColor(dilectImg,cv2.COLOR_BGR2GRAY)
    
    blurImg=cv2.GaussianBlur(gray,(5,5),0)
    
    ret1,threshImg=cv2.threshold(blurImg,70,200,cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    
    ## Step 2: Edge and line dectection => Detect quadrilateral
    
    edgeImg=cv2.Canny(threshImg,70,200)
    lines = cv2.HoughLinesP(edgeImg,rho=3,theta=0.3*np.pi/180,threshold=30,minLineLength=30,maxLineGap=10)

    contours,hierachy=cv2.findContours(edgeImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    sortedContours=sorted(contours,key=cv2.contourArea,reverse=True)

    #duyệt các lines
        # top-line , b-line là đường nằm trên / dưới horizontal  midle và có độ dài lớn nhất, có theta
        # left-line / right-line lác các đường nằm bên trái / phải của đường vertical midle
    maxContours=sortedContours[0]
    x, y, w, h = cv2.boundingRect(maxContours)

    hLine=(x,y+h/2,x+w,y+h/2)
    vLine=(x+w/2,y, x+w/2,y+h)

    hLine=np.array(hLine).astype(np.int32)
    vLine=np.array(vLine).astype(np.int32)

    maxTLineLen=0
    maxBLineLen=0
    maxLLineLen=0
    maxRLineLen=0
    topline=[]
    bottomline=[]
    leftline=[]
    rightline=[]
    for line in lines:
        p1=line[0][0:2]
        p2=line[0][2:4]
        angle= getLineAngle(p1,p2)
        angle=abs(angle)

        length=getLineLength(p1,p2)
        if(angle<45): # top/bottom line
            if(p1[1]<hLine[1]):
                if maxTLineLen<length:
                    maxTLineLen=length
                    topline=line[0]
            else:
                if maxBLineLen<length:
                    maxBLineLen=length
                    bottomline=line[0]
        else:
            if(p1[0]<vLine[0]):
                if maxLLineLen<length:
                    maxLLineLen=length
                    leftline=line[0]
            else:
                if maxRLineLen<length:
                    maxRLineLen=length
                    rightline=line[0]

    if(len(topline)>0 and len(bottomline)>0 and len(leftline)>0 and len(rightline)>0):
        # tính hệ số (a,b,c) của các đường thẳng
        aL,bL,cL=calcParams(leftline[0:2],leftline[2:4])
        aT,bT,cT=calcParams(topline[0:2],topline[2:4])

        aR,bR,cR=calcParams(rightline[0:2],rightline[2:4])
        aB,bB,cB=calcParams(bottomline[0:2],bottomline[2:4])

        # tìm các giao điểm tl,tr,bl,br
        topleftPoint=findIntersection((aL,bL,cL),(aT,bT,cT))
        toprightPoint=findIntersection((aR,bR,cR),(aT,bT,cT))
        bottomleftPoint=findIntersection((aL,bL,cL),(aB,bB,cB))
        bottomrightPoint=findIntersection((aR,bR,cR),(aB,bB,cB))



        # check if the polygon has four point
        corners=np.array([topleftPoint,toprightPoint,bottomleftPoint,bottomrightPoint])

        corners = order_points(corners)
        
        destination_corners = find_dest(corners)
        
        h, w = src.shape[:2]
        # Getting the homography.
        M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
        # Perspective transform using homography.
        cropedImg = cv2.warpPerspective(src, M, (destination_corners[2][0], destination_corners[2][1]),
                                        flags=cv2.INTER_LINEAR)

        cv2.line(lineImg,hLine[0:2],hLine[2:4],(0,0,255),3)
        cv2.line(lineImg,vLine[0:2],vLine[2:4],(0,0,255),3)

        # cv2.line(lineImg,topline[0:2],topline[2:4],(0,0,255),1)
        # cv2.line(lineImg,bottomline[0:2],bottomline[2:4],(0,0,255),1)
        # cv2.line(lineImg,leftline[0:2],leftline[2:4],(0,0,255),1)
        # cv2.line(lineImg,rightline[0:2],rightline[2:4],(0,0,255),1)

        cv2.circle(lineImg,topleftPoint,3,(0,0,255),3)
        cv2.circle(lineImg,toprightPoint,3,(0,0,255),3)
        cv2.circle(lineImg,bottomleftPoint,3,(0,0,255),3)
        cv2.circle(lineImg,bottomrightPoint,3,(0,0,255),3)
        
        cv2.line(lineImg,topleftPoint,toprightPoint,(0,0,255),3)
        cv2.line(lineImg,topleftPoint,bottomleftPoint,(0,0,255),3)
        cv2.line(lineImg,bottomrightPoint,toprightPoint,(0,0,255),3)
        cv2.line(lineImg,bottomrightPoint,bottomleftPoint,(0,0,255),3)
        
    
    if(debug):
        cv2.imshow("debug",lineImg)

    return cropedImg,lineImg

if __name__ == '__main__':
    path=sys.argv[1]

    if(os.path.exists(path)==False):
        print('Path is not exist')
        sys.exit()
    
    if(os.path.isdir(path)):
        #create output dir
        outpath=os.path.join(path,"out")
        if(os.path.exists(outpath)==False):
            os.mkdir(outpath)
        for f in os.listdir(path):
            filename=os.path.join(path,f)
            if(os.path.isfile(filename)==False):
                continue

            src=cv2.imread(filename)   
            if(type(src)!=np.ndarray):
                continue

            print(filename," ---> ", end='')
            cropedImg,debugImg=cardCrop(src)
            savedFilename=os.path.join(outpath,f)
            savedDebugFileName=os.path.join(outpath,"debug_"+f)
            cv2.imwrite(savedFilename,cropedImg)
            cv2.imwrite(savedDebugFileName,debugImg)
            print("Done")

    else:
        filename=path
        src=cv2.imread(filename)    
        cropedImg=cardCrop(src,debug=True)
        cv2.imshow("Croped Image",cropedImg)
        cv2.waitKey()
        cv2.destroyAllWindows()