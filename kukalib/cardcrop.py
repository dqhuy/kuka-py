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
import os
import sys
import datetime
import math
# getversion
def getVersionInfo():
    versionInfo = {"version": "0.3.2" ,
               "date": datetime.date(2023,6,19)   
               }
    return versionInfo
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
    BE CAREFULL: this is NOT correct if input points is not in shape of square of rectangular
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

def get2Lineangle(line1, line2):
    """
    Get angle of two lines
    Input line in shape (2,2)
    """
    # Get directional vectors
    d1 = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1])
    d2 = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1])
    # Compute dot product
    p = d1[0] * d2[0] + d1[1] * d2[1]
    # Compute norms
    n1 = np.sqrt(d1[0] * d1[0] + d1[1] * d1[1])
    n2 = np.sqrt(d2[0] * d2[0] + d2[1] * d2[1])
    # Compute angle
    ang = math.acos(p / (n1 * n2))
    # Convert to degrees if you want
    ang = np.rad2deg(ang)
    return ang

def sobel(gray):
    'Get edge image by sobel operators'
    ddepth=cv2.CV_16S
    gradX=cv2.Sobel(gray,ddepth=ddepth,dx=1,dy=0,ksize=3)
    gradY=cv2.Sobel(gray,ddepth=ddepth,dx=0,dy=1,ksize=3)
    abs_gradX=cv2.convertScaleAbs(gradX)
    abs_gradY=cv2.convertScaleAbs(gradY)
    grad=cv2.addWeighted(abs_gradX,1,abs_gradY,1,0)
    return grad

def isPoinOnLeft(line,point):
    """
    line in shape (2,2) , point in shape(1,2)
    return: true if point is left/above line
    """
    a=line[0]
    b=line[1]
    c=point
    return (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]) > 0;

def verifyQuadrilateral(toplineExtra,bottomlineExtra,leftlineExtra,rightlineExtra,debug=False):
    """
    Check 4 lines can create a  quadrangle shape that meet with following criterion
    Input: lines in shape (x1,y1,x2,y2,length,angle) 
    """
    isQuadrangle=True
    criterion = True
    criterionResult=[]

    topline = toplineExtra[0:4].reshape(2,2)
    bottomline = bottomlineExtra[0:4].reshape(2,2)
    leftline = leftlineExtra[0:4].reshape(2,2)
    rightline = rightlineExtra[0:4].reshape(2,2)

    a_angle = get2Lineangle(topline,leftline)
    b_angle = get2Lineangle(topline,rightline)
    c_angle = get2Lineangle(bottomline,rightline)
    d_angle = get2Lineangle(bottomline,leftline)

    if(debug):
        print("Lines top: {} bottom: {} left: {} right{}".format(topline,bottomline,leftline,rightline))
        print("Angle A:{:.2f}\tB:{:.2f}\tC:{:.2f}\tD:{:.2f}".format(a_angle,b_angle,c_angle,d_angle))

   
    #criterion 1
    minParallelAngle=10 
    toplineAngle=toplineExtra[5]
    bottomlineAngle=bottomlineExtra[5]
    leftlineAngle = leftlineExtra[5]
    rightlineAngle =rightlineExtra[5]
    criterion = abs(abs(toplineAngle) - abs(bottomlineAngle))<minParallelAngle and abs(abs(leftlineAngle)-abs(rightlineAngle))<minParallelAngle
    isQuadrangle = isQuadrangle and criterion
    criterionResult.append(criterion)
    if(debug):
        print("Meet criterion 1:",criterion)
        print("\tLine angle of Top {:.2f} - bottom {:.2f}  - left: {:.2f} - right: {:.2f} ".format(toplineAngle,bottomlineAngle,leftlineAngle,rightlineAngle))
    
    #criterion 2
    diffOppositeAngle=10
    criterion = (a_angle-b_angle+c_angle-d_angle)/2 <diffOppositeAngle and (a_angle-d_angle+b_angle-c_angle)/2<diffOppositeAngle
    isQuadrangle = isQuadrangle and criterion
    criterionResult.append(criterion)
    if(debug):
        print("Meet criterion 2:",criterion)
    #criterion 3
    averagePerpendicular =25 
    criterion =  abs((a_angle+b_angle+c_angle+d_angle)/4-90) < averagePerpendicular
    isQuadrangle =isQuadrangle and criterion
    criterionResult.append(criterion)
    if(debug):
        print("Meet criterion 3:",criterion)    

    return criterionResult,isQuadrangle

def cropImage(src,topline,bottomline,leftline,rightline):
    """
    Crop image by 4 qualified lines
    Input 
        Line input in shape (1,4)
    Return debug image and cropped image
    """
    lineImg = src.copy()
    cropedImg = np.zeros(src.shape,dtype=np.uint8)

    'draw line to debug image'
    if len(topline)>0:
        cv2.line(lineImg,topline[0:2],topline[2:4],255,1)
    if len(bottomline)>0:
        cv2.line(lineImg,bottomline[0:2],bottomline[2:4],255,1)
    if len(leftline)>0:
        cv2.line(lineImg,leftline[0:2],leftline[2:4],255,1)
    if len(rightline)>0:
        cv2.line(lineImg,rightline[0:2],rightline[2:4],255,1)

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
        corners=np.array([topleftPoint,toprightPoint,bottomrightPoint,bottomleftPoint])

        destination_corners = find_dest(corners)
        
        # Getting the homography.
        M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
        # Perspective transform using homography.
        cropedImg = cv2.warpPerspective(src, M, (destination_corners[2][0], destination_corners[2][1]),
                                        flags=cv2.INTER_LINEAR)

    
        #show debug: middle vertical,hozirontal line of boundingbox
        x,y,w,h=cv2.boundingRect(np.float32(corners))
        w2=int(w/2)
        h2=int(h/2)
        
        cv2.line(lineImg,(x,y+h2),(x+w,y+h2),(0,0,255),5) #horizontal
        cv2.line(lineImg,(x+w2,y),(x+w2,y+h),(0,0,255),5) # vertical

        cv2.circle(lineImg,topleftPoint,3,(0,0,255),3)
        cv2.circle(lineImg,toprightPoint,3,(0,0,255),3)
        cv2.circle(lineImg,bottomleftPoint,3,(0,0,255),3)
        cv2.circle(lineImg,bottomrightPoint,3,(0,0,255),3)
        
        cv2.line(lineImg,topleftPoint,toprightPoint,(0,0,255),3)
        cv2.line(lineImg,toprightPoint,bottomrightPoint,(0,0,255),3)
        cv2.line(lineImg,bottomrightPoint,bottomleftPoint,(0,0,255),3)
        cv2.line(lineImg,bottomleftPoint,topleftPoint,(0,0,255),3)
    
    return cropedImg,lineImg,(topleftPoint,toprightPoint,bottomrightPoint,bottomleftPoint)


def cardCrop(src):
    """
    try to detecting 4 vertices of by using Hough Transform
    if has enough 4 vertices then crop it by using homography matrix
    if not return orginal input
		Step 1: try to remove background and card content
		step 2: detect edge by sobel
		step 3: get line by hough then find 4 vertices by longest lines for each side (top / bottom / left / right)
		step 4: make homography transform and return result
    
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
    ### using Morph_close to remove text
    kernel=np.ones((7,7),np.uint8)
    dilectImg=cv2.morphologyEx(src,cv2.MORPH_CLOSE,kernel,iterations=5)
    
    gray=cv2.cvtColor(dilectImg,cv2.COLOR_BGR2GRAY)
    
    blurImg=cv2.GaussianBlur(gray,(5,5),0)
    
    ## Step 2: Edge and line dectection => Detect quadrilateral
    
    sobelImg=sobel(blurImg)
    ret1,edgeImg = cv2.threshold(sobelImg,40,200, cv2.THRESH_OTSU + cv2.THRESH_TOZERO + cv2.THRESH_BINARY)

    ## make egde more dilect
    edgeImg=cv2.morphologyEx(edgeImg,cv2.MORPH_DILATE,(3,3),iterations=2)

    lines = cv2.HoughLinesP(edgeImg,rho=1,theta=1*np.pi/180,threshold=50,minLineLength=30,maxLineGap=10)

    # contours,hierachy=cv2.findContours(edgeImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    # sortedContours=sorted(contours,key=cv2.contourArea,reverse=True)
    """
    Step 3: trying to find quadrilateral by estimating 4 lines: top/bottom , left/right
    Asumption:
        - card is nearly center of image
        - 4 edges of card nearly present in the image
    """
    
    
    # listPoint=lines.reshape(lines.shape[0]*2,2)
    # minX=min(listPoint[:,0])
    # maxX=max(listPoint[:,0])
    # minY=min(listPoint[:,1])
    # maxY=max(listPoint[:,1])

    x=0
    y=0
    w=src.shape[1]
    h=src.shape[0]

    #assumpt: card is center of image and the skew angle is less than 45 degree
    xcenter=w/2
    ycenter=h/2

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
    topleftPoint=[]
    toprightPoint=[]
    bottomleftPoint=[]
    bottomrightPoint=[]
    
   
    for line in lines:
        p1=line[0][0:2]
        p2=line[0][2:4]
        angle= getLineAngle(p1,p2)
        angle=abs(angle)

        length=getLineLength(p1,p2)
        if(angle<40): # top/bottom line
            if(p1[1]+p2[1])/2 < ycenter:
                if maxTLineLen<length:
                    maxTLineLen=length
                    topline=line[0]
            elif (p1[1]+p2[1])/2 > ycenter:
                if maxBLineLen<length:
                    maxBLineLen=length
                    bottomline=line[0]
        else:
            if (p1[0]+p2[0])/2 < xcenter:
                if maxLLineLen<length:
                    maxLLineLen=length
                    leftline=line[0]
            elif (p1[0]+p2[0])/2 > xcenter:
                if maxRLineLen<length:
                    maxRLineLen=length
                    rightline=line[0]

    'draw line to debug image'
    if len(topline)>0:
        cv2.line(lineImg,topline[0:2],topline[2:4],255,1)
    if len(bottomline)>0:
        cv2.line(lineImg,bottomline[0:2],bottomline[2:4],255,1)
    if len(leftline)>0:
        cv2.line(lineImg,leftline[0:2],leftline[2:4],255,1)
    if len(rightline)>0:
        cv2.line(lineImg,rightline[0:2],rightline[2:4],255,1)

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
        corners=np.array([topleftPoint,toprightPoint,bottomrightPoint,bottomleftPoint])

        destination_corners = find_dest(corners)
        
        h, w = src.shape[:2]
        # Getting the homography.
        M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
        # Perspective transform using homography.
        cropedImg = cv2.warpPerspective(src, M, (destination_corners[2][0], destination_corners[2][1]),
                                        flags=cv2.INTER_LINEAR)

        
        #show debug: middle vertical,hozirontal line of boundingbox
        x,y,w,h=cv2.boundingRect(np.float32(corners))
        w2=int(w/2)
        h2=int(h/2)
        
        cv2.line(lineImg,(x,y+h2),(x+w,y+h2),(0,0,255),5) #horizontal
        cv2.line(lineImg,(x+w2,y),(x+w2,y+h),(0,0,255),5) # vertical

        cv2.circle(lineImg,topleftPoint,3,(0,0,255),3)
        cv2.circle(lineImg,toprightPoint,3,(0,0,255),3)
        cv2.circle(lineImg,bottomleftPoint,3,(0,0,255),3)
        cv2.circle(lineImg,bottomrightPoint,3,(0,0,255),3)
        
        cv2.line(lineImg,topleftPoint,toprightPoint,(0,0,255),3)
        cv2.line(lineImg,toprightPoint,bottomrightPoint,(0,0,255),3)
        cv2.line(lineImg,bottomrightPoint,bottomleftPoint,(0,0,255),3)
        cv2.line(lineImg,bottomleftPoint,topleftPoint,(0,0,255),3)
    
    return cropedImg,lineImg,(topleftPoint,toprightPoint,bottomrightPoint,bottomleftPoint)

def cardCrop2(src):
    """
    version 2: select quadrangle based on criterion of angles of line
    """
    lineImg = np.zeros(src.shape,dtype=np.uint8)
    cropedImg = np.zeros(src.shape,dtype=np.uint8)
    
    #step1: remove text
    kernel=np.ones((7,7),np.uint8)
    dilectImg=cv2.morphologyEx(src,cv2.MORPH_CLOSE,kernel,iterations=5)

    gray=cv2.cvtColor(dilectImg,cv2.COLOR_BGR2GRAY)
        
    blurImg=cv2.GaussianBlur(gray,(5,5),0)
    #step 2: detect edge and line
    sobelImg=sobel(blurImg)
    ret1,edgeImg = cv2.threshold(sobelImg,40,200, cv2.THRESH_OTSU + cv2.THRESH_TOZERO + cv2.THRESH_BINARY)

    ## make egde more dilect
    edgeImg=cv2.morphologyEx(edgeImg,cv2.MORPH_DILATE,(2,2),iterations=2)

    lines = cv2.HoughLinesP(edgeImg,rho=1,theta=1*np.pi/180,threshold=50,minLineLength=30,maxLineGap=10)

    #step3: detect quadrangle
    x=0
    y=0
    w=src.shape[1]
    h=src.shape[0]

    #assumpt: card is center of image and the skew angle is less than 45 degree
    xcenter=w/2
    ycenter=h/2

    hLine=(x,y+h/2,x+w,y+h/2)
    vLine=(x+w/2,y, x+w/2,y+h)

    hLine=np.array(hLine).astype(np.int32)
    vLine=np.array(vLine).astype(np.int32)

    topline=[]
    bottomline=[]
    leftline=[]
    rightline=[]
    topleftPoint=[]
    toprightPoint=[]
    bottomleftPoint=[]
    bottomrightPoint=[]

    toplineList=[]
    bottomlineList=[]
    leftlineList=[]
    rightlineList=[]
    selectedQuadrangleList=[]

    # grouping lines in top,bottom,left,right

    for line in lines:
        p1=line[0][0:2]
        p2=line[0][2:4]
        angle= getLineAngle(p1,p2)
        linelength=getLineLength(p1,p2)
        angle=abs(angle)
        lineExtend=np.append(line[0],[round(linelength),round(angle)])
        if(angle<40): # top/bottom line
            if(p1[1]+p2[1])/2 < ycenter:
                toplineList.append(lineExtend)
            elif (p1[1]+p2[1])/2 > ycenter:
                bottomlineList.append(lineExtend)
        else:
            if (p1[0]+p2[0])/2 < xcenter:
                leftlineList.append(lineExtend)
            elif (p1[0]+p2[0])/2 > xcenter:
                rightlineList.append(lineExtend)

    # converting list(object) to ndarray then sorting length by descending
    if(len(toplineList) and len(bottomlineList) and len(leftlineList) and len(rightlineList)):
        toplineList = np.vstack(toplineList)
        bottomlineList = np.vstack(bottomlineList)
        leftlineList= np.vstack(leftlineList)
        rightlineList = np.vstack(rightlineList)

        #get top 20% line only
        num_percent=0.2
        num_selected_line=0
        if(len(toplineList)*num_percent>1):
            num_selected_line=int(len(toplineList)*num_percent)
            toplineList = toplineList[toplineList[:,4].argsort()[::-1]][:num_selected_line]
        if(len(bottomlineList)*num_percent>1):
            num_selected_line=int(len(bottomlineList)*num_percent)
            bottomlineList = bottomlineList[bottomlineList[:,4].argsort()[::-1]][:num_selected_line]
        if(len(leftlineList)*num_percent>1):
            num_selected_line=int(len(leftlineList)*num_percent)
            leftlineList = leftlineList[leftlineList[:,4].argsort()[::-1]][:num_selected_line]
        if(len(rightlineList)*num_percent>1):
            num_selected_line=int(len(rightlineList)*num_percent)
            rightlineList = rightlineList[rightlineList[:,4].argsort()[::-1]][:num_selected_line]

        stopSearching = False
        for topline in toplineList:
            if stopSearching: 
                break
            for bottomline in bottomlineList:
                if stopSearching:
                    break
                for leftline in leftlineList:
                    if stopSearching:
                        break
                    for rightline in rightlineList:
                        criterionResult,checkResult = verifyQuadrilateral(topline,bottomline,leftline,rightline,False)
                        if(checkResult):
                            selectedQuadrangleList.append([topline,bottomline,leftline,rightline])
                            stopSearching=checkResult
                        if stopSearching:
                            break

        if(len(selectedQuadrangleList)>0):
            q =selectedQuadrangleList[0]
            topline=q[0]
            bottomline=q[1]
            leftline=q[2]
            rightline=q[3]
            cropedImg,lineImg,(topleftPoint,toprightPoint,bottomrightPoint,bottomleftPoint) = cropImage(src,topline[0:4],bottomline[0:4],leftline[0:4],rightline[0:4])
    
    return cropedImg,lineImg,(topleftPoint,toprightPoint,bottomrightPoint,bottomleftPoint)
        

if __name__ == '__main__':
    if(len(sys.argv)<2):
        print('help: cardcrop [image_path]')
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
            cropedImg,debugImg,_=cardCrop2(src)
            savedFilename=os.path.join(outpath,f)
            savedDebugFileName=os.path.join(outpath,"debug_"+f)
            cv2.imwrite(savedFilename,cropedImg)
            cv2.imwrite(savedDebugFileName,debugImg)
            print("Done")

    else:
        filename=path
        src=cv2.imread(filename)    
        cropedImg,debugImg,_dImg=cardCrop(src,debug=True)
        cv2.imshow("Croped Image",cropedImg)
        cv2.waitKey()
        cv2.destroyAllWindows()