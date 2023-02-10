import cv2
import numpy as np

def deskew(src):
    """
    Try to detect skew angle of input document image 
    Return deskew image and skew angle
    """
    des=src.copy()
    skewAngle=0.0
    angle=0.0
    gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    blurImg=cv2.GaussianBlur(gray,(5,5),0)
    edgeImg=cv2.Canny(blurImg,50,200)

    widthResize=1280;
    height=edgeImg.shape[0] #num of row
    width=edgeImg.shape[1] # num of col
    if(width>widthResize):
        ratio=0
        if(width>height):
            ratio=widthResize/width
        else:
            ratio=widthResize/height
        #recalculate wh
        width = int(width * ratio)
        height = int(height * ratio)
        edgeImg=cv2.resize(edgeImg,(width,height),interpolation=cv2.INTER_LINEAR)

    angleStep=0.3
    roh=1
    lines=cv2.HoughLinesP(edgeImg,rho= roh,theta=angleStep*np.pi/180,threshold= 150,minLineLength= width/12,maxLineGap=10)

    nb_lines=0
    if(lines.any()):
        nb_lines=len(lines)
    listAngle=[]
    mAngle={} #create new diction of angle
    if(nb_lines>0): # if find at lead one line
        for line in lines:
            lineAngle=np.arctan2(line[0][3]-line[0][1],line[0][2]-line[0][0]) 

            #convert radian to degree
            lineAngle=lineAngle*180/np.pi
            lineAngle=round(lineAngle,1)
            listAngle.append(lineAngle)
            if(lineAngle not in mAngle):
                mAngle[lineAngle]=1
            else:
                mAngle[lineAngle] +=1

        if(len(mAngle)>2):
            listAngle=sorted(listAngle,reverse=True)
            maxElement=max(mAngle, key=mAngle.get)
            
            accumulateAngle = 0
            accumuateNumberLine = 0

			#tìm điểm ở giữa
            medianindex = 0
            medianValue = 0
            
            if(len(listAngle)% 2 == 0):
                medianindex = int(len(listAngle) / 2)
                medianValue = (listAngle[medianindex - 1] + listAngle[medianindex]) / 2.0;
            else:
                medianindex = int((len(listAngle) - 1)/2)
                medianValue = listAngle[medianindex - 1];
			#nếu giá trị ở giữa = giá trị có tần số cao nhất thì kết luận
			#ngược lại thì tính giá trị trung bình quanh giá trị median
            if(medianValue == maxElement):
                angle = medianValue;
            else:
                for e in  mAngle:
                    if (e >= medianValue - angleStep and e <= medianValue + angleStep):
                        accumuateNumberLine +=mAngle.get(e)
                        accumulateAngle += e * mAngle.get(e)
                angle = round(accumulateAngle / accumuateNumberLine, 1)

    skewAngle=angle
    minAngle=-10
    maxAngle= 10
    # if(skewAngle < minAngle or skewAngle > maxAngle):
    #     skewAngle=0.0
    if(skewAngle!=0):
        matRotation =cv2.getRotationMatrix2D((int(src.shape[1]/2),int(src.shape[0]/2)), skewAngle, 1)
        des=cv2.warpAffine(src,matRotation,(src.shape[1],src.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    
    #create debugImage
    debugImage=src.copy()
    debugImage=cv2.resize(src,(edgeImg.shape[1],edgeImg.shape[0]))
    if(lines.any()):
        for line in lines:
            x1,y1,x2,y2=line[0]
            cv2.line(debugImage,(x1,y1),(x2,y2),(0,0,255))

    return des,skewAngle,debugImage

if __name__=="__main__":
    filename=r"D:\Google-drive-huy-work\imagedata\skew\train\images\img_2.jpg"
    src=cv2.imread(filename)
    des,angle,debugImage = deskew(src)
    cv2.imshow(str(angle),debugImage)
    cv2.waitKey()
    cv2.destroyAllWindows()
