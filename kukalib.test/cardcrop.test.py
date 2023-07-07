import sys
import glob
from pathlib import Path

# adding root folder of kuka-lib to the system path
sys.path.insert(0, '../kuka-py')

from kukalib.cardcrop import *

import cv2
import numpy as np
import json


def getquardLabel(labelFile):
    """
    return (tl,tr,br,bl)
    """
    f = open(labelFile)
    label = json.load(f)
    if('quad' in label):
        (tl,tr,br,bl)=label.get("quad")
        return (tl,tr,br,bl)
    else:
        return (None,None,None,None)

def getBoxPoint(listPoint):
    listPoint=np.array(listPoint,dtype=np.float32)
    x1,y1,w,h=cv2.boundingRect(listPoint)
    return (x1,y1,x1+w,y1+h)

def get_iou(ground_truth, pred):
    """
        return metric IoU
    """
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])
     
    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
     
    area_of_intersection = i_height * i_width
     
    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
     
    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
     
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
     
    iou = area_of_intersection / area_of_union
     
    return iou

def eval2Point(p1,p2):
    maxDis=30 # max distance from p1 to p2
    confident=0.0
    l=getLineLength(p1,p2)
    if(l<=maxDis):
        confident=((maxDis-l)/maxDis)
    return confident

def get_confidence_2quard(quadr_true,quadr_pred):
    (tl_pred,tr_pred,br_pred,bl_pred)=quadr_pred
    (tl,tr,br,bl)=quadr_true
    tl_c=(eval2Point(tl_pred,tl))
    tr_c=(eval2Point(tr_pred,tr))
    br_c=(eval2Point(br_pred,br))
    bl_c=(eval2Point(bl_pred,bl))

    mean_c=(tl_c+tr_c+br_c+bl_c)/4
    return mean_c

def main(rootPath,savepath,debug=False):
    #set test function to cardCrop2
    testfunction=cardCrop2

    imagePath=os.path.join(rootPath,"images")
    labelPath=os.path.join(rootPath,"ground_truth")

    print("File","\t","Confidence","\t","Note")
    for f in glob.glob(os.path.join(imagePath,"**"),recursive=True):
        if(os.path.isdir(f) or ("out" in f) ):
            continue

        labelFile=os.path.splitext(f)[0] + ".json"
        labelFile=labelFile.replace("images","ground_truth")
       
        src=cv2.imread(f)
        if(type(src)!=np.ndarray):
                continue
        #get predict value
        cropedImg,debugImg,(tl_pred,tr_pred,br_pred,bl_pred)=testfunction(src)
        if(debug):
            outpath= savepath
            if(not os.path.exists(outpath)):
                os.mkdir(outpath)
            debugName="debug_" + os.path.basename(f)
            debugName=os.path.join(outpath,debugName)
            cv2.imwrite(debugName,debugImg)

        if(len(tl_pred)==0 or len(tr_pred)==0 or len(br_pred)==0 or len(bl_pred)==0):
            print(f,"\t",0,"\t","Fail")
            continue

        
        if(os.path.exists(labelFile)):
            #get ground_truth value
            (tl,tr,br,bl)=getquardLabel(labelFile)
            if( (tl,tr,br,bl)!=(None,None,None,None)):
                #calculate metric IoU
                confidence=get_confidence_2quard((tl,tr,br,bl),(tl_pred,tr_pred,br_pred,bl_pred))
                print(f,"\t", confidence)  



if __name__ == '__main__':
    import argparse
    import shutil
    parser=argparse.ArgumentParser()
    parser.add_argument('-p','--path',help='Path to root folder that containt 2 sub image /Images and /Ground_Truth')           # positional argument
    parser.add_argument('-d', '--debug',default=False,type=bool,help="write debug image to output file") 
    parser.add_argument('-sp', '--savepath',help="path to save output") 

    args = parser.parse_args()
    rootPath=args.path
    debug=args.debug
    savepath=args.savepath
    
    main(rootPath,savepath,debug)