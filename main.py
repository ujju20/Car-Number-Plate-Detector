import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def order_points(pts):
    pts=np.array(pts)
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(gr, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    maxWidth+=(maxWidth*12)//10
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(gr, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def asli_cnt(contours,img_area):
    for cnt in contours:
        #print(cv2.contourArea(cnt),img_area)
        if cv2.contourArea(cnt)>((img_area)//10):
            epsilon = 0.025*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            #print(len(approx))
            if(len(approx)>4 and len(approx)<7):
                diff=0.001
                epsilon = (0.025+diff)*cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,epsilon,True)
                while(len(approx)!=4):
                    diff=diff+0.001
                    if(diff>1.5):
                        break
                    epsilon = (0.025+diff)*cv2.arcLength(cnt,True)
                    approx = cv2.approxPolyDP(cnt,epsilon,True)
            if(len(approx)==3):
                diff=0.001
                epsilon = (0.025-diff)*cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,epsilon,True)
                while(len(approx)!=4):
                    if diff>0.025:
                        break
                    diff=diff+0.001
                    epsilon = (0.025-diff)*cv2.arcLength(cnt,True)
                    approx = cv2.approxPolyDP(cnt,epsilon,True)        
                
            if len(approx) == 4:
                screenCnt = approx
                return approx
    return None

def seedha_kar(image):
    image=cv2.copyMakeBorder(image,((image.shape[0]*2)//10),  ((image.shape[0]*2)//10),  ((image.shape[1]*2)//10), ((image.shape[1]*2)//10), cv2.BORDER_CONSTANT, None,255)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Blackhat , darkcharcter over light
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
    
    # light charchter over light
    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    #determining lines, through sobel filter
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")
    
    #saaf kar rhe hai, noise hata ke
    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
    thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    #aur saaf safayi
    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
    
    contours=cnts
    img_area=(thresh.shape[0])*(thresh.shape[1])
    screenCnt=asli_cnt(contours,img_area)
    
    if(screenCnt is None):
        print(2222222)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(gray, 75, 200)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
        screenCnt=asli_cnt(cnts,img_area)
        
    if(screenCnt is None):
        print("Sab Moh Maya Hai")
        return None
    image=four_point_transform(image, list([screenCnt[0][0],screenCnt[1][0],screenCnt[2][0],screenCnt[3][0]]))
    return image

def segment(image):

    chars=[]    
    H = 60.
    #print(image.shape)
    height, width,depth = image.shape
    imgScale = H/height
    newX,newY = image.shape[1]*imgScale, image.shape[0]*imgScale
    image = cv2.resize(image,(int(newX),int(newY)),interpolation = cv2.INTER_NEAREST)
    #print(newX,newY)
    #cv2.imshow("Show by CV2",image)
    #cv2.imwrite("resizeimg.jpg",image)
 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 105, 15)

    cv2.imshow("thresh",thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #checking for padding
    hor=(thresh.shape[1]*2)//10
    ver=(thresh.shape[0]*2)//10
    a=np.sum(thresh[:,:hor])
    b=np.sum(thresh[:,(4*hor):])
    c=np.sum(thresh[:,(2*hor):(3*hor+1)])
    #print(a,b,c)
    if(c>(a*1.4)):
        left=0
    else:
        left=hor//2
    if(c>(b*1.4)):
        right=0
    else:
        right=hor//2 
        
    a=np.sum(thresh[:ver,:])
    b=np.sum(thresh[(4*ver):,:])
    c=np.sum(thresh[(2*ver):(3*ver+1),:])
    #print(a,b,c)
    if(c>(a*1.4)):
        top=0
    else:
        top=ver*2
    if(c>(b*1.4)):
        bottom=0
    else:
        bottom=ver*2    
        
    thresh=cv2.copyMakeBorder(thresh,top,bottom,left,right, cv2.BORDER_CONSTANT, None,0)
    newX=thresh.shape[1]
    newY=thresh.shape[0]
    cv2.imshow("thresh",thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    boundingBoxes = sorted(boundingBoxes,key=lambda b:b[0], reverse=False)
    wid=[]
    for cnt in boundingBoxes:
        x,y,w,h =  cnt
        wid.append(w)
    
    
    wid=sorted(wid,reverse=True)
    st=0
    idx=1
    #print(wid)
    #print(((newX*8)//10))
    for i in range(1,len(wid)):
        if(wid[i-1]>(wid[i]*2.5)):
            break
        #print(idx)    
        idx=idx+1
        if(idx==10):
            break
    #print(st,idx)
    xx=[]
    ww=[]
    for cnt in boundingBoxes:
        #print(cnt)
        x,y,w,h =  cnt
        if(w>((newX)//40) and h>((newY*2)//10) and h<((newY*8)//10) and w<((newX*8)//10)):
            if len(xx)==0:
                xx.append(x)
                ww.append(w)
            elif ((xx[-1]<x)and((xx[-1]+ww[-1])>(x+w))):
                continue
            else:
                xx.append(x)
                ww.append(w)
            roi = thresh[y:y+h,x:x+w]
        
            #print("hehe")
            #print(x,y,w,h)
            roi=cv2.copyMakeBorder(roi,8,  8,  15, 15, cv2.BORDER_CONSTANT, None,0)
            roi = cv2.resize(roi, (28,28),interpolation = cv2.INTER_NEAREST)
            chars.append(roi)
    return chars

def PlateReconginization(img):
    image=img
    cv2.imshow("initial",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    asli=seedha_kar(image)
    cv2.imshow("correct",asli)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    chars=segment(asli)
    for i in chars:
        cv2.imshow("charcter segmented",i)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    final_char=[]
    for char in chars:
        final_char.append(char/255.0)

    model = keras.models.load_model("Model_plate.h5")
    final_char=np.array(final_char)
    final_char=np.reshape(final_char,(-1,28,28,1))
    ans=model.predict(final_char)

    mapping=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    final_ans=np.argmax(ans,axis = 1)
    ans=[]
    for i in final_ans:
        ans.append(mapping[i])
    print(ans)

def bonus(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    edge = cv2.Canny(smooth, 70, 400)
    contours, new = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = image.copy()
    _ = cv2.drawContours(image_copy, contours, -1, (255, 0, 0), 2)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    image_reduced = edge.copy()
    _ = cv2.drawContours(image_reduced, contours, -1, (255, 0, 0), 2)
    for i in contours:
        a = cv2.arcLength(i, True)
        edge_count = cv2.approxPolyDP(i, 0.02 * a, True)
        if len(edge_count) >=0:
            x, y, w, h = cv2.boundingRect(i)
            plate = image[y:y+h, x:x+w]
            #print(len(edge_count))
            break
    PlateReconginization(plate)    

images=['Chevrolet-Beat-525743c.png']
for image in images:
    PlateReconginization(cv2.imread(image))

    
