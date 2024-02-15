import cv2
import numpy as np
import os

min_area = 6000

def segment(input_image):
    blurred = cv2.GaussianBlur(input_image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 155, 23)
    image = thresh.copy()
    image = cv2.bitwise_not(image)
    image = cv2.resize(image, (800,500))
    
    kernel = np.ones((1,15),np.uint8)
    erosion = cv2.erode(image,kernel,iterations = 1)
    kernel = np.ones((9,30),np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations = 1)
    canny = cv2.Canny(dilation, 200, 255, 5)
    
    cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(cnts, key=cv2.contourArea)
    
    if cv2.contourArea(largest)>4000:
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [largest], 0, (255,255,255), thickness=cv2.FILLED)
        mask = 255-mask
        new_img = cv2.bitwise_and(image, mask)
    else:
        new_img = image

    kernel = np.ones((8,8),np.uint8)
    dilation = cv2.dilate(new_img, kernel, iterations = 1)
    blurred = cv2.GaussianBlur(dilation, (13,13), 0)
    canny = cv2.Canny(blurred, 200, 255, 5)

    contourss, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contourss = sorted(contourss, key=lambda c: cv2.boundingRect(c)[0])
    contours = [ctr for ctr in contourss if cv2.contourArea(ctr) >= 2000]

    contour_combination = []
    pair = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            contour_combination.append(c)

    for c in contour_combination:
        pair = [[contour_combination[i], contour_combination[i+1]] for i in range(0, len(contour_combination)-1, 2)]

    for p in pair:
        x1,y1,w1,h1 = cv2.boundingRect(p[0])
        x2,y2,w2,h2 = cv2.boundingRect(p[1])

    if not os.path.exists('./model/temp'):
        os.makedirs('./model/temp')
    
    image_number = 0

    binarized_image = image.copy()
    kernel = np.ones((6,6),np.uint8)
    dilation = cv2.dilate(binarized_image, kernel, iterations = 1)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(dilation,kernel,iterations = 1)
    blur = cv2.GaussianBlur(erosion, (5,5), 0)
    binarized_image = cv2.bitwise_not(blur)
    

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > min_area:
            x,y,w,h = cv2.boundingRect(contours[i])
            ROI = binarized_image[y-20:y+h, x:x+w]
            ###
            if(h>w):
                adjusted_height = h+20
                square_img = np.zeros((adjusted_height,adjusted_height), np.uint8)
                square_img[:,:] = (255)
                x_offset = int((adjusted_height-w)/2)
                square_img[:,x_offset:x_offset+w] = ROI.copy()
                
            cv2.imwrite("./model/temp/ROI_{}.png".format(image_number), square_img if h>w else ROI)
            image_number += 1
            ###
        else:
            for p in pair:
                if cv2.contourArea(p[0]) == area:
                    x1,y1,w1,h1 = cv2.boundingRect(p[0])
                    x2,y2,w2,h2 = cv2.boundingRect(p[1])
                    ROI = binarized_image[y1-20:y2+h2, x1:x2+w2]
                    ###
                    if ROI.shape[0] > ROI.shape[1]:
                        adjusted_height = ROI.shape[0]
                        square_img = np.zeros((adjusted_height,adjusted_height), np.uint8)
                        square_img[:,:] = (255)
                        x_offset = int((adjusted_height-ROI.shape[1])/2) 
                        square_img[:,x_offset:x_offset+ROI.shape[1]] = ROI.copy()
                    ###
                    cv2.imwrite("./model/temp/ROI_{}.png".format(image_number), square_img if ROI.shape[0] > ROI.shape[1] else ROI)
                    image_number += 1