tr#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 21:38:57 2017

@author: Hannah Holman (hholman - R01118537)

On my honor, I have not given, nor recieved, nor witnessed any unathorized 
assistance on this work.

I worked on this project alone, and referred to the following resources:
https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html#grabcut
https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
https://docs.opencv.org/3.3.0/dd/d49/tutorial_py_contour_features.html
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-features
https://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
"""
import numpy as np
import cv2
    
def convert_to_hsv(image):
    """This function converts an image from the RGB color space to HSV
        
        Args:
            image (numpy.ndarray): RGB image to be converted
        Returns:
            hsv (numpy.ndarray): HSV version of input image   
    """
    cpy = image.copy()
    
    hsv = cv2.cvtColor(cpy, cv2.COLOR_BGR2HSV)
        
    return hsv

def quantize_color(image):
    """ This function performs a quanization of an image, reducing color variation
    
        Args:
            image (numpy.ndarray): the image to be quantized
        Returns:
            res2 (numpy.ndarray): the quantized image
    """
    
    img = image.copy()
    
    #reshape image into imagePixelsx3 size
    img_sample = img.reshape((-1, 3))
    img_sample = np.float32(img_sample)
    
    #define critieria for quantization with type of criteria, max num of 
    #iterations, and required level of accuracy (epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    #number of clusters/colors to be represented in output image
    k = 8
    
    #apply kmeans 
    ret, label, center = cv2.kmeans(img_sample, k, None, criteria, 10, 
                                    cv2.KMEANS_RANDOM_CENTERS)
    
    #convert image back to original size and shape
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    return res2
    
def extract_foreground(image):
    """
        This function segements and then finds the contours within an image.
        These contours are bounded by a rectangle, the largest of which
        is used for the Grab Cut algorithm in order to extract the foreground 
        of the image
        
        Args:
            image (np.ndarray): image from which to extract the foreground
        Returns:
            foreground_extracted (np.ndarray): The extracted foreground from the image      
    """
    img = image.copy()
    
    #kernel for closing edges
    kernel = np.ones((5,5))
    
    #Perform color quantizization
    quantized = quantize_color(img)

    #Threshold the image to segment it
    ret, threshold = cv2.threshold(quantized, 125, 235, cv2.THRESH_BINARY)

    #De-noise image before edge detection
    blur = cv2.GaussianBlur(threshold, (11,11), 3)

    #Blur edges
    edges = cv2.Canny(blur, 50, 55, 7)
    
    #Close edges to create cohesive edge
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    #Find the external contours of the edge image
    img, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                                cv2.CHAIN_APPROX_NONE)

    #initial max values for finding largest rectangle in contours
    w_max = 0
    h_max = 0
    
    #iterate through each contour found in the image
    for c in contours:
        #find the bounding rectangles in the contours
        x,y,w,h = cv2.boundingRect(c)
        
        #Identify largest rectangle as foreground component
        if (h >= h_max and w >= w_max):
            r = (x, y, w, h)
            w_max = w
            h_max = h

    
        rect = cv2.minAreaRect(c)
        print("rect", rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
 
        #draw contours on image
        img = cv2.drawContours(img, c, -1, (255,128,0), 2)
        img = cv2.drawContours(img, [box], -1, (128, 0, 255), 2)
    
    #Copy to preserve original
    foreground_extracted = image.copy()
    
    #Create initial mask of zeros and foreground and background of zeros
    mask = np.zeros(foreground_extracted.shape[:2], np.uint8)
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    
    #Extract the area bounded by rectangle r and create mask
    cv2.grabCut(foreground_extracted, mask, r, background, foreground, 5, 
                cv2.GC_INIT_WITH_RECT)  
    
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    foreground_extracted = foreground_extracted * mask2[:,:,np.newaxis]
    
    return foreground_extracted


def convert_to_bw(image):
    """ This function converts a grayscale image to black and white.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        numpy.ndarray: The black and white image.
    """
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape
    pixel_val = 0
    
    #iterate through all rows and columns
    for i in range(rows):
        for j in range(cols):
            pixel_val = img[i][j]
            
            #convert any pixel values above 128 to 255, and any below to 0
            if (pixel_val > 1):
                img[i][j] = 255
            else:
                img[i][j] = 0
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
    return img

def equalize(image):
    img = image.copy()
    
    #iterate through each channel in the array
    for i in range (0, 2):
        #equalize histogram at each channel in an image
        img[:,:,i] = cv2.equalizeHist(img[:,:,i])
            
    return img


def main():
    foreground = cv2.imread('lyman_old.jpg', cv2.IMREAD_COLOR)
    foreground = cv2.resize(foreground, (700, 600))
    
    background = cv2.imread('lyman_new.jpg', cv2.IMREAD_COLOR)
    background = cv2.resize(background, (700, 600))
   
    #copy to preserve original image
    foreground_copy = foreground.copy()
    
    #extract foreground to generate a color mask
    color_mask = extract_foreground(foreground_copy)
    
    #convert color mask to binary mask for blending
    mask = convert_to_bw(color_mask)
    mask = cv2.resize(mask, (700, 600))
    
    #blurring the mask yields smoother edges in the composite image
    blur = cv2.GaussianBlur(mask, (11,11), 3)
    mask = np.array(blur)

    foreground = np.array(foreground)
    background = np.array(background)
    
    #Create binary mask from bw mask
    mask = mask / 255
    
    #blend mask with foreground and background
    foreground = mask * foreground
    background = (1.0 - mask) * background
    
    #add images to create composite
    out = foreground + background
    out = out.astype(np.uint8)

    #OPTIONAL ARTISTIC OPTION
    #equalizes the colors in an image if colors are drasticallu different
    #out = equalize(out)
    
    cv2.imshow("Composite Image", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
if __name__ == "__main__": main()