
import numpy as np
import pydicom
from mask_functions import mask2rle, rle2mask
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

import cython


ROOT_DIR = os.getcwd()
print(ROOT_DIR)
datadir = os.path.join(ROOT_DIR,'siim\\dataset\\train')
csvfile = os.path.join(datadir,'train-rle.csv')
imagepath = os.path.join(datadir, '1.2.276.0.7230010.3.1.4.8323329.1314.1517875167.222290.dcm')
df = pd.read_csv(csvfile,header = None)
#df.columns['ImageId','Encodings']
dcm = pydicom.dcmread(imagepath)
pixel = dcm.pixel_array
samples = df.iloc[:, -1].values
rle_m = rle2mask(samples[6], 1024, 1024)
#annotation = df.loc[df['ImageId'] =='1.2.276.0.7230010.3.1.4.8323329.1314.1517875167.222290']
gamma = 2
def enhance_gamma(image):

    max_pixel = np.max(image)
    print(image.shape)
    height,width = image.shape
    e_image = np.zeros(image.shape)
    for h in range(0,height):
        for w in range(0,width):
            e_image[h,w] = (image[h,w]/max_pixel)**gamma
    e_image = e_image * 255
    e_image = e_image.astype(np.uint8)
    return e_image


def display(pixel):
    
    fig, ax = plt.subplots(1, figsize=(20,20))
    #str = "/home/sa-279/Mask_RCNN/datasets/pneumothorax/val/1.2.276.0.7230010.3.1.4.8323329.10557.1517875224.257683.dcm"

    pixel = pixel
    #display_enhanced_gamma(pixel)
    im = Image.fromarray(pixel)
    #im.save(os.path.join(ROOT_DIR,"siim\\1.2.276.0.7230010.3.1.4.8323329.1314.1517875167.222290.png"))
    ax.imshow(pixel, cmap="gray")
    #ax.imshow(rle_m.T, alpha = 0.2)
    plt.show()

def fillholes(image):

    print("here1")
    #ret, th = cv2.threshold(image, 75, 155, 0)
    # Otsu's thresholding after Gaussian filtering
    #blur = cv2.GaussianBlur(image, (5, 5), 0)
    ret, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #print("here2")
    #im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return th

def display_enhanced_gamma(pixel):

    pixel = pixel
    e_image = enhance_gamma(pixel)
    #im = Image.fromarray(e_image)
    #im.save(os.path.join(ROOT_DIR,"siim\\1.2.276.0.7230010.3.1.4.8323329.1314.1517875167.222290_e.png"))
    return e_image

enhanced_image = display_enhanced_gamma(pixel)
#display(enhanced_image)
filled_image = fillholes(pixel)
display(filled_image)