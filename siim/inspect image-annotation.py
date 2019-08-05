
import numpy as np
import pydicom
from mask_functions import mask2rle, rle2mask
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os

import cython


ROOT_DIR = os.getcwd()
print(ROOT_DIR)
datadir = os.path.join(ROOT_DIR,'siim\\dataset\\train')
csvfile = os.path.join(datadir,'train-rle.csv')
image = os.path.join(datadir,'1.2.276.0.7230010.3.1.4.8323329.1314.1517875167.222290.dcm')
df = pd.read_csv(csvfile,header = None)
#df.columns['ImageId','Encodings']

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
    return e_image



def display():   
    
    fig, ax = plt.subplots(1, figsize=(20,20))
    #str = "/home/sa-279/Mask_RCNN/datasets/pneumothorax/val/1.2.276.0.7230010.3.1.4.8323329.10557.1517875224.257683.dcm"
    samples = df.iloc[:,-1].values
    dcm = pydicom.dcmread(image)
    pixel = dcm.pixel_array
    display_enhanced_gamma(pixel)
    im = Image.fromarray(pixel)
    im.save(os.path.join(ROOT_DIR,"siim\\1.2.276.0.7230010.3.1.4.8323329.1314.1517875167.222290.png"))
    ax.imshow(pixel, cmap="gray")
    rle_m = rle2mask(samples[6], 1024, 1024)
    ax.imshow(rle_m.T, alpha = 0.2)      
    plt.show()


def display_enhanced_gamma():

    #samples = df.iloc[:,-1].values
    dcm = pydicom.dcmread(image)
    pixel = dcm.pixel_array
    e_image = enhance_gamma(pixel)
    fig, ax = plt.subplots(1, figsize=(20,20))
    ax.imshow(e_image, cmap="gray")
    plt.show()
    #im = Image.fromarray(e_image)
    #im.save(os.path.join(ROOT_DIR,"siim\\1.2.276.0.7230010.3.1.4.8323329.1314.1517875167.222290_e.png"))


display_enhanced_gamma()