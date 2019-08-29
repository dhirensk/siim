# gray scale 1 channel image
import numpy as np
import pydicom
import pandas as pd
import os
import cv2


class ImagesPixelMean():
    def __init__(self, datasetdir, csvpath):
        self.csvpath = csvpath
        self.datasetdir = datasetdir

    def enhance_gamma(self,image, gamma):
        max_pixel = np.max(image)
        height,width = image.shape
        e_image = np.zeros(image.shape)
        for h in range(0,height):
            for w in range(0,width):
                e_image[h,w] = (image[h,w]/max_pixel)**gamma
        e_image = e_image * 255
        e_image = e_image.astype(np.uint8)
        #e_image = cv2.medianBlur(e_image,ksize=3)

        #e_image = cv2.erode(e_image, None, iterations=2)
        #e_image = cv2.dilate(e_image, None, iterations=2)
        return e_image
        
    def getMeanImage(self):
        annotations = pd.read_csv(self.csvpath)
        annotations.columns = ['ImageId', 'ImageEncoding']
        image_ids = annotations.iloc[:, 0].values
        rles = annotations.iloc[:, 1].values
        images = []
        total = len(annotations)
        for i, row in enumerate(annotations.itertuples()):
            id = row.ImageId
            encoding = row.ImageEncoding
            image_path = os.path.join(self.datasetdir, id + ".dcm")
            pyimage = pydicom.dcmread(image_path)
            height = pyimage.Rows
            width = pyimage.Columns
            image = pyimage.pixel_array   ## pixel array is shape (2,2)
            enhanced_hist = cv2.equalizeHist(image)
            remove_noise = cv2.fastNlMeansDenoising(image,None,3,7,21)
            blur = cv2.GaussianBlur(remove_noise,(3,3),0)
            sharpen = cv2.addWeighted(enhanced_hist,1.0,blur,-1.0,0)
            enhanced_pixel = cv2.addWeighted(enhanced_hist,1.0, sharpen,0.7,0)           
            images.append(enhanced_pixel)      # creates (n, 1024,1024)
            print("image {} of {}".format(i, total))

        images = np.array(images)
        meanimage = np.mean(images, axis=0)   #take mean across axis = 0 returns ( 1024,1024)
        pixelmean = np.mean(meanimage)
        return pixelmean, meanimage


if __name__ == '__main__':

    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Calculate pixel mean for  training set')

    parser.add_argument('--datasetdir', required=True,
                        metavar="/path/to/dataset/",
                        help='Directory of the dcm files')
    parser.add_argument('--csvpath', required=True,
                        metavar="/path/to/csv file",
                        help="Path to csv file")

    args = parser.parse_args()

    # Validate arguments

    assert args.datasetdir, "Argument --dataset is required"
    assert args.csvpath , "Argument --csvpath is required"

    imagepixels = ImagesPixelMean(args.datasetdir, args.csvpath)
    pixel_mean , mean_image = imagepixels.getMeanImage()
    print(pixel_mean)
    print(mean_image.shape)





