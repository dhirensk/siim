
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import local_binary_pattern


class LungSegmentation():
    
    gamma = 2
    kernel1 = np.ones((3,3),np.uint8)
    # Elliptical Kernel
    
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kernel3 = np.array([[0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [1, 1, 1, 1, 1],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0]]).astype(np.uint8)
    
    def enhance_gamma(self,image, gamma):
    
        max_pixel = np.max(image)
        #print(image.shape)
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
    
    
    def display(self,pixel):
    
        fig, ax = plt.subplots(1, figsize=(20,20))
        #str = "/home/sa-279/Mask_RCNN/datasets/pneumothorax/val/1.2.276.0.7230010.3.1.4.8323329.10557.1517875224.257683.dcm"
    
        pixel = pixel
        #display_enhanced_gamma(pixel)
        #im = Image.fromarray(pixel)
        #im.save(os.path.join(ROOT_DIR,"siim\\1.2.276.0.7230010.3.1.4.8323329.1314.1517875167.222290.png"))
        ax.imshow(pixel, cmap="gray")
        #ax.imshow(rle_m.T, alpha = 0.2)
        plt.show()
    
    def otsu_mask(self,image):
    
        #print("here1")
        # Otsu's thresholding after Gaussian filtering
        #blur1 = cv2.GaussianBlur(image, (5, 5), 0)
        #image = cv2.equalizeHist(image)
        #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        #image = clahe.apply(image)
        #th3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        ret, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #ret, th = cv2.threshold(image, 100,255, cv2.THRESH_BINARY)
        th = cv2.bitwise_not(th)
        #print("here2")
        #im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return ret, th
    
    def connectedcomponents(self,image, connectivity):
        # The first cell is the number of labels
        # The second cell is the label matrix
        # The third cell is the stat matrix
        # The fourth cell is the centroid matrix
    
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)    
        centroids = np.round(centroids).astype(np.int)
        height, width = image.shape
    
    
        sizes = np.copy(stats[:, -1])
        #initialize to first centroid as having max area, find next 2 major areas should correspond to lung region
        max_size0 = 0
        max_label0 = 0
        max_size1 = 0
        max_size2 = 0
        max_label1 = 0
        max_label2 = 0
        max_width1 = 0
        max_width2 = 0
        # retain top two sizes after removing first size
        max_label0 = np.argmax(sizes)
        max_size0 = sizes[max_label0]
        # remove the maxsize 1
        sizes[max_label0] = 0
        max_label1 = np.argmax(sizes)
        max_size1 = sizes[max_label1]
        max_width1 = stats[max_label1, 2]
    
        sizes[max_label1] = 0
        max_label2 = np.argmax(sizes)
        max_size2 = sizes[max_label2]
        max_width2 = stats[max_label2, 2]
    
        filled_image = np.zeros((height, width))
        # Assuming centroid0 is unwanted region and occupying most size. then centroid 2 should be atleast over 60% of centroid 1.
        # if not then the image is not bimodal with 2 peaks, its monomodal and otsu thresholding is not correct to apply here
        # so include all 3 centroids or entire image
        lungs_centroids = np.array([max_label1, max_label2])
        skip_otsu = True
        if (max_width1) < 650 and (max_width2<650) and (max_width1 > 200) and (max_width2 > 200):           
            skip_otsu = False
    
        if skip_otsu == False:
            for h in range(height):
                for w in range(width):
                    if labels[h,w] in lungs_centroids:
                        filled_image[h,w] = 1
        else:
            filled_image = np.ones((height, width))
        filled_image = filled_image.astype(np.uint8)
        return filled_image, num_labels, labels, stats, centroids
    
    def smoothimage(self,image):
         #image = cv2.erode(image, None, iterations=2)
         image = cv2.dilate(image, self.kernel3, iterations=10)
         return image
    
    def lbp(self,image):
    
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(image, n_points, radius, method='default' )
        height,width = image.shape
        lbp_image = np.zeros((height,width))
        K = np.array([0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112,
                      120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225,
                      227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254,255])
        K = np.expand_dims(K,axis=1)
        K = np.append(K, np.zeros((len(K),1)), axis=1).astype(np.int)
        N = np.copy(K)
        for r, k in enumerate(K[:, 0]):
            for h in range(height):
                for w in range(width):
                    if lbp[h,w] == k:
                        N[r,1] += 1
                        lbp_image[h,w] = 1
        e_image = np.multiply(image,lbp_image)
        e_image = e_image.astype(np.uint8)
        return N, e_image
    
    
    
    
    def applylungmask(self,image, mask):
        return (np.multiply(image,mask))
    
    
    def hist(ax, lbp):
        n_bins = int(lbp.max() + 1)
        return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                       facecolor='0.5')
    
    def getSegmentedImage(self, image, gamma):
        enhanced_image = self.enhance_gamma(image, gamma=self.gamma)
        enhanced_image1 = cv2.equalizeHist(enhanced_image)
        #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        #enhanced_image = clahe.apply(enhanced_image)
        #enhanced_image = cv2.morphologyEx(enhanced_image, cv2.MORPH_CLOSE, kernel1)  #this is causing merging of unwanted regions
       
        ret, otsu_image = self.otsu_mask(enhanced_image1)
        
        #display(otsu_image)
        filled_image, num_labels, labels, stats, centroids = self.connectedcomponents(otsu_image,8)
        smooth_image = self.smoothimage(filled_image)
        lung_image = self.applylungmask(image, smooth_image)
        return lung_image
    #display(lung_image)
    
