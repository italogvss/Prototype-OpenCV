# import the necessary packages
from imutils import paths
import numpy as np
from cv2 import cv2
import rasterio


def quantify_image(image, bins=(4, 6, 3)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute a 3D color histogram over the image and normalize it
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    # return the histogram
    return hist

def load_dataset(datasetPath, bins):
    # grab the paths to all images in our dataset directory, then
    # initialize our lists of images
    imagePaths = list(paths.list_images(datasetPath))
    data = []
    # loop over the image paths
    for imagePath in imagePaths:
        # load the image and convert it to the HSV color space
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # quantify the image and update the data list
        features = quantify_image(image, bins)
        data.append(features)
    # return our data list as a NumPy array
    return np.array(data)

def isAnomaly(image, path):

    rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    lower = np.array([60, 73, 70])
    upper = np.array([255, 255, 255])

    mask = cv2.GaussianBlur(cv2.inRange(rgb, lower, upper),(7,7),0)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, (0,0,255), 2)
    cv2.imshow("Imagem", image)
    cv2.waitKey(1)
    return image

def writeMeta(src, dst):
    
    with rasterio.open(src) as data:
        out_meta = data.meta.copy()
        out_transform =  data.transform
        out_height = data.height
        out_width = data.width
        crs = data.crs
        out_meta.update({"driver":"GTiff",
                    "height": out_height,
                     "weight": out_width,
                    "transform": out_transform,
                    "crs" : data.crs })
        with rasterio.open(dst, 'w', ) as dest: 
            dest.meta = out_meta


