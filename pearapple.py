import numpy as np
import argparse
import cv2
import pickle


# Create parser for command line arguments
parser = argparse.ArgumentParser(description='Tell whether picture is that of an apple or a pear.')
parser.add_argument("path", help="Path to image.")
parser.add_argument("-l", "--leaf", help="Use if there is a leaf in the picture.", action="store_true", default=False)
args = parser.parse_args()


# Load trained logistic regression model
pkl_filename = "train_model.pkl"
with open(pkl_filename, 'rb') as file:
    train_model = pickle.load(file)


# predict(x) uses the loaded trained model to predict the label of the input x,
#   returning True if x corresponds to an apple and False if it corresponds to a
#   pear
# Effects: Calls the .predict() method of object train_model
# predict: float -> Bool
# Examples:
#   predict(1) => True
#   predict(2) => False
def predict(x):
    x = np.array([x])
    value = train_model.predict(x.reshape(-1,1))
    if value == 1:
       return True
    else:
        return False


# preProcess(img) tries to remove shadows from grayscale image img
# Effects: Applies a dilation morphological operator and
#   a median blur to img
# preProcess: (array of Int) -> (array of Int)
def preProcess(img):
    dilated_img = cv2.dilate(img, np.ones((5,5), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    norm_img = diff_img.copy() # Needed for 3.x compatibility
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return norm_img


# main() is a method for recognizing whether a picture is that of an apple
#      or that of a pear
# Effects: Reads file 'path' passed from command line. If "--leaf" flag is one, create mask to occult leaf.
#      Prints "This is probably an apple" or "This is probably a pear" and shows cutout version of fruit with a
#      bounding box.
# main: None -> None
# Examples:
#   For args.path = Test_Examples/apple0.jpg => Shows picture, prints "This is probably an
#       apple"
#   For args.path = Test_Examples/pear1.jpg, args.leaf=True => Shows picture, prints "This is
#       probably a pear"
def main():
    # load image data
    img = cv2.imread(args.path, cv2.IMREAD_COLOR)

    # Check leaf flag. If true, try to get a mask for leaf in img. First
    # simplify img with k means clustering.
    if args.leaf:
        it = 15 # aggressive no. of iterations for erode morph. trans. later on
        # k means clustering
        Z = img.reshape((-1,3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 11
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))

        # get red and green channels and smooth them out
        red = res2[:, :, 2]
        green = res2[:, :, 1]
        red = cv2.medianBlur(red,5)
        green = cv2.medianBlur(green,5)

        # get a mask by threshholding red and green values
        red_max = np.max(red)
        red_min = np.min(red)
        green_max = np.max(green)
        green_min = np.min(green)
        red_range = red_max - red_min
        green_range = green_max - green_min
        cutoff_red = red_min + (0.4*red_range)
        cutoff_green = green_max - (0.75*green_range)
        leaf_mask = np.where((red < cutoff_red) & (green > cutoff_green), 255 , 0)
        leaf_mask = leaf_mask.astype(np.uint8)
    else:
        leaf_mask = np.zeros(img.shape[:2], np.uint8) # no mask for leaf
        it = 6 # less aggressive no. of iterations for erode morph. trans.

    # Preprocess img
    pimg = preProcess(img) # get rid of shadows in img
    gray = cv2.cvtColor(pimg, cv2.COLOR_BGR2GRAY) # work with grayscale version

    # detect edges, dilate then find floodfill to get cutout of shape
    mask = np.zeros(img.shape[:2], np.uint8)
    mask2 = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
    _, image_thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(gray, 20, 50)
    kernel = (5, 5)
    dilation = cv2.dilate(edges, kernel, iterations = 11)
    floodfill = dilation.copy()
    cv2.floodFill(floodfill, mask2, (0,0), 255)
    floodfill = cv2.bitwise_not(floodfill)
    cutout = (dilation | floodfill) & ~leaf_mask
    cutout =  cv2.erode(cutout, (5,5), iterations=it)
    contours, _ = cv2.findContours(cutout, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, [max(contours, key = cv2.contourArea)], -1, 255, thickness=-1)

    # find bounding rect. and return prediction of apple/pear based on the
    # ratio of its sides
    rect = cv2.minAreaRect(max(contours, key = cv2.contourArea))
    ratio = rect[1][0] / rect[1][1]
    if ratio < 1: # always return ratio of larger to smaller side
        ratio = 1/ratio
    tvalue = predict(ratio)
    if tvalue==True:
        print("This is probably an apple")
    if tvalue==False:
        print("This is probably a pear")


    # draw cutout with bounding rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,(0, 0, 255),2)
    cv2.imshow("Bounding rectangle", img)
    cv2.waitKey(0)
    return


main()
