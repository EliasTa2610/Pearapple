# Pearapple
Short program to try to tell whether picture is that of an apple or that of a pear.

# Introduction
I was playing around with the Open CV and Scikit-learn libraries. I decided a fun little project would be to write a program that recognize whether a picture is that of an apple or that of a pear based on geometric features of the two fruits. In the end, looking at the ratio of the sides of a minimum area bounding rectangle turned out to give a somewhat reliable classifying scheme. This ratio is close to 1 and close to 1.3 for apples and pears respectively. I used a training set (stored as data.npy) on a logistic regression model (pickled in train_model.pkl).

# Usage
pearapple.py [-h] [-l] path_to_img

positional arguments:
  path_to_img  Path to image.

optional arguments:
  -h, --help   show this help message and exit
  -l, --leaf   Use if there is a leaf in the picture.

# Remarks
* The algorithm employed uses a number of standard procedures from computer vision and image processing, namely canny edge detection, k means clustering, morphological operators, contour detection etc.

* The algorithm does not work if the fruit in the question is flush against the frame of the picture. This will be rectified in a future version.
