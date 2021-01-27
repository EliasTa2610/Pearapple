# Pearapple
Short program to try to tell whether picture is that of an apple or that of a pear.

# Introduction
I was playing around with the OpenCV and Scikit-learn libraries. I decided a fun little project would be to write a program that guesses whether a picture shown to it is that of an apple or that of a pear based on geometric features of the two fruits. In the end, looking at the ratio of the sides of a minimum area bounding rectangle turned out to give a somewhat reliable classifying scheme. This ratio is close to 1 and close to 1.3 for apples and pears respectively. A training set (stored as data.npy) was used on a logistic regression model (pickled in train_model.pkl) in order to generate the predictions.

# Usage
pearapple.py [-h] [-l] path_to_img

positional arguments:
  path_to_img  Path to image.

optional arguments:
  -h, --help   show this help message and exit
  -l, --leaf   Use if there is a leaf in the picture.

# Output/Effect
Prints "This is probably an apple" or "This is probably a pear" depending on the prediction. Shows a picture showing the bounding rectangle used in the algorithm superposed onto the original picture.

# Examples
Doing
```python3
pearapple.py Test_Examples/apple0.jpg
```
will print "This is probably an apple" and produce the image

![alt text](https://github.com/EliasTa2610/Pearapple/blob/main/result_ex1.jpg?raw=true)


On the other hand, doing
```python3
pearapple.py Test_Examples/pear1.jpg -l
```
will print "This is probably a pear" and produce the image

![alt text](https://github.com/EliasTa2610/Pearapple/blob/main/result_ex0.jpg?raw=true)

Note the -l flag: the leaf in the picture is accounted for and the bounding rectangle ignores it

# Technical
* Written for Python 3

* Uses a number of standard procedures from computer vision and image processing provided by the OpenCV library, namely canny edge detection, k means clustering, morphological operators, contour detection etc. These techniques are marshalled in order to obtain a cutout mask of the fruit without the background details, and from there a bounding rectangle from which whether the fruit is an apple or pear can be guessed.

* The learning step is done via logistic regression model, using the facilities provided by the Scikit-learn library.

* The various constants, kernel sizes etc. found in the code were chosen for best performance empirically after trying many test inputs.

# Remarks
* The algorithm is meant to process pictures of individual fruits and will therefore not give meaningful results for pictures containing several of them. The background of the picture is assumed to be more or less uniform. Furthermore, the algorithm does not work if the fruit in the question is flush against the frame of the picture. This will be rectified in a future version. 
