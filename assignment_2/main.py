import cv2
import numpy as np
import matplotlib.pyplot as plt

def padding(image, border_width): # Task 1

    reflect = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)

    # Comparing the data between reflect and shape
    #print(image.shape)
    #print(reflect.shape)

    print(cv2.imshow('reflect', reflect))
    print("padding")
    cv2.waitKey()
    cv2.destroyAllWindows()

def crop(image, x_0, x_1, y_0, y_1): # Task 2

    # Need to think about how the cropping function works
    """
    Using amount of pixels from each direction to crop picture

    cropped_img = image[y_start:y_end, x_start:x_end]
    I have to think of the values in image as where the image will be included

    A little bit like y_start:y_end decides from the top to the bottom, and
    x_start:x_end decides from the left to the right.

    Since the picture decides how far to one of the sides we're going, we have to
    use the height and the width to change the y_end and x_end (or in our case, x_1
    and y_1.
    """

    # Taking the values from the matrix, and cropping away unwanted values
    height, width = image.shape[:2]
    cropped_image = image[y_0:height - y_1, x_0:width - x_1]

    # Printing data and picture to verify results
    print(cropped_image.shape)
    print(cv2.imshow('cropped_image', cropped_image))
    print("crop")
    cv2.waitKey()
    cv2.destroyAllWindows()

def resize(image, width, height): # Task 3
    """
    Understanding how to resize the picture

    Seems like I just have to use the function mentioned below
    cv2.resize(src. dsize[, dst[. fx[, fy[, interpolation]]]])

    Since we know that the size should be just 200x200, there should be no problems
    and just use cv2.resize(image, dsize[height, width])

    Also chose to use INTER_AREA, for minimal distortion
    """

    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Printing data and picture to verify results
    print(resized_image.shape)
    print(cv2.imshow('resized_image', resized_image))
    print("resize")
    cv2.waitKey()
    cv2.destroyAllWindows()

def copy(image, emptyPictureArray): # Task 4
    """
    How should I understand this?
    "Manual Copy - emptyPictureArray np.zeros((height, width, 3), dtype=np.uint8)"

    Since we're trying to just copy the values from one picture into the other,
    we'll most likely just copy the values from one picture into the other one through
    matrices.

    Hence, we'll probably just use several for loops for handling the y-axis and x-axis.
    Basically going through one selected y value, and inserting all of the x values from
    there.
    """

    height, width = image.shape[:2]

    for i in range(height):
        for j in range(width):
            emptyPictureArray[i, j] = image[i, j]

    print(emptyPictureArray.shape)
    print(cv2.imshow('emptyPictureArray', emptyPictureArray))
    print("copy")
    cv2.waitKey()
    cv2.destroyAllWindows()

def grayscale(image): # Task 5

    # This is just one command, so is not really anything to say here
    grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("grayscale")

    cv2.imshow('grayscale', grayscaled_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def hsv(image): # Task 6
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow('hue_shifted', hsv_image)
    print("hsv")

    cv2.waitKey()
    cv2.destroyAllWindows()

def hue_shifted(image, emptyPictureArray, hue):
    """
    Seems like the task wants me to change the color value by 50 for all the color values.
    I have no concept over what the hue shifting is supposed to be there for...
    :param image:
    :param emptyPictureArray:
    :param hue:
    :return:
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    h = ((h.astype(np.int16) + hue) % 180).astype(np.uint8)
    merged = cv2.merge([h, s, v])

    cv2.imshow('hue_shifting', merged)

    print("hue_shifting")

    cv2.waitKey()
    cv2.destroyAllWindows()

def smoothing(image): # Task 8
    """
    Understanding how to use the gaussian blur
    Here we're just using the command from cv2, and it is performing the work for us.


    """
    ksize = (15,15)
    smoothed_image = cv2.GaussianBlur(image, ksize, sigmaX=0, borderType=cv2.BORDER_DEFAULT)
    cv2.imshow('smoothed_image', smoothed_image)
    print("smoothing")

    cv2.waitKey()
    cv2.destroyAllWindows()


def rotation(image, rotation_angle): # Task 9

    if rotation_angle == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) # Could use cv2.ROTATE_90_COUNTERCLOCKWISE

    elif rotation_angle == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)

    else:
        print("Seems like you inserted a value that neither were 180 or 90 degrees")
        return

    cv2.imshow('rotated_image', image)
    print("rotation")
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    # Creating picture and variable for Task 1
    image = cv2.imread("lena-2.png")
    border_width = 100

    padding(image, border_width) # Task 1

    # Defining variables for cropping lena-2.png
    x_0 = 80
    x_1 = 130
    y_0 = 80
    y_1 = 130

    crop(image, x_0, x_1, y_0, y_1) # Task 2

    width = 200
    height = 200
    resize(image, width, height) # Task 3

    # Creating an empty array
    # Having to redifine height and width due to variables being used for resize task
    height, width = image.shape[:2]
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    copy(image, emptyPictureArray) # Task 4

    grayscale(image) # Task 5

    hsv(image) # Task 6

    hue = 50
    hue_shifted(image, emptyPictureArray, hue) # Task 7

    smoothing(image) # Task 8

    print("Are we rotating the image 90 degrees or 180")
    rotation_angle = int(input())
    rotation(image, rotation_angle) # Task 9


if __name__ == '__main__':
    main()
