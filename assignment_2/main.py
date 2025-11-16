import cv2
import numpy as np

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

    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Printing data and picture to verify results
    print(resized_image.shape)
    print(cv2.imshow('resized_image', resized_image))
    print("resize")
    cv2.waitKey()
    cv2.destroyAllWindows()

def copy(image, emptyPictureArray): # Task 4

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

def hue_shifted(image, emptyPictureArray, hue): # Task 7
    # Work directly with RGB/BGR values, not HSV
    # Add hue value to all color channels and handle overflow with modulo
    shifted_image = ((image.astype(np.int16) + hue) % 256).astype(np.uint8)
    cv2.imshow('hue_shifting', shifted_image)
    print("hue_shifting")
    cv2.waitKey()
    cv2.destroyAllWindows()

def smoothing(image): # Task 8

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
