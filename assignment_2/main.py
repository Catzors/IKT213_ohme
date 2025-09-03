import cv2
import numpy as np
import matplotlib.pyplot as plt

def padding(image, border_width):
    reflect = cv2.copyMakeBorder(image,border_width,border_width,border_width,border_width,cv2.BORDER_REFLECT)
    print(image.shape)
    print(reflect.shape)
    print(cv2.imshow('reflect', reflect))
    cv2.waitKey()
    cv2.destroyAllWindows()

def main():
    image = cv2.imread("lena-2.png")
    border_width = 100
    padding(image, border_width) # Task 1





if __name__ == '__main__':
    main()
