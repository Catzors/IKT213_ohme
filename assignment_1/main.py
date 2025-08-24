import numpy as np
import cv2

# Task IV
def print_image_information(image):
    cv2.imshow('image', image)
    height, width, channels = image.shape
    print("Height = ", height)
    print("Width = ", width)
    print("Channels = ", channels)

    size = image.size
    print("Size = ", size)

    data_type = image.dtype
    print("Data Type = ", data_type)

    print(cv2.imshow('image', image))
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image = cv2.imread('lena-1.png', cv2.IMREAD_COLOR)
    print_image_information(image)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    with open(r"solutions\output.txt", "w") as f:
        f.write(f"Frame Rate = {frame_rate}\n")
        f.write(f"Frame Width = {frame_width}\n")
        f.write(f"Frame Height = {frame_height}\n")


