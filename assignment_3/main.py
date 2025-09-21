import cv2
import numpy as np


def sobel_edge_detection(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (3, 3), sigmaX=0)
    sobel = cv2.Sobel(blurred, cv2.CV_64F, dx=1, dy=1, ksize=1)
    sobel = np.absolute(sobel)
    sobel = np.uint8(sobel)
    cv2.imwrite('picture_solutions/sobel_edges.jpg', sobel)


def canny_edge_detection(image, threshold_1=50, threshold_2=50):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (3, 3), sigmaX=0)
    canny = cv2.Canny(blurred, threshold_1, threshold_2)
    cv2.imwrite('picture_solutions/canny_edges.jpg', canny)


def template_match(image, template):
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image.copy()

    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template.copy()

    h, w = template_gray.shape[:2]
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    locations = np.where(res >= threshold)
    result_img = image.copy()

    for pt in zip(*locations[::-1]):
        cv2.rectangle(result_img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite('picture_solutions/template_match_result.jpg', result_img)


def resize(image, scale_factor=2, up_or_down="up"):
    current_img = image.copy()

    if up_or_down.lower() == "up":
        for i in range(int(np.log2(scale_factor))):
            current_img = cv2.pyrUp(current_img)
    elif up_or_down.lower() == "down":
        for i in range(int(np.log2(scale_factor))):
            current_img = cv2.pyrDown(current_img)
    else:
        print("Invalid direction. Use 'up' or 'down'")
        return

    filename = f'picture_solutions/resized_{up_or_down}_{scale_factor}x.jpg'
    cv2.imwrite(filename, current_img)


def main():
    img = cv2.imread('lambo.png')
    shapes_img = cv2.imread('shapes-1.png')
    template_img = cv2.imread('shapes_template.jpg')

    sobel_edge_detection(img)
    canny_edge_detection(img, 50, 50)
    template_match(shapes_img, template_img)
    resize(img, 2, "up")
    resize(img, 2, "down")

    print("All images have been processed and saved to picture_solutions folder")


if __name__ == "__main__":
    main()