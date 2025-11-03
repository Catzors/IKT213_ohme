import cv2
import numpy as np

def harris_corner_detection(reference_image_path, output_path='harris.png'):
    img = cv2.imread(reference_image_path)
    if img is None:
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imwrite(output_path, img)
    return img

def align_images_sift(image_to_align_path, reference_image_path,
                      max_features=10, good_match_percent=0.7):
    img_to_align = cv2.imread(image_to_align_path)
    reference_img = cv2.imread(reference_image_path)

    if img_to_align is None or reference_img is None:
        return None, None

    img_to_align_gray = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(img_to_align_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(reference_gray, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance, reverse=False)
    num_good_matches = int(len(good_matches) * good_match_percent)
    good_matches = good_matches[:num_good_matches]

    matches_img = cv2.drawMatches(img_to_align, keypoints1,
                                  reference_img, keypoints2,
                                  good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    height, width, channels = reference_img.shape
    aligned_img = cv2.warpPerspective(img_to_align, h, (width, height))

    return aligned_img, matches_img

def main():
    harris_img = harris_corner_detection('reference_img.png', 'harris.png')

    aligned_img, matches_img = align_images_sift(
        image_to_align_path='align_this.jpg',
        reference_image_path='reference_img.png',
        max_features=10,
        good_match_percent=0.7
    )

    if aligned_img is not None:
        cv2.imwrite('aligned.png', aligned_img)

    if matches_img is not None:
        cv2.imwrite('matches.png', matches_img)

if __name__ == "__main__":
    main()