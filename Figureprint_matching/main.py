import cv2
import matplotlib.pyplot as plt

def preprocess_fingerprint(image_path):
    img = cv2.imread(image_path, 0)
    return img

def compare_two_fingerprints(img1_path, img2_path, threshold=20, save_result=None):
    img1 = preprocess_fingerprint(img1_path)
    img2 = preprocess_fingerprint(img2_path)

    sift = cv2.SIFT_create(nfeatures=1000)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

    match_count = len(good_matches)
    is_match = match_count > threshold

    if match_count > 0:
        good_matches_sorted = sorted(good_matches, key=lambda x: x.distance)
        top_matches = good_matches_sorted[:min(50, len(good_matches_sorted))]
        avg_distance = sum([m.distance for m in top_matches]) / len(top_matches)
        similarity_score = max(0, 100 - (avg_distance / 3))
    else:
        similarity_score = 0

    img1_display = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_display = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    match_img = cv2.drawMatches(
        img1_display, kp1,
        img2_display, kp2,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))

    result_text = "MATCH" if is_match else "NO MATCH"
    plt.title(f'Fingerprint Comparison: {result_text}\n'
              f'{match_count} good matches | Similarity: {similarity_score:.2f}%',
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    if save_result:
        plt.savefig(save_result, dpi=150, bbox_inches='tight')

    plt.show()

    return match_count, is_match, similarity_score


compare_two_fingerprints("img_2.png", "img_3.png")