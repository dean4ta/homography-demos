import cv2
import numpy as np
import sys


# To try this out, download http://hugin.sourceforge.net/tutorials/two-photos/974-1.jpg as image1,
# http://hugin.sourceforge.net/tutorials/two-photos/975-1.jpg as image2 and run
# python panorama_stitcher.py image1.jpg image2.jpg


# Configuration
good_distance_ratio = 0.85
min_matching_features = 10

# Read images
image1 = cv2.imread(sys.argv[1])
image2 = cv2.imread(sys.argv[2])


# Create an ORB feature finder
orb_finder = cv2.ORB_create(nfeatures=1000,
                            scoreType=cv2.ORB_FAST_SCORE)


# Extract features from both the images
def getFeatures(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, descriptors = orb_finder.detectAndCompute(gray, None)
    return {'features': features, 'descriptors': descriptors}


features_image1 = getFeatures(image1)
features_image2 = getFeatures(image2)

# Initializer and a run a knn-based matcher
matcher = cv2.BFMatcher()
raw_matches = matcher.knnMatch(features_image1['descriptors'],
                               features_image2['descriptors'],
                               k=2)

# Basic filter on the matches to based on descriptor distance
good_features = []
good_matches = []
for m1, m2 in raw_matches:
    if m1.distance < good_distance_ratio * m2.distance:
        good_features.append((m1.trainIdx, m1.queryIdx))
        good_matches.append([m1])

# Display the matches
matches_image = cv2.drawMatchesKnn(image1, features_image1['features'],
                                   image2, features_image2['features'],
                                   good_matches, None, flags=0)

# Go ahead populate feature coordinates and compute homography only if
# a minimum number of matching features are found
if len(good_features) > min_matching_features:
    feature_coordinates_image1 = np.float32(
        [features_image1['features'][i].pt for (_, i) in good_features])
    feature_coordinates_image2 = np.float32(
        [features_image2['features'][i].pt for (i, _) in good_features])

    # Compute Homography from feature coordinates
    H, _ = cv2.findHomography(
        feature_coordinates_image2, feature_coordinates_image1, cv2.RANSAC, 5.0)

# Begin constructing a panorama. Height is just the height of one of the images,
# width is the sum of widths of the two input images
panorama_width = image1.shape[1] + image2.shape[1]
panorama_height = image1.shape[0]

# Correct perspective of the second image into the panorama
panorama = cv2.warpPerspective(image2, H,
                               (panorama_width, panorama_height))

# Add the first image to the panorama
panorama[0:image1.shape[0], 0:image1.shape[1], :] = image1

# Display
cv2.imshow("Panorama", panorama)
cv2.imshow("Matches", matches_image)
cv2.imshow("Second image", image2)
cv2.imshow("First image", image1)

cv2.waitKey(0)