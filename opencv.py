import cv2
import matplotlib.pyplot as plt

img = cv2.imread("assets/mallard.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp = sift.detect(gray, None)

img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(img)
plt.show()
# cv2.imwrite("image-with-keypoints.jpg", img)
