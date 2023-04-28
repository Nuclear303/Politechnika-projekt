import numpy as np
import cv2

# Load an image
img = cv2.imread('./source/2h_milling1200C_2h.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Sobel operator to obtain the gradient images
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)

# Calculate the magnitude of the gradient at each pixel
mag = np.sqrt(sobelx**2 + sobely**2)

# Apply threshold to obtain binary image of edges
thresh = 50
edge_img = np.zeros_like(img)
edge_img[mag > thresh] = 255

# Display the edge image
cv2.imshow('Edge Image', edge_img)
cv2.waitKey(0)
cv2.destroyAllWindows()