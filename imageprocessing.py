import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
for file in os.listdir("./source"):
  # Load the image
  img = Image.open(f"./source/{file}")

  # Convert the image to grayscale
  gray = img.convert('L')

  # Convert the grayscale image to a NumPy array
  img_array = np.array(gray)

  # Define the Sobel filter
  sobel_filter = np.array([[-1], [0], [1]])

  # Apply the Sobel filter to the image
  edge_x = np.abs(convolve2d(img_array, sobel_filter, mode="same"))
  edge_y = np.abs(convolve2d(img_array, sobel_filter.T, mode="same"))
  edge = edge_x + edge_y
  edge = edge.reshape(img_array.shape)

  # Save the modified image
  img_output = Image.fromarray(edge)
  img_output = img_output.convert("RGB")
  img_output.save(f"./output/{file}")