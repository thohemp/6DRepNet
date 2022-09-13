# Import SixDRepNet
from SixDRepNet import SixDRepNet

import cv2
import numpy as np
from math import cos, sin


# Create model
# Weights are automatically downloaded
model = SixDRepNet()

img = cv2.imread('/path/to/image.jpg')

pitch, yaw, roll = model.predict(img)

model.draw_axis(img, yaw, pitch, roll)

cv2.imshow("test_window", img)
cv2.waitKey(0)

