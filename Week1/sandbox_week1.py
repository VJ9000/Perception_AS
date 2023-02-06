import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('appletree.jpg')

cv2.imshow("test",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
