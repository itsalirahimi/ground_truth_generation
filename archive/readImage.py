from email.mime import image
import cv2
import os

img = 'df.png'
print("file exists?", os.path.exists(img))

image = cv2.imread(img)

cv2.imshow("img", image)
cv2.waitKey()