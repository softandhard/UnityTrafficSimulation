import cv2
import numpy as np

# 指定生成白色图像的大小
image_width = 1920
image_height = 1080

# 创建全白色图像
white_image = np.ones((image_height, image_width), dtype=np.uint8) * 255

# 显示或保存生成的白色图像
cv2.imshow('White Image', white_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('white_image.jpg', white_image)