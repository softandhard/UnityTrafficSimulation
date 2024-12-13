import numpy as np
import cv2

# 读取图片
image = cv2.imread('C:/Users/cjh/Desktop/MTMCT/data_process/rois/label.png')

# 将彩色图像转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image = cv2.resize(gray_image, (1920, 1080))

# 显示灰度图像和二值图像
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存二值图像
cv2.imwrite('grayscale.png', gray_image)

# 读取灰度图像
gray_image = cv2.imread('grayscale.png', cv2.IMREAD_GRAYSCALE)

# 创建具有自定义黑白颜色的图像
black_color = 0  # 黑色
white_color = 255  # 白色
custom_black_white_image = np.where(gray_image < 1, black_color, white_color).astype(np.uint8)

# 显示或保存转换后的图像
cv2.imshow('Custom Black White Image', custom_black_white_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('rois/drone2.jpg', custom_black_white_image)