import cv2

# 读取两张黑白图像
image1 = cv2.imread('C:/Users/cjh/Desktop/MTMCT/data_process/rois/drone1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('C:/Users/cjh/Desktop/MTMCT/data_process/rois/drone2.jpg', cv2.IMREAD_GRAYSCALE)

# 对两张图像进行按位与操作
overlap_image = cv2.bitwise_and(image1, image2)

# 显示或保存处理后的图像
cv2.imshow('Overlapping Area', overlap_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('C:/Users/cjh/Desktop/MTMCT/data_process/overlaps/drone1_drone2.jpg', overlap_image)