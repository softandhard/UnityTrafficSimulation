import os
import cv2
import shutil

# Naming Rule of the bboxes
# In bbox "0001_c1s1_001051_00.jpg"
# "0001" is object ID
# "c1" is the camera ID
# "s1" is sequence ID of camera "1".
# "001051" is the 1051^th frame in the sequence "c1s1"

# Create patches directory
crop_save = 'C:/Users/cjh/Desktop/Fast_Online_MTMCT-main/dataset/VeRi/'
if not os.path.exists(crop_save):
    os.makedirs(crop_save)


# For VeRi
data_paths = ['D:/Users/ddd/VeRi/VeRi/image_query/', 'D:/Users/ddd/VeRi/VeRi/image_test/', 'D:/Users/ddd/VeRi/VeRi/image_train/']

for data_path in data_paths:
    img_names = os.listdir(data_path)
    for img_name in img_names:
        obj_id, cam, f_num, _ = img_name.split('.jpg')[0].split('_')
        new_img_name = '%04d_c%03d_%s_1.jpg' % (int(obj_id) - 1, int(cam[1:]) + 40, f_num)
        shutil.copy(data_path + img_name, crop_save + new_img_name)
