import os
import cv2
import shutil


# Create patches directory
crop_save = 'D:/Users/ddd/AIC/merge_dataset/'
if not os.path.exists(crop_save):
    os.makedirs(crop_save)

# For AIC19
data_path = 'D:/Users/ddd/AIC/AIC22/train/'

scenes = os.listdir(data_path)
for scene in scenes:
    # For each cam
    cams = os.listdir(data_path + scene)
    for cam in cams:
        # Set path
        cam_path = data_path + scene + '/' + cam + '/'
        print("cam-path", cam_path)
        # Read gt_file and write to csv_file
        gt_list = open(cam_path + 'gt/gt.txt', 'r').readlines()
        index = 1
        for line in gt_list:
            # Read GT
            line = line.split(',')
            f_num, obj_id = int(line[0]), int(line[1])
            left, top = round(float(line[2])), round(float(line[3]))
            w, h = round(float(line[4])), round(float(line[5]))
            obj_id = (obj_id - 1) if obj_id < 96 else (obj_id - 146)

            # Read frame image
            img_path = cam_path + 'frame/%s_f%04d.jpg' % (cam, f_num)

            print("imh_path==", img_path)
            frame_img = cv2.imread(img_path)

            # Save bbox patch
            bbox = frame_img[top:top+h+1, left:left+w+1, :]
            cv2.imwrite(crop_save + '%04d_%s_%08d_0.jpg' % (obj_id, cam, f_num), bbox)

        # print current status
        print('%s_%s Finished' % (scene, cam))

# For VeRi
data_paths = ['D:/Users/ddd/VeRi/VeRi/image_query/', 'D:/Users/ddd/VeRi/VeRi/image_test/', 'D:/Users/ddd/VeRi/VeRi/image_train/']

for data_path in data_paths:
    img_names = os.listdir(data_path)
    for img_name in img_names:
        obj_id, cam, f_num, _ = img_name.split('.jpg')[0].split('_')
        new_img_name = '%04d_c%03d_%s_1.jpg' % (int(obj_id) + 183, int(cam[1:]) + 40, f_num)
        shutil.copy(data_path + img_name, crop_save + new_img_name)

# For VehicleID
data_paths = ['C:/Users/cjh/Desktop/VehicleID/data/2016-CVPR-VehicleReId-dataset/1B/',
              'C:/Users/cjh/Desktop/VehicleID/data/2016-CVPR-VehicleReId-dataset/2B/',
              'C:/Users/cjh/Desktop/VehicleID/data/2016-CVPR-VehicleReId-dataset/3B/',
              'C:/Users/cjh/Desktop/VehicleID/data/2016-CVPR-VehicleReId-dataset/4B/',
              'C:/Users/cjh/Desktop/VehicleID/data/2016-CVPR-VehicleReId-dataset/5B/'
              ]

for data_path in data_paths:
    img_names = os.listdir(data_path)
    for img_name in img_names:
        cam = data_path.split("/")[-2][0]
        obj_id, f_num, _ = img_name.split('.jpg')[0].split('_')
        new_img_name = '%04d_c%03d_%s_1.jpg' % (int(obj_id) + 959, int(cam) + 60, f_num)
        shutil.copy(data_path + img_name, crop_save + new_img_name)