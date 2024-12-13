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
crop_save = 'D:/Users/ddd/AIC/AIC_test/'
if not os.path.exists(crop_save):
    os.makedirs(crop_save)

# For AIC19
data_path = 'D:/Users/ddd/AIC/AIC22/test/'

scenes = os.listdir(data_path)
for scene in scenes:
    # For each cam
    cams = os.listdir(data_path + scene)
    for cam in cams:
        # Set path
        cam_path = data_path + scene + '/' + cam + '/'
        cam_save = crop_save + '/' + cam + '/'
        if not os.path.exists(cam_save):
            os.makedirs(cam_save)
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

            # Read frame image
            img_path = cam_path + 'frame/000%04d.jpg' % f_num
            # img_path = cam_path + 'frame/000%04d.jpg' % index
            # index = index+1
            print("imh_path==", img_path)
            frame_img = cv2.imread(img_path)

            # Save bbox patch
            bbox = frame_img[top:top+h+1, left:left+w+1, :]
            cv2.imwrite(cam_save + '%04d_%04d.jpg' % (obj_id, f_num), bbox)

        # print current status
        print('%s_%s Finished' % (scene, cam))
