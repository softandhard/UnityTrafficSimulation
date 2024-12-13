import os
import cv2
import shutil

# Naming Rule of the bboxes
# In bbox "0001_c1s1_001051_00.jpg"
# "0001" is object ID
# "c1" is the camera ID
# "s1" is sequence ID of camera "1".
# "001051" is the 1051^th frame in the sequence "c1s1"

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
        print("cam==", cam)
        # Read gt_file and write to csv_file
        gt_list = open(cam_path + 'gt/gt.txt', 'r').readlines()
        for line in gt_list:
            # Read GT
            line = line.split(',')
            f_num, obj_id = int(line[0]), int(line[1])
            left, top = round(float(line[2])), round(float(line[3]))
            w, h = round(float(line[4])), round(float(line[5]))
            obj_id = (obj_id - 1) if obj_id < 96 else (obj_id - 146)
            print("f-num==", f_num)

            old_img_path = cam_path + 'frame/000%04d.jpg' % f_num
            if os.path.exists(old_img_path):
                # 新的图片路径
                new_img_path = cam_path + 'frame/%s_f%04d.jpg' % (cam, f_num)
                # 重命名图片
                os.rename(old_img_path, new_img_path)
            else:
                print("文件路径不存在")

        # print current status
        print('%s_%s Finished' % (scene, cam))



from PIL import Image
import os

# 指定图片文件夹路径
image_folder_path = "D:/Users/ddd/AIC/AIC22/train/"

# 要搜索的特定字符
search_character = "c"

# 用于存储包含特定字符的文件名的列表
matching_file_names = []
count = 0
scenes = os.listdir(image_folder_path)
# 遍历图片文件夹中的文件
for scene in scenes:
    # For each cam
    cams = os.listdir(image_folder_path + scene)
    for cam in cams:
        # Set path
        img_path = image_folder_path + scene + '/' + cam + '/frame/'
        for file_name in os.listdir(img_path):
            # 构建完整的文件路径
            file_path = os.path.join(img_path, file_name)
            # 检查文件是否是图片文件
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # 读取图片文件
                try:
                    with Image.open(file_path):
                        # 检查文件名是否包含特定字符
                        if search_character not in file_name:
                            count = count+1
                            matching_file_names.append(file_name)
                except Exception as e:
                    print(f"无法打开图片文件 {file_name}: {e}")


# 打印包含特定字符的文件名
print("不包含特定字符的文件名：", matching_file_names)
print("文件总数: ", count)
