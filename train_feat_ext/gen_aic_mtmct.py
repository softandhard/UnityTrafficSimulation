import os
import sys
import cv2
import glob

def video2frames(sourceVdo, dstDir):
    videoData = cv2.VideoCapture(sourceVdo)
    count = 0
    while (videoData.isOpened()):
        count += 1
        ret, frame = videoData.read()
        if ret:
            cv2.imwrite(f"{dstDir}\\{count:07d}.jpg", frame)
            if count % 20 == 0:
                print(f"{dstDir}\\{count:07d}.jpg")
        else:
            break
    videoData.release()

def transSeq(seqs_path, new_root):
    sonCameras = glob.glob(seqs_path + "\\*")
    sonCameras.sort()
    for vdoList in sonCameras:
        Seq = vdoList.split('\\')[-2]
        Camera = vdoList.split('\\')[-1]
        os.system(f"mkdir {new_root}\\{Seq}\\images\\{Camera}\\img1")

        roi_path = vdoList + '\\roi.jpg'
        new_roi_path = f"{new_root}\\{Seq}\\images\\{Camera}"
        os.system(f"copy {roi_path} {new_roi_path}")

        video2frames(f"{vdoList}\\vdo.avi", f"{new_root}\\{Seq}\\images\\{Camera}\\img1")

if __name__ == "__main__":

    # 新数据集的路径
    new_root = 'D:\\Users\\ddd\\AIC\\aic21mtmct_vehicle'
    # 原始数据集的路径，默认为AIC22的S02场景
    seq_path = "D:\\Users\\ddd\\AIC\\AIC22\\test\\S02"
    seq_name = seq_path.split('\\')[-1]
    data_path = seq_path.split('\\')[-3]
    print("data-path==", data_path)
    print("seq_name==", seq_name)
    data_path_new = "AIC22"
    os.system(f"mkdir {new_root}\\{seq_name}\\gt")
    os.system(f"copy {data_path}\\eval\\ground*.txt {new_root}\\{seq_name}\\gt")

    # extract video frames
    transSeq(seq_path, new_root)

