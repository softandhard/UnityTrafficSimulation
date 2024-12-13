# coding: utf-8
import json
import os
from PIL import Image
import torch
import numpy as np
import cv2
from torchvision import transforms
from torchvision import models
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from online_MTMC import config as config
from train_feat_ext.nets import estimator as estimator
from train_feat_ext.nets import estimator_resnext as estimator_resnext
from train_feat_ext.nets import estimator_CBAM_attention_net as estimator_CBAM_attention
from train_feat_ext.cam_tools import GradCAM, show_cam_on_image, center_crop_img
from train_feat_ext.nets.resnet_ibn_a import resnet50_ibn_a
from train_feat_ext.nets.resnet_ibn_a import resnet101_ibn_a
from train_feat_ext.nets.resnext_ibn import resnext101_ibn_a
from train_feat_ext.nets.deepsort_CBAM_model import AttentionNet
from online_MTMC.opts_merge import opt
from online_MTMC.models.experimental import attempt_load
from online_MTMC.utils.general import check_img_size, non_max_suppression, scale_coords
from online_MTMC.utils.datasets import letterbox as convert_letterbox

def preprocess_img(ori_img, img_size, stride):
    img = None
    img_ori = None

    # Padded resize
    img1 = convert_letterbox(ori_img, img_size, stride=stride)[0]
    # Convert
    img1 = img1[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img1 = np.ascontiguousarray(img1)
    img = img1
    img_ori = ori_img

    return img, img_ori

det_model = attempt_load(opt.det_weights + opt.det_name + '.pt')
det_model = det_model.cuda().eval().half()

# 设置检测模型参数
stride = int(det_model.stride.max())
img_size = opt.img_size.copy()
img_size[0] = check_img_size(opt.img_size[0], s=stride)
img_size[1] = check_img_size(opt.img_size[1], s=stride)
img_h, img_w = opt.img_ori_size

batch_img = torch.zeros((1, 3, img_size[0], img_size[1]), device='cuda').half()
batch_img_ori = torch.zeros((1, 3, opt.img_ori_size[0], opt.img_ori_size[1]), device='cuda').half()

ori_img = cv2.imread(r"C:\Users\cjh\Desktop\MTMCT\train_feat_ext\test.jpg")
convert_img, ori_img = preprocess_img(ori_img, img_size, stride)
batch_img[0] = torch.tensor(convert_img / 255.0, device='cuda').half()

# 预热模型
with torch.autocast('cuda'):
    for _ in range(10):
        det_model(torch.rand((4, 3, img_size[0], img_size[1]), device='cuda').half())

# 检测图像 =====================================================================================================
with torch.autocast('cuda'):
    # preds 是推理的结果，通常包括了检测到的目标的类别、位置和置信度等信息
    preds = det_model(batch_img, augment=opt.augment)[0]

# 非极大值抑制
preds = non_max_suppression(preds, opt.conf_thres, opt.iou_thres,
                            classes=opt.classes, agnostic=opt.agnostic_nms)

conf_list = []
for pdx, pred in enumerate(preds):

    # If there are valid predictions
    if len(pred) > 0:
        # Rescale boxes from img_size to im0s size
        pred[:, :4] = scale_coords(batch_img.shape[2:], pred[:, :4], batch_img_ori.shape[2:4])

        # Post-process detections
        for *xyxy, conf, _ in reversed(pred):
            # Convert to integer
            x1, y1 = round(xyxy[0].item()), round(xyxy[1].item())
            x2, y2 = round(xyxy[2].item()), round(xyxy[3].item())

            # 选择大于最小阈值的框
            if (x2 - x1) * (y2 - y1) > img_h * img_w * opt.min_box_size / 2:
                conf_list.append(conf.item())

max_conf = 0
for i in range(len(conf_list)):
    if conf_list[i] > max_conf:
        max_conf = conf_list[i]

print("max-conf=", max_conf)

def draw_CAM(model, img_path, save_path):
    # ----载入自己的模型，按照自己训练集训练的
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    target_layers = [model.backbone]  # 拿到最后一个层结构

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = img_path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    # img = img.resize((384, 384))
    img = np.array(img, dtype=np.uint8)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    # # 将所有张量移动到选择的设备上
    input_tensor = input_tensor.to(device, dtype=torch.float32)
    # 实例化，输出模型，要计算的层
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    # 感兴趣的label
    target_category = [2]  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog
    # 计算cam图
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # 实例化
    # 将只传入的一张图的cam图提取出来
    grayscale_cam = grayscale_cam[0, :]
    # 变成彩色热力图的形式
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,  # 将原图缩放到[0,1]之间
                                      grayscale_cam,
                                      use_rgb=True)
    # 保存图片
    # plt.imshow(visualization)
    # plt.show()
    # plt.imsave("./resnext101.jpg", visualization)
    cv2.imshow("res", visualization)
    cv2.imwrite(save_path, visualization)
    cv2.waitKey(0)

def draw_merge_image(model1, model2, confidence, img_path, save_path):
    # ----载入自己的模型，按照自己训练集训练的
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model1 = model1.to(device)
    target_layers = [model1.backbone]  # 拿到最后一个层结构

    model2 = model2.to(device)
    target_layers2 = [model2.backbone]  # 拿到最后一个层结构

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = img_path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    # img = img.resize((384, 384))
    img = np.array(img, dtype=np.uint8)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    # # 将所有张量移动到选择的设备上
    input_tensor = input_tensor.to(device)
    # 实例化，输出模型，要计算的层
    cam = GradCAM(model=model1, target_layers=target_layers, use_cuda=False)
    cam2 = GradCAM(model=model2, target_layers=target_layers2, use_cuda=False)
    # 感兴趣的label
    target_category = [2]  # tabby, tabby cat
    # 计算cam图
    grayscale_cam1 = cam(input_tensor=input_tensor, target_category=target_category)  # 实例化
    grayscale_cam2 = cam2(input_tensor=input_tensor, target_category=target_category)  # 实例化
    # 将只传入的一张图的cam图提取出来
    grayscale_cam1 = grayscale_cam1[0, :]
    grayscale_cam2 = grayscale_cam2[0, :]
    grayscale_cam = confidence / 2 * grayscale_cam1 + confidence / 2 * grayscale_cam2 + (1-confidence) * grayscale_cam1

    # 变成彩色热力图的形式
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,  # 将原图缩放到[0,1]之间
                                      grayscale_cam,
                                      use_rgb=True)
    # 保存图片
    # plt.imshow(visualization)
    # plt.show()
    # plt.imsave("./merge.jpg", visualization)
    cv2.imshow("res", visualization)
    cv2.imwrite(save_path, visualization)
    cv2.waitKey(0)


model1 = resnext101_ibn_a()
model1.load_param(config.model_path1)
model1 = estimator_resnext.Estimator(model1, config.avg_type)

model2 = resnet101_ibn_a()
model2.load_param(config.model_path2)
model2 = estimator.Estimator(model2, config.avg_type)
# model = estimator_resnet101.Estimator(config.model_name, config.avg_type, config.model_path)
print("model==", model1)


draw_CAM(model=model1, img_path=r"C:\Users\cjh\Desktop\MTMCT\train_feat_ext\test.jpg",
    save_path=r"C:\Users\cjh\Desktop\MTMCT\online_MTMC\resnet50.jpg")

# draw_merge_image(model1, model2, confidence=max_conf, img_path=r"C:\Users\cjh\Desktop\MTMCT\train_feat_ext\test2.jpg",
#                  save_path=r"C:\Users\cjh\Desktop\MTMCT\online_MTMC\merge.jpg")

