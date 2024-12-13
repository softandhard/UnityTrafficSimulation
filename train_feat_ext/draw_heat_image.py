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
from train_feat_ext import config as config
from train_feat_ext.nets import estimator_resnet101 as estimator_resnet101
from train_feat_ext.nets import estimator_resnext101 as estimator_resnext101
from train_feat_ext.nets import estimator_CBAM_attention_net as estimator_CBAM_attention
from cam_tools import GradCAM, show_cam_on_image, center_crop_img



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
    plt.imshow(visualization)
    plt.show()
    plt.imsave(save_path, visualization)

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
    plt.imshow(visualization)
    plt.show()
    plt.imsave(save_path, visualization)

# model1 = resnext101_ibn_a()
# model1.load_param(config.model_path1)
# model1 = estimator_resnext.Estimator(model1, config.avg_type)
model1 = estimator_resnext101.Estimator(config.model_name, config.avg_type, config.model_path1)

# model2 = resnet101_ibn_a()
# model2.load_param(config.model_path2)
# model2 = estimator.Estimator(model2, config.avg_type)
model2 = estimator_resnet101.Estimator(config.model_name2, config.avg_type, config.model_path2)
print("model==", model1)


draw_CAM(model=model2, img_path=r"C:\Users\cjh\Desktop\MTMCT\train_feat_ext\test.jpg",
    save_path=r"C:\Users\cjh\Desktop\MTMCT\train_feat_ext\resnet50.jpg")

# draw_merge_image(model1, model2, confidence=0.9, img_path=r"C:\Users\cjh\Desktop\MTMCT\train_feat_ext\test.jpg",
#                  save_path=r"C:\Users\cjh\Desktop\MTMCT\train_feat_ext\merge2.jpg")

