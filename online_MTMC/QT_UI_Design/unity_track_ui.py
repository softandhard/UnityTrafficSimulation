#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author   : one_republic
# @file     : ui_yolov5.py
# @Time     : 2024/1/20 10:13

import time
import os
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import QTimer, Qt
import cv2
import sys
import torch
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMessageBox, QHBoxLayout, QVBoxLayout, \
    QWidget, QLabel, QInputDialog, QLayout, QPushButton, QSlider, QCheckBox
from unity_detect_backend import main_detect_process, lodel_detector_model, load_detector_opt
from unity_track_backend import main_track_process, create_unity_connection, receive_unity_data, \
    load_trackor_opt, load_model

print(torch.cuda.is_available())

'''仿真和视频实时检测界面'''

# 检测框厚度调整
class ThicknessSlider(QWidget):
    def __init__(self):
        super().__init__()
        self.confirm = False
        self.initUI()

    def initUI(self):
        self.setWindowTitle('检测框厚度输入界面')
        self.setGeometry(100, 100, 400, 200)
        # 设置窗口图标
        icon = QIcon(os.getcwd() + '\\data\\source_image\\icon.jpg')
        self.setWindowIcon(icon)

        self.decimal_label = QLabel('选择的检测框厚度为: 1', self)
        self.decimal_slider = QSlider()
        self.decimal_slider.setOrientation(1)  # 设置为垂直方向
        self.decimal_slider.setMinimum(1)
        self.decimal_slider.setMaximum(10)
        self.decimal_slider.setTickInterval(1)
        self.decimal_slider.setTickPosition(1)
        self.decimal_slider.valueChanged.connect(self.onSliderChange)

        self.confirm_button = QPushButton('确认输入', self)
        self.confirm_button.clicked.connect(self.confirmInput)

        vbox = QVBoxLayout()
        vbox.addWidget(self.decimal_label)
        vbox.addWidget(self.decimal_slider)
        vbox.addWidget(self.confirm_button)

        self.setLayout(vbox)

    def onSliderChange(self):
        value = self.decimal_slider.value()
        self.decimal_label.setText(f'输入的检测框厚度为: {value:.2f}')

    def confirmInput(self):
        value = self.decimal_slider.value()
        QMessageBox.information(self, '确认输入', f'您输入的检测框厚度为: {value:.2f}')
        self.confirm = True
        self.hide()
        self.close()

# 处理设备选择菜单
class DeviceMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.device = "0"
        self.confirm = False
        self.initUI()

    def initUI(self):
        self.setWindowTitle('处理设备菜单')
        self.setGeometry(100, 100, 400, 400)
        # 设置窗口图标
        icon = QIcon(os.getcwd() + '\\data\\source_image\\icon.jpg')
        self.setWindowIcon(icon)
        # 创建下拉菜单并添加选项
        self.comboBox = QComboBox(self)
        self.comboBox.setGeometry(150, 10, 100, 32)
        self.comboBox.addItem('CPU处理')
        self.comboBox.addItem('显卡处理')

        self.confirm_button = QPushButton('确认输入', self)
        self.confirm_button.setGeometry(150, 200, 100, 32)
        self.confirm_button.clicked.connect(self.confirmInput)

    def confirmInput(self):
        # 获取菜单中勾选的项目
        check_item = self.comboBox.currentText()
        QMessageBox.information(self, '确认输入', f'您选择的处理设备为: {check_item}')
        if check_item == 'CPU处理':
            self.device = 'cpu'
        elif check_item == '显卡处理':
            self.device = "cuda:0"

        self.confirm = True
        self.hide()
        self.close()

# 检测器模型选择菜单
class DetectorModelMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.detector_weight = 'yolov7-w6'
        self.confirm = False

    def initUI(self):
        self.setWindowTitle('检测模型菜单')
        self.setGeometry(100, 100, 400, 400)
        # 设置窗口图标
        icon = QIcon(os.getcwd() + '\\data\\source_image\\icon.jpg')
        self.setWindowIcon(icon)
        # 创建下拉菜单并添加选项
        self.comboBox = QComboBox(self)
        self.comboBox.setGeometry(130, 10, 140, 20)
        self.comboBox.addItem('yolov7-w6')
        self.comboBox.addItem('yolov7-d6')
        self.comboBox.addItem('yolov7-e6')
        self.comboBox.addItem('yolov7-e6e')

        self.confirm_button = QPushButton('确认输入', self)
        self.confirm_button.setGeometry(150, 200, 100, 32)
        self.confirm_button.clicked.connect(self.confirmInput)

    def confirmInput(self):
        # 获取菜单中勾选的项目
        check_item = self.comboBox.currentText()
        QMessageBox.information(self, '确认输入', f'您选择的模型为: {check_item}')
        if check_item == 'yolov7-w6':
            self.detector_weight = 'yolov7-w6'
        elif check_item == 'yolov7-d6':
            self.detector_weight = 'yolov7-d6'
        elif check_item == 'yolov7-e6':
            self.detector_weight = 'yolov7-e6'
        elif check_item == 'yolov7-e6e':
            self.detector_weight = 'yolov7-e6e'

        self.confirm = True
        self.hide()
        self.close()

# 追踪器重识别模型选择菜单
class TrackorModelMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.trackor_weight = 'resnet50_ibn_a'
        self.confirm = False

    def initUI(self):
        self.setWindowTitle('重识别模型菜单')
        self.setGeometry(100, 100, 400, 400)
        # 设置窗口图标
        icon = QIcon(os.getcwd() + '\\data\\source_image\\icon.jpg')
        self.setWindowIcon(icon)
        # 创建下拉菜单并添加选项
        self.comboBox = QComboBox(self)
        self.comboBox.setGeometry(130, 10, 140, 20)
        self.comboBox.addItem('resnet50')
        self.comboBox.addItem('resnet101')

        self.confirm_button = QPushButton('确认输入', self)
        self.confirm_button.setGeometry(150, 200, 100, 32)
        self.confirm_button.clicked.connect(self.confirmInput)

    def confirmInput(self):
        # 获取菜单中勾选的项目
        check_item = self.comboBox.currentText()
        QMessageBox.information(self, '确认输入', f'您选择的模型为: {check_item}')
        if check_item == 'resnet50':
            self.trackor_weight = 'resnet50_ibn_a'
        elif check_item == 'resnet101':
            self.trackor_weight = 'resnet101_ibn_a'

        self.confirm = True
        self.hide()
        self.close()

# 检测器置信度阈值设置
class Conf_threshold_Slider(QWidget):
    def __init__(self):
        super().__init__()
        self.confirm = False
        self.MAX_DIST_value = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle('置信度阈值输入界面')
        self.setGeometry(100, 100, 400, 200)
        # 设置窗口图标
        icon = QIcon(os.getcwd() + '\\data\\source_image\\icon.jpg')
        self.setWindowIcon(icon)

        self.decimal_label = QLabel('选择的置信度阈值为: 0.0', self)
        self.decimal_slider = QSlider()
        self.decimal_slider.setOrientation(1)  # 设置为垂直方向
        self.decimal_slider.setMinimum(0)
        self.decimal_slider.setMaximum(100)
        self.decimal_slider.setTickInterval(1)
        self.decimal_slider.setTickPosition(1)
        self.decimal_slider.valueChanged.connect(self.onSliderChange)

        self.confirm_button = QPushButton('确认输入', self)
        self.confirm_button.clicked.connect(self.confirmInput)

        vbox = QVBoxLayout()
        vbox.addWidget(self.decimal_label)
        vbox.addWidget(self.decimal_slider)
        vbox.addWidget(self.confirm_button)

        self.setLayout(vbox)

    def onSliderChange(self):
        value = self.decimal_slider.value() / 100.0
        self.decimal_label.setText(f'输入的置信度阈值为: {value:.2f}')

    def confirmInput(self):
        value = self.decimal_slider.value() / 100.0
        self.MAX_DIST_value = value
        QMessageBox.information(self, '确认输入', f'您输入的置信度阈值为: {value:.2f}')
        self.confirm = True
        self.hide()
        self.close()

# 检测器IOU阈值设置
class Detector_IOU_threshold_Slider(QWidget):
    def __init__(self):
        super().__init__()
        self.confirm = False
        self.MIN_CONFIDENCE_value = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle('检测器IOU阈值输入界面')
        self.setGeometry(100, 100, 400, 200)
        # 设置窗口图标
        icon = QIcon(os.getcwd() + '\\data\\source_image\\icon.jpg')
        self.setWindowIcon(icon)

        self.decimal_label = QLabel('选择的检测器IOU阈值为: 0.0', self)
        self.decimal_slider = QSlider()
        self.decimal_slider.setOrientation(1)  # 设置为垂直方向
        self.decimal_slider.setMinimum(0)
        self.decimal_slider.setMaximum(100)
        self.decimal_slider.setTickInterval(1)
        self.decimal_slider.setTickPosition(1)
        self.decimal_slider.valueChanged.connect(self.onSliderChange)

        self.confirm_button = QPushButton('确认输入', self)
        self.confirm_button.clicked.connect(self.confirmInput)

        vbox = QVBoxLayout()
        vbox.addWidget(self.decimal_label)
        vbox.addWidget(self.decimal_slider)
        vbox.addWidget(self.confirm_button)

        self.setLayout(vbox)

    def onSliderChange(self):
        value = self.decimal_slider.value() / 100.0
        self.decimal_label.setText(f'输入的检测器IOU阈值为: {value:.2f}')

    def confirmInput(self):
        value = self.decimal_slider.value() / 100.0
        self.MIN_CONFIDENCE_value = value
        QMessageBox.information(self, '确认输入', f'您输入的检测器IOU阈值为: {value:.2f}')
        self.confirm = True
        self.hide()
        self.close()

# 追踪器余弦相似度阈值设置
class Trackor_cos_threshold_Slider(QWidget):
    def __init__(self):
        super().__init__()
        self.confirm = False
        self.NMS_MAX_OVERLAP_value = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle('追踪器余弦相似度阈值输入界面')
        self.setGeometry(100, 100, 400, 200)
        # 设置窗口图标
        icon = QIcon(os.getcwd() + '\\data\\source_image\\icon.jpg')
        self.setWindowIcon(icon)

        self.decimal_label = QLabel('选择的追踪器余弦相似度阈值为: 0.0', self)
        self.decimal_slider = QSlider()
        self.decimal_slider.setOrientation(1)  # 设置为垂直方向
        self.decimal_slider.setMinimum(0)
        self.decimal_slider.setMaximum(100)
        self.decimal_slider.setTickInterval(1)
        self.decimal_slider.setTickPosition(1)
        self.decimal_slider.valueChanged.connect(self.onSliderChange)

        self.confirm_button = QPushButton('确认输入', self)
        self.confirm_button.clicked.connect(self.confirmInput)

        vbox = QVBoxLayout()
        vbox.addWidget(self.decimal_label)
        vbox.addWidget(self.decimal_slider)
        vbox.addWidget(self.confirm_button)

        self.setLayout(vbox)

    def onSliderChange(self):
        value = self.decimal_slider.value() / 100.0
        self.decimal_label.setText(f'输入的追踪器余弦相似度阈值为: {value:.2f}')

    def confirmInput(self):
        value = self.decimal_slider.value() / 100.0
        self.NMS_MAX_OVERLAP_value = value
        QMessageBox.information(self, '确认输入', f'您输入的追踪器余弦相似度阈值为: {value:.2f}')
        self.confirm = True
        self.hide()
        self.close()

# 追踪器IOU阈值设置
class Trackor_IOU_threshold_Slider(QWidget):
    def __init__(self):
        super().__init__()
        self.confirm = False
        self.MAX_IOU_DISTANCE_value = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle('追踪器IOU阈值输入界面')
        self.setGeometry(100, 100, 400, 200)
        # 设置窗口图标
        icon = QIcon(os.getcwd() + '\\data\\source_image\\icon.jpg')
        self.setWindowIcon(icon)

        self.decimal_label = QLabel('选择的追踪器IOU阈值为: 0.0', self)
        self.decimal_slider = QSlider()
        self.decimal_slider.setOrientation(1)  # 设置为垂直方向
        self.decimal_slider.setMinimum(0)
        self.decimal_slider.setMaximum(100)
        self.decimal_slider.setTickInterval(1)
        self.decimal_slider.setTickPosition(1)
        self.decimal_slider.valueChanged.connect(self.onSliderChange)

        self.confirm_button = QPushButton('确认输入', self)
        self.confirm_button.clicked.connect(self.confirmInput)

        vbox = QVBoxLayout()
        vbox.addWidget(self.decimal_label)
        vbox.addWidget(self.decimal_slider)
        vbox.addWidget(self.confirm_button)

        self.setLayout(vbox)

    def onSliderChange(self):
        value = self.decimal_slider.value() / 100.0
        self.decimal_label.setText(f'输入的追踪器IOU阈值为: {value:.2f}')

    def confirmInput(self):
        value = self.decimal_slider.value() / 100.0
        self.MAX_IOU_DISTANCE_value = value
        QMessageBox.information(self, '确认输入', f'您输入的追踪器IOU阈值为: {value:.2f}')
        self.confirm = True
        self.hide()
        self.close()

# 参数信息显示菜单
class ParamsInfoDialog(QDialog):
    def __init__(self, line_thickness, device, detector_model, conf_thres, iou_thres, cos_thr, iou_thr):
        super().__init__()
        self.line_thickness = line_thickness
        self.device = device
        self.detector_model = detector_model
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.cos_thr = cos_thr
        self.iou_thr = iou_thr
        self.init_UI()

    def init_UI(self):
        self.setWindowTitle('参数信息显示')
        self.setGeometry(100, 100, 800, 800)
        # 设置窗口图标
        icon = QIcon(os.getcwd() + '\\data\\source_image\\icon.jpg')
        self.setWindowIcon(icon)
        # 创建一个字体对象
        font = QFont()
        font.setFamily('Arial')  # 设置字体样式
        font.setPointSize(20)  # 设置字体大小
        font.setBold(True)  # 设置为粗体

        self.layout = QHBoxLayout()
        self.title = QLabel('检测框厚度: ')
        self.title.setAlignment(Qt.AlignLeft)
        self.title.setFont(font)
        self.label = QLabel(str(self.line_thickness))
        self.label.setAlignment(Qt.AlignHCenter)
        self.label.setFont(font)

        self.layout2 = QHBoxLayout()
        self.title2 = QLabel('处理设备: ')
        self.title2.setAlignment(Qt.AlignLeft)
        self.title2.setFont(font)
        self.label2 = QLabel(str(self.device))
        self.label2.setAlignment(Qt.AlignHCenter)
        self.label2.setFont(font)

        self.layout3 = QHBoxLayout()
        self.title3 = QLabel('检测模型: ')
        self.title3.setAlignment(Qt.AlignLeft)
        self.title3.setFont(font)
        self.label3 = QLabel(str(self.detector_model))
        self.label3.setAlignment(Qt.AlignHCenter)
        self.label3.setFont(font)

        self.layout4 = QHBoxLayout()
        self.title4 = QLabel('置信度阈值: ')
        self.title4.setAlignment(Qt.AlignLeft)
        self.title4.setFont(font)
        self.label4 = QLabel(str(self.conf_thres))
        self.label4.setAlignment(Qt.AlignHCenter)
        self.label4.setFont(font)

        self.layout5 = QHBoxLayout()
        self.title5 = QLabel('检测器IOU阈值: ')
        self.title5.setAlignment(Qt.AlignLeft)
        self.title5.setFont(font)
        self.label5 = QLabel(str(self.iou_thres))
        self.label5.setAlignment(Qt.AlignHCenter)
        self.label5.setFont(font)

        self.layout6 = QHBoxLayout()
        self.title6 = QLabel('追踪器余弦相似度阈值: ')
        self.title6.setAlignment(Qt.AlignLeft)
        self.title6.setFont(font)
        self.label6 = QLabel(str(self.cos_thr))
        self.label6.setAlignment(Qt.AlignHCenter)
        self.label6.setFont(font)

        self.layout7 = QHBoxLayout()
        self.title7 = QLabel('追踪器IOU阈值: ')
        self.title7.setAlignment(Qt.AlignLeft)
        self.title7.setFont(font)
        self.label7 = QLabel(str(self.iou_thr))
        self.label7.setAlignment(Qt.AlignHCenter)
        self.label7.setFont(font)

        self.layout.addWidget(self.title)
        self.layout.addWidget(self.label)
        self.layout2.addWidget(self.title2)
        self.layout2.addWidget(self.label2)
        self.layout3.addWidget(self.title3)
        self.layout3.addWidget(self.label3)
        self.layout4.addWidget(self.title4)
        self.layout4.addWidget(self.label4)
        self.layout5.addWidget(self.title5)
        self.layout5.addWidget(self.label5)
        self.layout6.addWidget(self.title6)
        self.layout6.addWidget(self.label6)
        self.layout7.addWidget(self.title7)
        self.layout7.addWidget(self.label7)

        self.global_layout = QVBoxLayout()
        self.global_layout.addLayout(self.layout)
        self.global_layout.addLayout(self.layout2)
        self.global_layout.addLayout(self.layout3)
        self.global_layout.addLayout(self.layout4)
        self.global_layout.addLayout(self.layout5)
        self.global_layout.addLayout(self.layout6)
        self.global_layout.addLayout(self.layout7)

        self.setLayout(self.global_layout)

    def show_params(self, line_thickness, device, detector_model, conf_thres, iou_thres, cos_thr, iou_thr):
        self.line_thickness = line_thickness
        if device == 'cpu':
            self.device = 'CPU'
        else:
            self.device = 'GPU'
        if detector_model == 'yolov7-w6':
            self.detector_model = 'yolov7-w6'
        elif detector_model == 'yolov7-d6':
            self.detector_model = 'yolov7-d6'
        elif detector_model == 'yolov7-e6':
            self.detector_model = 'yolov7-e6'
        elif detector_model == 'yolov7-e6e':
            self.detector_model = 'yolov7-e6e'

        self.label.setText((str(self.line_thickness)))
        self.label2.setText((str(self.device)))
        self.label3.setText((str(self.detector_model)))
        self.label4.setText(str(conf_thres))
        self.label5.setText(str(iou_thres))
        self.label6.setText(str(cos_thr))
        self.label7.setText(str(iou_thr))

# 检测结果信息显示菜单
class ResultInfoDialog(QDialog):
    def __init__(self, UAV1_detect, UAV2_detect, UAV1_track, UAV2_track, detect_fps, track_fps):
        super().__init__()
        self.UAV1_detect = UAV1_detect
        self.UAV2_detect = UAV2_detect
        self.UAV1_track = UAV1_track
        self.UAV2_track = UAV2_track
        self.detect_fps = detect_fps
        self.track_fps = track_fps
        self.init_UI()

    def init_UI(self):
        self.setWindowTitle('检测追踪结果信息显示')
        self.setGeometry(100, 100, 800, 800)
        # 设置窗口图标
        icon = QIcon(os.getcwd() + '\\data\\source_image\\icon.jpg')
        self.setWindowIcon(icon)
        # 创建一个字体对象
        font = QFont()
        font.setFamily('Arial')  # 设置字体样式
        font.setPointSize(20)  # 设置字体大小
        font.setBold(True)  # 设置为粗体

        self.layout = QHBoxLayout()
        self.title = QLabel('一号无人机检测结果: ')
        self.title.setAlignment(Qt.AlignLeft)
        self.title.setFont(font)
        self.label = QLabel(str(self.UAV1_detect))
        self.label.setAlignment(Qt.AlignHCenter)
        self.label.setFont(font)

        self.layout2 = QHBoxLayout()
        self.title2 = QLabel('二号无人机检测结果: ')
        self.title2.setAlignment(Qt.AlignLeft)
        self.title2.setFont(font)
        self.label2 = QLabel(str(self.UAV2_detect))
        self.label2.setAlignment(Qt.AlignHCenter)
        self.label2.setFont(font)

        self.layout3 = QHBoxLayout()
        self.title3 = QLabel('一号无人机追踪结果: ')
        self.title3.setAlignment(Qt.AlignLeft)
        self.title3.setFont(font)
        self.label3 = QLabel(str(self.UAV1_track))
        self.label3.setAlignment(Qt.AlignHCenter)
        self.label3.setFont(font)

        self.layout4 = QHBoxLayout()
        self.title4 = QLabel('二号无人机追踪结果: ')
        self.title4.setAlignment(Qt.AlignLeft)
        self.title4.setFont(font)
        self.label4 = QLabel(str(self.UAV2_track))
        self.label4.setAlignment(Qt.AlignHCenter)
        self.label4.setFont(font)

        self.layout5 = QHBoxLayout()
        self.title5 = QLabel('检测帧率: ')
        self.title5.setAlignment(Qt.AlignLeft)
        self.title5.setFont(font)
        self.label5 = QLabel(str(self.detect_fps))
        self.label5.setAlignment(Qt.AlignHCenter)
        self.label5.setFont(font)

        self.layout6 = QHBoxLayout()
        self.title6 = QLabel('追踪帧率: ')
        self.title6.setAlignment(Qt.AlignLeft)
        self.title6.setFont(font)
        self.label6 = QLabel(str(self.track_fps))
        self.label6.setAlignment(Qt.AlignHCenter)
        self.label6.setFont(font)

        self.layout.addWidget(self.title)
        self.layout.addWidget(self.label)
        self.layout2.addWidget(self.title2)
        self.layout2.addWidget(self.label2)
        self.layout3.addWidget(self.title3)
        self.layout3.addWidget(self.label3)
        self.layout4.addWidget(self.title4)
        self.layout4.addWidget(self.label4)
        self.layout5.addWidget(self.title5)
        self.layout5.addWidget(self.label5)
        self.layout6.addWidget(self.title6)
        self.layout6.addWidget(self.label6)

        global_layout = QVBoxLayout()
        global_layout.addLayout(self.layout)
        global_layout.addLayout(self.layout2)
        global_layout.addLayout(self.layout3)
        global_layout.addLayout(self.layout4)
        global_layout.addLayout(self.layout5)
        global_layout.addLayout(self.layout6)

        self.setLayout(global_layout)
    def show_results(self, UAV1_detect, UAV2_detect, UAV1_track, UAV2_track, detect_fps, track_fps):
        self.UAV1_detect = UAV1_detect
        self.UAV2_detect = UAV2_detect
        self.UAV1_track = UAV1_track
        self.UAV2_track = UAV2_track
        self.label.setText(str(self.UAV1_detect))
        self.label2.setText(str(self.UAV2_detect))
        self.label3.setText(str(self.UAV1_track))
        self.label4.setText(str(self.UAV2_track))
        self.label5.setText(str(detect_fps))
        self.label6.setText(str(track_fps))

class Ui_MainWindow(QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        self.timer_camera1 = QtCore.QTimer()
        self.timer_camera2 = QtCore.QTimer()
        self.timer_camera3 = QtCore.QTimer()
        self.timer_camera4 = QtCore.QTimer()
        self.cap = cv2.VideoCapture()

        self.CAM_NUM = 0

        self.__flag_work = 0
        self.x = 0
        self.count = 0
        self.setWindowTitle("多无人机目标追踪仿真平台")
        self.setWindowIcon(QIcon(os.getcwd() + '\\data\\source_image\\Detective.ico'))
        window_pale = QtGui.QPalette()
        window_pale.setBrush(self.backgroundRole(), QtGui.QBrush(
            QtGui.QPixmap(os.getcwd() + '\\data\\source_image\\backgroud.jpg')))
        self.setPalette(window_pale)
        self.setFixedSize(1600, 900)
        self.opt = load_detector_opt()
        self.opt, self.det_stride, self.det_img_size, self.det_cams, self.det_roi_masks, self.detector_model, \
            self.det_device, self.det_half, self.det_names, self.det_is_show_tracks, self.det_label = \
            lodel_detector_model(self.opt)

        self.client_socket = create_unity_connection()
        self.args = load_trackor_opt()
        self.line_thickness = self.args.line_thickness
        self.device = self.args.device
        print("det: ", self.args.det_name)
        self.detector_weights = self.args.det_name
        self.reid_weights = self.args.feat_ext_name
        self.conf_thres = self.args.conf_thres
        self.iou_thres = self.args.iou_thres
        self.cos_thr = self.args.cos_thr
        self.iou_thr = self.args.iou_thr
        self.UAV1_detect = "None"
        self.UAV2_detect = "None"
        self.UAV1_track = "None"
        self.UAV2_track = "None"
        self.detect_fps = "None"
        self.track_fps = "None"
        self.detect1_info, self.detect2_info, self.track1_info, self.track2_info = "None", "None", "None", "None"
        self.args, self.stride, self.img_size, self.normalize, self.cams, self.roi_masks, self.overlap_regions, self.det_model,\
            self.feat_ext_model, self.scene_feat_ext_model, self.trackers, self.device, self.half, self.names, \
            self.is_show_detections, self.is_show_tracks, self.label = load_model(self.args)

        self.button_open_camera = QPushButton(self)
        self.button_open_camera.setText(u'开始目标检测')
        self.button_open_camera.setStyleSheet(''' 
                                     QPushButton
                                     {text-align : center;
                                     background-color : white;
                                     font: bold;
                                     border-color: red;
                                     border-width: 5px;
                                     border-radius: 10px;
                                     padding: 6px;
                                     height : 14px;
                                     border-style: outset;
                                     font : 14px;}
                                     QPushButton:pressed
                                     {text-align : center;
                                     background-color : light gray;
                                     font: bold;
                                     border-color: gray;
                                     border-width: 2px;
                                     border-radius: 10px;
                                     padding: 6px;
                                     height : 14px;
                                     border-style: outset;
                                     font : 16px;}
                                     ''')
        self.button_open_camera.move(10, 20)
        self.button_open_camera.clicked.connect(self.button_open_camera_click)

        self.track_btn = QPushButton(self)
        self.track_btn.setText("开始目标追踪")
        self.track_btn.setStyleSheet(''' 
                                             QPushButton
                                             {text-align : center;
                                             background-color : white;
                                             font: bold;
                                             border-color: red;
                                             border-width: 5px;
                                             border-radius: 10px;
                                             padding: 6px;
                                             height : 14px;
                                             border-style: outset;
                                             font : 14px;}
                                             QPushButton:pressed
                                             {text-align : center;
                                             background-color : light gray;
                                             font: bold;
                                             border-color: gray;
                                             border-width: 2px;
                                             border-radius: 10px;
                                             padding: 6px;
                                             height : 14px;
                                             border-style: outset;
                                             font : 16px;}
                                             ''')
        self.track_btn.move(10, 60)
        self.track_btn.clicked.connect(self.button_open_camera_click1)

        self.open_video = QPushButton(self)
        self.open_video.setText("打开视频")
        self.open_video.setStyleSheet(''' 
                                             QPushButton
                                             {text-align : center;
                                             background-color : white;
                                             font: bold;
                                             border-color: gray;
                                             border-width: 2px;
                                             border-radius: 10px;
                                             padding: 6px;
                                             height : 14px;
                                             border-style: outset;
                                             font : 14px;}
                                             QPushButton:pressed
                                             {text-align : center;
                                             background-color : light gray;
                                             font: bold;
                                             border-color: gray;
                                             border-width: 2px;
                                             border-radius: 10px;
                                             padding: 6px;
                                             height : 14px;
                                             border-style: outset;
                                             font : 14px;}
                                             ''')
        self.open_video.move(10, 160)
        self.open_video.clicked.connect(self.open_video_button)
        print("QPushButton构建")

        self.btn1 = QPushButton(self)
        self.btn1.setText("检测视频文件")
        self.btn1.setStyleSheet(''' 
                                             QPushButton
                                             {text-align : center;
                                             background-color : white;
                                             font: bold;
                                             border-color: gray;
                                             border-width: 2px;
                                             border-radius: 10px;
                                             padding: 6px;
                                             height : 14px;
                                             border-style: outset;
                                             font : 14px;}
                                             QPushButton:pressed
                                             {text-align : center;
                                             background-color : light gray;
                                             font: bold;
                                             border-color: gray;
                                             border-width: 2px;
                                             border-radius: 10px;
                                             padding: 6px;
                                             height : 14px;
                                             border-style: outset;
                                             font : 14px;}
                                             ''')
        self.btn1.move(10, 200)
        self.btn1.clicked.connect(self.detect_video)
        print("QPushButton构建")

        self.btn2 = QPushButton(self)
        self.btn2.setText("返回上一界面")
        self.btn2.setStyleSheet(''' 
                                             QPushButton
                                             {text-align : center;
                                             background-color : white;
                                             font: bold;
                                             border-color: red;
                                             border-width: 5px;
                                             border-radius: 10px;
                                             padding: 6px;
                                             height : 14px;
                                             border-style: outset;
                                             font : 14px;}
                                             QPushButton:pressed
                                             {text-align : center;
                                             background-color : light gray;
                                             font: bold;
                                             border-color: gray;
                                             border-width: 2px;
                                             border-radius: 10px;
                                             padding: 6px;
                                             height : 14px;
                                             border-style: outset;
                                             font : 16px;}
                                             ''')
        self.btn2.move(200, 20)
        self.btn2.clicked.connect(self.back_lastui)
        # 设置检测框厚度菜单和按键
        self.line_thickness_slider = ThicknessSlider()
        self.line_thickness_slider.hide()
        self.btn3 = QPushButton(self)
        self.btn3.setText("设置检测框厚度")
        self.btn3.move(10, 300)
        self.btn3.clicked.connect(self.set_line_thickness_menu)
        # 设置检测框厚度定时器
        self.thickness_timer = QTimer()
        self.thickness_timer.timeout.connect(self.set_line_thickness)
        self.thickness_timer.start(20)
        # 设置检测设备按键和菜单
        self.device_menu = DeviceMenu()
        self.device_menu.hide()
        self.btn4 = QPushButton(self)
        self.btn4.setText("设置检测设备")
        self.btn4.move(10, 350)
        self.btn4.clicked.connect(self.set_device_menu)
        # 设置检测设备定时器
        self.device_timer = QTimer()
        self.device_timer.timeout.connect(self.set_device)
        self.device_timer.start(20)
        # 设置检测模型按键和菜单
        self.detector_model_menu = DetectorModelMenu()
        self.detector_model_menu.hide()
        self.btn5 = QPushButton(self)
        self.btn5.setText("设置检测模型")
        self.btn5.move(10, 400)
        self.btn5.clicked.connect(self.set_detector_model_menu)
        # 设置检测模型定时器
        self.detector_model_timer = QTimer()
        self.detector_model_timer.timeout.connect(self.set_detector_model)
        self.detector_model_timer.start(20)
        # 设置追踪器重识别模型菜单和按键
        self.reid_model_menu = TrackorModelMenu()
        self.reid_model_menu.hide()
        self.btn6 = QPushButton(self)
        self.btn6.setText("设置重识别模型")
        self.btn6.move(10, 450)
        self.btn6.clicked.connect(self.set_reid_model_menu)
        # 设置追踪器重识别模型定时器
        self.reid_model_timer = QTimer()
        self.reid_model_timer.timeout.connect(self.set_reid_model)
        self.reid_model_timer.start(20)

        # 显示参数信息
        self.parameters_InfoDialog = ParamsInfoDialog(self.line_thickness, self.device, self.detector_weights,
                                         self.conf_thres, self.iou_thres, self.cos_thr, self.iou_thr)
        self.parameters_InfoDialog.hide()
        # 设置参数信息菜单和按键
        self.btn8 = QPushButton(self)
        self.btn8.setText("显示参数信息")
        self.btn8.move(10, 550)
        self.btn8.clicked.connect(self.show_parameters_InfoDialog)
        # 显示检测追踪结果
        self.result_InfoDialog = ResultInfoDialog(self.UAV1_detect, self.UAV2_detect, self.UAV1_track, self.UAV2_track,
                                                  self.detect_fps, self.track_fps)
        self.result_InfoDialog.hide()
        # 设置显示结果菜单和按键
        self.btn9 = QPushButton(self)
        self.btn9.setText("显示检测追踪结果")
        self.btn9.move(10, 600)
        self.btn9.clicked.connect(self.show_result_InfoDialog)
        # 设置检测追踪结果定时器
        self.result_timer = QTimer()
        self.result_timer.timeout.connect(self.show_result)
        self.result_timer.start(20)
        # 设置置信度阈值按键
        self.conf_menu = Conf_threshold_Slider()
        self.conf_menu.hide()
        self.btn10 = QPushButton(self)
        self.btn10.setText("设置置信度阈值")
        self.btn10.move(10, 650)
        self.btn10.clicked.connect(self.set_conf_menu)
        # 设置置信度阈值定时器
        self.conf_timer = QTimer()
        self.conf_timer.timeout.connect(self.set_conf_threshold)
        self.conf_timer.start(20)
        # 设置检测器IOU阈值按键
        self.Detector_IOU_menu = Detector_IOU_threshold_Slider()
        self.Detector_IOU_menu.hide()
        self.btn11 = QPushButton(self)
        self.btn11.setText("设置检测器IOU阈值")
        self.btn11.move(10, 700)
        self.btn11.clicked.connect(self.set_Detector_IOU_menu)
        # 设置检测器IOU阈值定时器
        self.Detector_IOU_timer = QTimer()
        self.Detector_IOU_timer.timeout.connect(self.set_Detector_IOU)
        self.Detector_IOU_timer.start(20)
        # 设置追踪器余弦相似度阈值按键
        self.Trackor_cos_menu = Trackor_cos_threshold_Slider()
        self.Trackor_cos_menu.hide()
        self.btn12 = QPushButton(self)
        self.btn12.setText("设置追踪器余弦相似度阈值")
        self.btn12.move(10, 750)
        self.btn12.clicked.connect(self.set_Trackor_cos_menu)
        # 设置追踪器余弦相似度阈值定时器
        self.Trackor_cos_timer = QTimer()
        self.Trackor_cos_timer.timeout.connect(self.set_Trackor_cos)
        self.Trackor_cos_timer.start(20)
        # 设置MAX_IOU_DISTANCE按键
        self.Trackor_IOU_menu = Trackor_IOU_threshold_Slider()
        self.Trackor_IOU_menu.hide()
        self.btn13 = QPushButton(self)
        self.btn13.setText("设置追踪器IOU阈值")
        self.btn13.move(10, 800)
        self.btn13.clicked.connect(self.set_Trackor_IOU_menu)
        # 设置MAX_IOU_DISTANCE定时器
        self.Trackor_IOU_timer = QTimer()
        self.Trackor_IOU_timer.timeout.connect(self.set_Trackor_IOU)
        self.Trackor_IOU_timer.start(20)

        # unity检测追踪界面显示
        self.label_show_camera = QLabel(self)
        self.label_move = QLabel()
        self.label_move.setFixedSize(100, 100)
        self.label_show_camera.setFixedSize(640, 640)
        self.label_show_camera.setAutoFillBackground(True)
        self.label_show_camera.move(200, 80)
        self.label_show_camera.setStyleSheet("QLabel{background:#F5F5DC;}"
                                             "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                             )
        self.label_show_camera1 = QLabel(self)
        self.label_show_camera1.setFixedSize(640, 640)
        self.label_show_camera1.setAutoFillBackground(True)
        self.label_show_camera1.move(920, 80)
        self.label_show_camera1.setStyleSheet("QLabel{background:#F5F5DC;}"
                                              "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                              )

        self.timer_camera1.timeout.connect(self.show_camera)
        self.timer_camera2.timeout.connect(self.show_camera1)
        self.timer_camera4.timeout.connect(self.show_camera2)
        self.timer_camera4.timeout.connect(self.show_camera3)
        self.clicked = False

        self.frame_s = 3

    # 返回上一个界面
    def back_lastui(self):
        self.timer_camera1.stop()
        self.cap.release()
        self.label_show_camera.clear()
        self.timer_camera2.stop()

        self.label_show_camera1.clear()
        cam_t.close()
        ui_p.show()

    # 生成检测框厚度菜单
    def set_line_thickness_menu(self):
        self.line_thickness_slider.show()

    # 设置检测框厚度
    def set_line_thickness(self):
        if self.line_thickness_slider.confirm:
            self.line_thickness = self.line_thickness_slider.decimal_slider.value()
            self.args.line_thickness = self.line_thickness
            self.args, self.stride, self.img_size, self.normalize, self.cams, self.roi_masks, self.overlap_regions, self.det_model, \
                self.feat_ext_model, self.scene_feat_ext_model, self.trackers, self.device, self.half, self.names, \
                self.is_show_detections, self.is_show_tracks, self.label = load_model(self.args)
            self.line_thickness_slider.confirm = False
            self.line_thickness_slider.hide()
            self.line_thickness_slider.close()

    # 生成检测设备菜单
    def set_device_menu(self):
        self.device_menu.show()

    # 设置检测设备
    def set_device(self):
        if self.device_menu.confirm:
            self.device = self.device_menu.device
            self.args.device = self.device
            self.args, self.stride, self.img_size, self.normalize, self.cams, self.roi_masks, self.overlap_regions, self.det_model, \
                self.feat_ext_model, self.scene_feat_ext_model, self.trackers, self.device, self.half, self.names, \
                self.is_show_detections, self.is_show_tracks, self.label = load_model(self.args)
            self.device_menu.confirm = False
            self.device_menu.hide()
            self.device_menu.close()

    # 生成检测模型菜单
    def set_detector_model_menu(self):
        self.detector_model_menu.show()

    # 设置检测模型
    def set_detector_model(self):
        if self.detector_model_menu.confirm:
            self.detector_weights = self.detector_model_menu.detector_weight
            self.args.det_name = self.detector_weights
            self.args, self.stride, self.img_size, self.normalize, self.cams, self.roi_masks, self.overlap_regions, self.det_model, \
                self.feat_ext_model, self.scene_feat_ext_model, self.trackers, self.device, self.half, self.names, \
                self.is_show_detections, self.is_show_tracks, self.label = load_model(self.args)
            self.detector_model_menu.confirm = False
            self.detector_model_menu.hide()
            self.detector_model_menu.close()

    # 生成追踪模型菜单
    def set_reid_model_menu(self):
        self.reid_model_menu.show()

    # 设置追踪模型
    def set_reid_model(self):
        if self.reid_model_menu.confirm:
            self.reid_weights = self.reid_model_menu.trackor_weight
            self.args.feat_ext_name = self.reid_weights
            self.args, self.stride, self.img_size, self.normalize, self.cams, self.roi_masks, self.overlap_regions, self.det_model, \
                self.feat_ext_model, self.scene_feat_ext_model, self.trackers, self.device, self.half, self.names, \
                self.is_show_detections, self.is_show_tracks, self.label = load_model(self.args)
            self.reid_model_menu.confirm = False
            self.reid_model_menu.hide()
            self.reid_model_menu.close()

    # 生成置信度阈值菜单
    def set_conf_menu(self):
        self.conf_menu.show()

    # 设置置信度阈值
    def set_conf_threshold(self):
        if self.conf_menu.confirm:
            self.conf_menu = self.conf_menu.MAX_DIST_value
            self.args.conf_thres = self.conf_thres
            self.args, self.stride, self.img_size, self.normalize, self.cams, self.roi_masks, self.overlap_regions, self.det_model, \
                self.feat_ext_model, self.scene_feat_ext_model, self.trackers, self.device, self.half, self.names, \
                self.is_show_detections, self.is_show_tracks, self.label = load_model(self.args)
            self.conf_menu.confirm = False
            self.conf_menu.hide()
            self.conf_menu.close()

    # 生成检测器IOU阈值菜单
    def set_Detector_IOU_menu(self):
        self.Detector_IOU_menu.show()

    # 设置检测器IOU阈值
    def set_Detector_IOU(self):
        if self.Detector_IOU_menu.confirm:
            self.iou_thres = self.Detector_IOU_menu.MIN_CONFIDENCE_value
            self.args.iou_thres = self.iou_thres
            self.args, self.stride, self.img_size, self.normalize, self.cams, self.roi_masks, self.overlap_regions, self.det_model, \
                self.feat_ext_model, self.scene_feat_ext_model, self.trackers, self.device, self.half, self.names, \
                self.is_show_detections, self.is_show_tracks, self.label = load_model(self.args)
            self.Detector_IOU_menu.confirm = False
            self.Detector_IOU_menu.hide()
            self.Detector_IOU_menu.close()

    # 生成追踪器余弦相似度阈值菜单
    def set_Trackor_cos_menu(self):
        self.Trackor_cos_menu.show()

    # 设置追踪器余弦相似度阈值
    def set_Trackor_cos(self):
        if self.Trackor_cos_menu.confirm:
            self.cos_thr = self.Trackor_cos_menu.NMS_MAX_OVERLAP_value
            self.args.cos_thr = self.cos_thr
            self.args, self.stride, self.img_size, self.normalize, self.cams, self.roi_masks, self.overlap_regions, self.det_model, \
                self.feat_ext_model, self.scene_feat_ext_model, self.trackers, self.device, self.half, self.names, \
                self.is_show_detections, self.is_show_tracks, self.label = load_model(self.args)
            self.Trackor_cos_menu.confirm = False
            self.Trackor_cos_menu.hide()
            self.Trackor_cos_menu.close()

    # 生成追踪器IOU阈值菜单
    def set_Trackor_IOU_menu(self):
        self.Trackor_IOU_menu.show()

    # 设置追踪器IOU阈值
    def set_Trackor_IOU(self):
        if self.Trackor_IOU_menu.confirm:
            self.iou_thr = self.Trackor_IOU_menu.MAX_IOU_DISTANCE_value
            self.args.iou_thr = self.iou_thr
            self.args, self.stride, self.img_size, self.normalize, self.cams, self.roi_masks, self.overlap_regions, self.det_model, \
                self.feat_ext_model, self.scene_feat_ext_model, self.trackers, self.device, self.half, self.names, \
                self.is_show_detections, self.is_show_tracks, self.label = load_model(self.args)
            self.Trackor_IOU_menu.confirm = False
            self.Trackor_IOU_menu.hide()
            self.Trackor_IOU_menu.close()


    # 显示参数信息
    def show_parameters_InfoDialog(self):
        self.parameters_InfoDialog.line_thickness = self.line_thickness
        self.parameters_InfoDialog.device = self.device
        self.parameters_InfoDialog.detector_model = self.detector_weights
        self.parameters_InfoDialog.conf_thres = self.conf_thres
        self.parameters_InfoDialog.iou_thres = self.iou_thres
        self.parameters_InfoDialog.cos_thr = self.cos_thr
        self.parameters_InfoDialog.iou_thr = self.iou_thr
        self.parameters_InfoDialog.show_params(self.line_thickness, self.device, self.detector_weights,
                                self.conf_thres, self.iou_thres, self.cos_thr, self.iou_thr)
        self.parameters_InfoDialog.show()

    # 显示结果信息菜单
    def show_result_InfoDialog(self):
        self.result_InfoDialog.show()

    # 显示结果信息
    def show_result(self):
        self.result_InfoDialog.UAV1_detect = self.detect1_info
        self.result_InfoDialog.UAV2_detect = self.detect2_info
        self.result_InfoDialog.UAV1_track = self.track1_info
        self.result_InfoDialog.UAV2_track = self.track2_info
        self.result_InfoDialog.detect_fps = self.detect_fps
        self.result_InfoDialog.track_fps = self.track_fps

        self.result_InfoDialog.show_results(self.detect1_info, self.detect2_info, self.track1_info, self.track2_info,
                                            self.detect_fps, self.track_fps)



    '''摄像头'''

    def button_open_camera_click(self):
        if self.timer_camera1.isActive() == False:
            self.drone1_img, self.drone2_img = receive_unity_data(self.client_socket)
            if self.drone1_img is None and self.drone2_img is None:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检查unity是否连接",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera1.start(100)
                self.timer_camera2.stop()
                self.button_open_camera.setText(u'关闭目标检测')
                self.track_btn.setText(u'开始目标追踪')
        else:
            self.timer_camera1.stop()
            self.label_show_camera.clear()
            self.label_show_camera1.clear()
            self.button_open_camera.setText(u'开始目标检测')

    # 显示目标检测结果
    def show_camera(self):
        self.is_show_tracks = False
        self.is_show_detections = True
        self.drone1_image, self.drone2_image, self.label, self.detect1_info, self.detect2_info, \
        self.track1_info, self.track2_info, self.detect_fps, self.track_fps = main_track_process(self.args, self.client_socket, self.stride,
                                                                 self.img_size, self.normalize, self.cams,
                                                                 self.roi_masks,
                                                                 self.overlap_regions, self.det_model,
                                                                 self.feat_ext_model,
                                                                 self.scene_feat_ext_model, self.trackers, self.device,
                                                                 self.half, self.names, self.is_show_detections,
                                                                 self.is_show_tracks, self.label)

        width = self.drone1_image.shape[1]
        height = self.drone1_image.shape[0]

        # 设置新的图片分辨率框架
        width_new = 640
        height_new = 640

        # 判断图片的长宽比率
        if width / height >= width_new / height_new:
            show = cv2.resize(self.drone1_image, (width_new, height_new))
            show2 = cv2.resize(self.drone2_image, (width_new, height_new))
        else:
            show = cv2.resize(self.drone1_image, (width_new, height_new))
            show2 = cv2.resize(self.drone2_image, (width_new, height_new))


        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        show2 = cv2.cvtColor(show2, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], 3 * show.shape[1], QtGui.QImage.Format_RGB888)
        showImage2 = QtGui.QImage(show2.data, show2.shape[1], show2.shape[0], 3 * show2.shape[1], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.label_show_camera1.setPixmap(QtGui.QPixmap.fromImage(showImage2))

    def button_open_camera_click1(self):
        if self.timer_camera2.isActive() == False:
            self.drone1_img, self.drone2_img = receive_unity_data(self.client_socket)
            if self.drone1_img is None and self.drone2_img is None:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检查unity是否连接",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)

            else:
                self.timer_camera2.start(2000)
                self.timer_camera1.stop()
                self.track_btn.setText(u'关闭目标追踪')
                self.button_open_camera.setText(u'开始目标检测')
        else:
            self.timer_camera2.stop()
            self.label_show_camera1.clear()
            self.label_show_camera.clear()
            self.track_btn.setText(u'开始目标追踪')

    # 显示追踪结果
    def show_camera1(self):

        self.is_show_tracks = True
        self.is_show_detections = False
        self.drone1_image, self.drone2_image, self.label, self.detect1_info, self.detect2_info, \
        self.track1_info, self.track2_info, self.detect_fps, self.track_fps = main_track_process(self.args, self.client_socket, self.stride,
                                                           self.img_size, self.normalize, self.cams, self.roi_masks,
                                                           self.overlap_regions, self.det_model, self.feat_ext_model,
                                                           self.scene_feat_ext_model, self.trackers, self.device,
                                                           self.half, self.names, self.is_show_detections,
                                                           self.is_show_tracks, self.label)

        width = self.drone1_image.shape[1]
        height = self.drone1_image.shape[0]

        # 设置新的图片分辨率框架
        width_new = 640
        height_new = 640

        # 判断图片的长宽比率
        if width / height >= width_new / height_new:
            show = cv2.resize(self.drone1_image, (width_new, height_new))
            show2 = cv2.resize(self.drone2_image, (width_new, height_new))
        else:
            show = cv2.resize(self.drone1_image, (width_new, height_new))
            show2 = cv2.resize(self.drone2_image, (width_new, height_new))

        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        show2 = cv2.cvtColor(show2, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], 3 * show.shape[1], QtGui.QImage.Format_RGB888)
        showImage2 = QtGui.QImage(show2.data, show2.shape[1], show2.shape[0], 3 * show2.shape[1],
                                  QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.label_show_camera1.setPixmap(QtGui.QPixmap.fromImage(showImage2))

        if self.label == 'debug':
            print("debug")


    '''视频检测'''

    def open_video_button(self):

        if self.timer_camera4.isActive() == False:

            imgName, imgType = QFileDialog.getOpenFileName(self, "打开视频", "", "*.mp4;;*.AVI;;*.rmvb;;All Files(*)")

            self.cap_video = cv2.VideoCapture(imgName)

            flag = self.cap_video.isOpened()

            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"未检测到视频文件",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)

            else:
                self.show_camera2()
                self.open_video.setText(u'关闭视频')
        else:
            self.cap_video.release()
            self.label_show_camera.clear()
            self.timer_camera4.stop()
            self.frame_s = 3
            self.label_show_camera1.clear()
            self.open_video.setText(u'打开视频')

    def detect_video(self):

        if self.timer_camera4.isActive() == False:
            flag = self.cap_video.isOpened()
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"未检测到视频文件",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)

            else:
                self.timer_camera4.start(30)

        else:
            self.timer_camera4.stop()
            self.cap_video.release()
            self.label_show_camera1.clear()

    # 显示原始视频
    def show_camera2(self):

        # 抽帧
        length = int(self.cap_video.get(cv2.CAP_PROP_FRAME_COUNT))  # 抽帧
        print(self.frame_s, length)  # 抽帧
        flag, self.image1 = self.cap_video.read()  # image1是视频的
        if flag == True:
            if self.frame_s % 3 == 0:  # 抽帧

                dir_path = os.getcwd()
                camera_source = dir_path + "\\data\\test\\video.jpg"

                cv2.imwrite(camera_source, self.image1)

                width = self.image1.shape[1]
                height = self.image1.shape[0]

                # 设置新的图片分辨率框架
                width_new = 700
                height_new = 500

                # 判断图片的长宽比率
                if width / height >= width_new / height_new:

                    show = cv2.resize(self.image1, (width_new, height_new))
                else:

                    show = cv2.resize(self.image1, (width_new, height_new))

                show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)

                showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], 3 * show.shape[1],
                                         QtGui.QImage.Format_RGB888)

                self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            self.cap_video.release()
            self.label_show_camera.clear()
            self.timer_camera4.stop()

            self.label_show_camera1.clear()
            self.open_video.setText(u'打开视频')

    # 显示视频检测结果
    def show_camera3(self):

        flag, self.image1 = self.cap_video.read()
        self.frame_s += 1
        if flag == True:
            if self.frame_s % 3 == 0:  # 抽帧

                dir_path = os.getcwd()
                camera_source = dir_path + "\\data\\test\\video.jpg"

                cv2.imwrite(camera_source, self.image1)
                im0, label = main_detect_process(self.opt, self.det_stride, self.det_img_size, self.det_cams, self.det_roi_masks,
                                    self.detector_model, self.det_device, self.det_half, self.det_names, self.det_is_show_tracks,
                                    self.det_label, camera_source)
                if label == 'debug':
                    print("labelkong")

                width = im0.shape[1]
                height = im0.shape[0]

                # 设置新的图片分辨率框架
                width_new = 700
                height_new = 500

                # 判断图片的长宽比率
                if width / height >= width_new / height_new:

                    show = cv2.resize(im0, (width_new, height_new))
                else:

                    show = cv2.resize(im0, (width_new, height_new))

                im0 = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)

                showImage = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1], QtGui.QImage.Format_RGB888)

                self.label_show_camera1.setPixmap(QtGui.QPixmap.fromImage(showImage))


class picture(QWidget):

    def __init__(self):
        super(picture, self).__init__()

        self.str_name = '0'
        self.opt = load_detector_opt()
        self.opt, self.det_stride, self.det_img_size, self.det_cams, self.det_roi_masks, self.detector_model, \
            self.det_device, self.det_half, self.det_names, self.det_is_show_tracks, self.det_label = \
            lodel_detector_model(self.opt)
        self.resize(1600, 900)
        self.setWindowIcon(QIcon(os.getcwd() + '\\data\\source_image\\Detective.ico'))
        self.setWindowTitle("无人机目标检测和追踪仿真平台")

        window_pale = QtGui.QPalette()
        window_pale.setBrush(self.backgroundRole(), QtGui.QBrush(
            QtGui.QPixmap(os.getcwd() + '\\data\\source_image\\backgroud.jpg')))
        self.setPalette(window_pale)

        camera_or_video_save_path = 'data\\test'
        if not os.path.exists(camera_or_video_save_path):
            os.makedirs(camera_or_video_save_path)
        self.label1 = QLabel(self)
        self.label1.setText("   待检测图片")
        self.label1.setFixedSize(640, 640)
        self.label1.move(160, 80)

        self.label1.setStyleSheet("QLabel{background:#7A6969;}"
                                  "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}"
                                  )
        self.label2 = QLabel(self)
        self.label2.setText("   检测结果")
        self.label2.setFixedSize(640, 640)
        self.label2.move(880, 80)

        self.label2.setStyleSheet("QLabel{background:#7A6969;}"
                                  "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}"
                                  )

        self.label3 = QLabel(self)
        self.label3.setText("")
        self.label3.move(1200, 620)
        self.label3.setStyleSheet("font-size:20px;")
        self.label3.adjustSize()

        btn = QPushButton(self)
        btn.setText("打开图片")
        btn.setStyleSheet(''' 
                                                     QPushButton
                                                     {text-align : center;
                                                     background-color : white;
                                                     font: bold;
                                                     border-color: gray;
                                                     border-width: 2px;
                                                     border-radius: 10px;
                                                     padding: 6px;
                                                     height : 14px;
                                                     border-style: outset;
                                                     font : 14px;}
                                                     QPushButton:pressed
                                                     {text-align : center;
                                                     background-color : light gray;
                                                     font: bold;
                                                     border-color: gray;
                                                     border-width: 2px;
                                                     border-radius: 10px;
                                                     padding: 6px;
                                                     height : 14px;
                                                     border-style: outset;
                                                     font : 14px;}
                                                     ''')
        btn.move(10, 30)
        btn.clicked.connect(self.openimage)

        btn1 = QPushButton(self)
        btn1.setText("检测图片")
        btn1.setStyleSheet(''' 
                                                     QPushButton
                                                     {text-align : center;
                                                     background-color : white;
                                                     font: bold;
                                                     border-color: gray;
                                                     border-width: 2px;
                                                     border-radius: 10px;
                                                     padding: 6px;
                                                     height : 14px;
                                                     border-style: outset;
                                                     font : 14px;}
                                                     QPushButton:pressed
                                                     {text-align : center;
                                                     background-color : light gray;
                                                     font: bold;
                                                     border-color: gray;
                                                     border-width: 2px;
                                                     border-radius: 10px;
                                                     padding: 6px;
                                                     height : 14px;
                                                     border-style: outset;
                                                     font : 14px;}
                                                     ''')
        btn1.move(10, 80)
        btn1.clicked.connect(self.button1_test)

        btn3 = QPushButton(self)
        btn3.setText("仿真和视频检测")
        btn3.setStyleSheet(''' 
                                                     QPushButton
                                                     {text-align : center;
                                                     background-color : white;
                                                     font: bold;
                                                     border-color: red;
                                                     border-width: 5px;
                                                     border-radius: 10px;
                                                     padding: 6px;
                                                     height : 14px;
                                                     border-style: outset;
                                                     font : 14px;}
                                                     QPushButton:pressed
                                                     {text-align : center;
                                                     background-color : light gray;
                                                     font: bold;
                                                     border-color: gray;
                                                     border-width: 2px;
                                                     border-radius: 10px;
                                                     padding: 6px;
                                                     height : 14px;
                                                     border-style: outset;
                                                     font : 16px;}
                                                     ''')
        btn3.move(160, 30)
        btn3.clicked.connect(self.camera_find)

        self.imgname1 = '0'

    # 切换界面，打开仿真界面，关闭图片界面
    def camera_find(self):
        ui_p.close()
        cam_t.show()

    # 判断图片是否包含中文
    def is_has_chineese(self, img):
        # 判断图片是否包含中文
        for i in range(len(img)):
            if '\u4e00' <= img[i] <= '\u9fa5':
                return True
        return False

    # 打开图片
    def openimage(self):

        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.png;;*.jpg;;All Files(*)")
        is_has_chineese = self.is_has_chineese(imgName)
        if imgName != '' and is_has_chineese == False:
            self.imgname1 = imgName
            im0 = cv2.imread(imgName)

            width = im0.shape[1]
            height = im0.shape[0]

            # 设置新的图片分辨率框架
            width_new = 640
            height_new = 640

            # 判断图片的长宽比率
            if width / height >= width_new / height_new:

                show = cv2.resize(im0, (width_new, int(height * width_new / width)))
            else:

                show = cv2.resize(im0, (int(width * height_new / height), height_new))

            im0 = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
            showImage = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1], QtGui.QImage.Format_RGB888)
            self.label1.setPixmap(QtGui.QPixmap.fromImage(showImage))

        else:
            QMessageBox.information(self, '错误', '图片名中包含中文', QMessageBox.Yes, QMessageBox.Yes)

    # 检测图片
    def button1_test(self):

        if self.imgname1 != '0':
            QApplication.processEvents()
            im0, label = main_detect_process(self.opt, self.det_stride, self.det_img_size, self.det_cams,
                                             self.det_roi_masks,
                                             self.detector_model, self.det_device, self.det_half, self.det_names,
                                             self.det_is_show_tracks,
                                             self.det_label, self.imgname1)

            QApplication.processEvents()

            width = im0.shape[1]
            height = im0.shape[0]

            # 设置新的图片分辨率框架
            width_new = 640
            height_new = 640

            # 判断图片的长宽比率
            if width / height >= width_new / height_new:

                show = cv2.resize(im0, (width_new, height_new))
            else:

                show = cv2.resize(im0, (width_new, height_new))
            im0 = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
            image_name = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1], QtGui.QImage.Format_RGB888)
            self.label2.setPixmap(QtGui.QPixmap.fromImage(image_name))
        else:
            QMessageBox.information(self, '错误', '请先选择一个图片文件', QMessageBox.Yes, QMessageBox.Yes)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    splash = QSplashScreen(QPixmap(".\\data\\source_image\\logo.png"))
    # 设置画面中的文字的字体
    splash.setFont(QFont('Microsoft YaHei UI', 12))
    # 显示画面
    splash.show()
    # 显示信息
    splash.showMessage("程序初始化中... 0%", QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom, QtCore.Qt.black)
    time.sleep(0.3)

    splash.showMessage("正在加载模型配置文件...60%", QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom, QtCore.Qt.black)
    cam_t = Ui_MainWindow()
    splash.showMessage("正在加载模型配置文件...100%", QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom, QtCore.Qt.black)

    ui_p = picture()
    ui_p.show()
    splash.close()

    sys.exit(app.exec_())
