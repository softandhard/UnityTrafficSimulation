import torch
import torch.nn as nn
import torch.nn.functional as F
from train_feat_ext.nets.resnext_ibn import resnext50_ibn_a, resnext101_ibn_a
import train_feat_ext.config_resnext as config
from online_MTMC.models.efficient_kan import KAN

class Estimator(nn.Module):
    def __init__(self, model_name, avg_type, pretrained_dir):
        super(Estimator, self).__init__()
        self.act = nn.ReLU(inplace=True)

        # Construct backbone network

        self.model_name = model_name
        if model_name == 'resnext50_ibn_a':
            self.backbone = resnext50_ibn_a()
        elif model_name == 'resnext101_ibn_a':
            self.backbone = resnext101_ibn_a()

        # Global Average Pooling, Dropout
        self.avg = nn.AdaptiveAvgPool2d(1) if avg_type == 'gap' else self.gem
        self.drop = nn.Dropout(0.5)

        self.kan = KAN([512 * 4, 64, 960])

        # BNNeck
        self.bnn = nn.BatchNorm2d(2048)
        nn.init.constant_(self.bnn.weight, 1)
        nn.init.constant_(self.bnn.bias, 0)
        self.bnn.bias.requires_grad_(False)

        # IDE
        self.fc_ide = nn.Linear(2048, config.num_ide_class)

        # Load preliminary weight
        self.load_param(pretrained_dir)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def load_param(self, trained_path):
        print("path= ", trained_path)
        param_dict = torch.load(trained_path)['model']
        for param_name in param_dict:
            if 'fc_ide' in param_name:
                continue
            self.state_dict()[param_name.replace('module.', '')].copy_(param_dict[param_name])

    def forward(self, patch):
        # Extract appearance feature
        feat_tri = self.avg(self.backbone(patch))

        # BNNeck
        feat_infer = self.bnn(self.drop(feat_tri))

        # IDE
        feat_ide = feat_infer.view(feat_infer.size(0), -1)
        ide = self.fc_ide(feat_ide)

        return feat_tri, feat_infer, ide
