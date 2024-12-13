import torch
import torch.nn as nn
import torch.nn.functional as F
from online_MTMC.models.se_resnet_ibn import se_resnet50_ibn_a, se_resnet101_ibn_a
from online_MTMC.models.efficient_kan import KAN

class FeatureExtractor(nn.Module):
    def __init__(self, model_name, avg_type, pretrained_dir):
        super(FeatureExtractor, self).__init__()
        self.act = nn.ReLU(inplace=True)

        # Construct backbone network
        self.model_name = model_name
        if model_name == 'se_resnet50_ibn_a':
            self.backbone = se_resnet50_ibn_a()
        elif model_name == 'se_resnet101_ibn_a':
            self.backbone = se_resnet101_ibn_a()

        # Global Average Pooling, Dropout
        self.avg = nn.AdaptiveAvgPool2d(1) if avg_type == 'gap' else self.gem
        self.drop = nn.Dropout(0.5)

        # BNNeck
        self.bnn = nn.BatchNorm2d(2048)
        nn.init.constant_(self.bnn.weight, 1)
        nn.init.constant_(self.bnn.bias, 0)
        self.bnn.bias.requires_grad_(False)

        self.kan = KAN([512 * 4, 64, 960])
        print("model=", '%s_%s_120.t7' % (model_name, avg_type))
        # Load preliminary weight
        self.load_param(pretrained_dir + '%s_%s_120.t7' % (model_name, avg_type))


    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model']
        for param_name in param_dict:
            if 'fc_ide' in param_name:
                continue
            self.state_dict()[param_name.replace('module.', '')].copy_(param_dict[param_name])

    def forward(self, patch):
        # Extract appearance feature
        feat_tri = self.avg(self.backbone(patch)).half()
        # feat_tri = self.act(feat_tri)

        # BNNeck
        feat_infer = self.bnn(self.drop(feat_tri))
        feat_infer = feat_infer.view(feat_infer.size(0), -1)
        # feat_infer = self.kan(feat_infer)

        return feat_infer
