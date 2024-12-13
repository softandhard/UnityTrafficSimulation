import torch
import torch.nn as nn
import torch.nn.functional as F
from online_MTMC.models.deepsort_model import Net, AttentionNet


class FeatureExtractor(nn.Module):
    def __init__(self, model_name, avg_type, pretrained_dir):
        super(FeatureExtractor, self).__init__()
        self.act = nn.ReLU(inplace=True)

        # Construct backbone network
        self.model_name = model_name
        if model_name == 'net':
            self.backbone = Net(num_classes=960, reid=True)
        elif model_name == 'attention_net':
            self.backbone = AttentionNet(num_classes=960, reid=True)

        # Global Average Pooling, Dropout
        self.avg = nn.AvgPool2d((8, 4), 1) if avg_type == 'avg' else self.gem
        self.classifier = nn.Sequential(
            nn.Linear(5120, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        print("model=", '%s_%s.t7' % (model_name, avg_type))
        # Load preliminary weight
        self.load_param(pretrained_dir + '%s_%s.t7' % (model_name, avg_type))

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

        feat_tri = feat_tri.view(feat_tri.size(0), -1)

        # BNNeck
        feat_infer = self.classifier(feat_tri)

        feat_infer = feat_infer.view(feat_infer.size(0), -1)

        return feat_infer
