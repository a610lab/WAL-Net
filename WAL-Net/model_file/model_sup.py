import torch
import torch.nn as nn
from model_file.resnest import resnest
import timm


class Classifier(nn.Module):
    def __init__(self, model_name):
        super(Classifier, self).__init__()
        if model_name in 'rexnet':
            self.backbone = timm.create_model('rexnet_100', num_classes=3)
        elif model_name in 'resnet':
            self.backbone = timm.create_model('resnet50d', num_classes=3)
        elif model_name in 'densenet':
            self.backbone = timm.create_model('densenet121', num_classes=3)
        elif model_name in 'swinv2':
            self.backbone = timm.create_model('swinv2_tiny_window16_256', num_classes=3)
        elif model_name in 'convnext':
            self.backbone = timm.create_model('convnextv2_base', num_classes=3)
        elif model_name in 'sequencer':
            self.backbone = timm.create_model('sequencer2d_s', num_classes=3)
        elif model_name in 'res2net':
            self.backbone = timm.create_model('res2net50d', num_classes=3)
        elif model_name in 'repvit':
            self.backbone = timm.create_model('repvit_m1', num_classes=3)
        elif model_name in 'dpn':
            self.backbone = timm.create_model('dpn98', num_classes=3)
        elif model_name in 'resnest':
            self.backbone = resnest.resnest50(num_classes=3)

    def forward(self, x):
        y = self.backbone.forward(x)
        return [y, 0], 0

