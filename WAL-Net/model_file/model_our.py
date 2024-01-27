import torch
import torch.nn as nn
from model_file.resnest import resnest_our
import torch.nn.functional as F
import torchvision.ops as ops


class Aggregation(nn.Module):
    def __init__(self, in_channels):
        super(Aggregation, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels + in_channels * 2, in_channels * 2, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels * 2 + in_channels * 4, in_channels * 4, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels * 4 + in_channels * 8, in_channels * 8, kernel_size=1, bias=False)

    def forward(self, g1, g2, g3, x):
        g1 = self.conv1(g1)

        g2 = torch.cat([g1, g2], dim=1)
        g2 = self.conv2(g2)
        g2 = self.conv3(g2)

        g3 = torch.cat([g2, g3], dim=1)
        g3 = self.conv4(g3)
        g3 = self.conv5(g3)

        x = torch.cat([g3, x], dim=1)
        x = self.conv6(x)
        return x


class DeepLabV3Plus(nn.Module):
    def __init__(self, model_name, lae_method='crop&dilate', num_class_seg=2, num_class_cla=3):
        super(DeepLabV3Plus, self).__init__()

        if model_name in 'resnest_our':
            self.backbone = resnest_our.resnest50(num_classes=num_class_cla)

        low_channels = 256
        high_channels = 2048

        self.lae_method = lae_method

        self.head = ASPPModule(high_channels, [6, 12, 18])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, num_class_seg, 1, bias=True)

        self.aggregation = Aggregation(low_channels)

    def forward(self, x, epoch=20):
        h, w = x.shape[-2:]

        feats, g1, g2, g3 = self.backbone.forward_features(x)
        c1, c4 = feats[0], feats[-1]
        out_s = self._decode(c1, c4)
        # Region Weight Module
        if epoch >= 20:
            part1 = region_crop_module(g1[0], out_s, lae_method=self.lae_method)
            part2 = region_crop_module(g2[0], out_s, lae_method=self.lae_method)
            part3 = region_crop_module(g3[0], out_s, lae_method=self.lae_method)
            part4 = region_crop_module(c4, out_s, lae_method=self.lae_method)
        else:
            part1 = g1[0]
            part2 = g2[0]
            part3 = g3[0]
            part4 = c4
        out_c = self.backbone.forward_head(part1, part2, part3, part4)

        return [out_c, out_s], g1[-1], g2[-1], g3[-1]

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)
        c1 = self.reduce(c1)
        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)
        out = self.classifier(feature)

        return out


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)


def region_crop_module(x, seg, lae_method):
    base_ratio = x.shape[-1] / 7
    operated_x = torch.zeros(x.shape).cuda()

    roi_mask = torch.unsqueeze(torch.softmax(seg, 1)[:, 1, :, :], dim=1)
    roi_mask_temp = torch.clone(roi_mask)
    roi_mask_temp[roi_mask_temp < 0.5] = 0
    roi_mask_temp[roi_mask_temp >= 0.5] = 1.0
    roi_mask_temp = F.interpolate(roi_mask_temp, size=(x.shape[-2], x.shape[-1]), mode='nearest')
    if lae_method == 'zero':
        operated_x = x * roi_mask_temp
        return operated_x
    elif lae_method == 'rwm':
        softmax_weight_map = torch.softmax(seg, 1)
        softmax_weight_map = softmax_weight_map[:, 1, :, :]
        softmax_weight_map = torch.unsqueeze(softmax_weight_map, dim=1)
        x = x * F.interpolate(softmax_weight_map, size=(x.shape[-2], x.shape[-1]), mode='nearest')
        return x
    elif lae_method == 'zero_dilate':
        x = x * roi_mask_temp

    for index in range(roi_mask_temp.shape[0]):
        coords = torch.where(roi_mask_temp[index][0] == 1)
        try:
            y1 = int(coords[0].min())
            x1 = int(coords[1].min())
            y2 = int(coords[0].max())
            x2 = int(coords[1].max())
        except RuntimeError:
            operated_x[index] = x[index]
            continue
        if lae_method == 'crop&dilate':
            # dilate
            new_y1 = int(y1 - base_ratio) if int(y1 - base_ratio) >= 0 else 0
            new_x1 = int(x1 - base_ratio) if int(x1 - base_ratio) >= 0 else 0
            new_y2 = int(y2 + base_ratio) if int(y2 + base_ratio) <= roi_mask.shape[-1] else roi_mask.shape[-1]
            new_x2 = int(x2 + base_ratio) if int(x2 + base_ratio) <= roi_mask.shape[-1] else roi_mask.shape[-1]
        else:
            new_x1 = x1
            new_x2 = x2
            new_y1 = y1
            new_y2 = y2
        resized_x = x[index, :, new_y1:new_y2 + 1, new_x1:new_x2 + 1]
        resized_x = torch.unsqueeze(resized_x, dim=0)
        resized_x = F.interpolate(resized_x, size=(x.shape[-2], x.shape[-1]), mode='nearest')
        operated_x[index] = resized_x[0]
    return operated_x
