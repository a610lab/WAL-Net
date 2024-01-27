import math
import sys

import numpy as np
import torch
from skimage.segmentation import slic, felzenszwalb
import torch.nn.functional as F
from tqdm import tqdm


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))[0]
        pred_classes = torch.max(pred[0], dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred[0], labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch + 1,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def train_one_epoch_our(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    accu_cla_loss = torch.zeros(1).to(device)  # 累计分类损失
    accu_seg_loss = torch.zeros(1).to(device)  # 累计分割损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred, att1, att2, att3 = model(images.to(device), epoch)

        # Get Segmentation Map
        att2 = F.interpolate(att2, size=(att1.shape[-2], att1.shape[-1]), mode='nearest')
        att3 = F.interpolate(att3, size=(att1.shape[-2], att1.shape[-1]), mode='nearest')
        att_aggregation = map_aggregation(att1, att2, att3)
        seg_target = get_segmentation_map(images, att_aggregation)

        pred_classes = torch.max(pred[0], dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss_cla = loss_function(pred[0], labels.to(device))
        loss_seg = loss_function(pred[1], seg_target.type(torch.LongTensor).to(device))
        loss = loss_cla + loss_seg
        loss.backward()
        accu_cla_loss += loss_cla.detach()
        accu_seg_loss += loss_seg.detach()

        data_loader.desc = "[train epoch {}] cla loss: {:.3f}, seg loss: {:.3f}, acc: {:.3f}".format(epoch + 1,
                                                                                                     accu_cla_loss.item() / (
                                                                                                             step + 1),
                                                                                                     accu_seg_loss.item() / (
                                                                                                             step + 1),
                                                                                                     accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return (accu_cla_loss.item() + accu_seg_loss.item() / (step + 1)), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()

    accu_cla_num = torch.zeros(1).to(device)  # 累计分类预测正确的样本数

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))[0]
        pred_classes = torch.max(pred[0], dim=1)[1]
        accu_cla_num += torch.eq(pred_classes, labels.to(device)).sum()

        data_loader.desc = "[valid epoch {}] acc: {:.3f}".format(
            epoch + 1,
            accu_cla_num.item() / sample_num
        )

    return accu_cla_num.item() / sample_num


def get_segmentation_map(images, sa_map):
    images = F.interpolate(images, size=(sa_map.shape[-2], sa_map.shape[-1]), mode='nearest')
    weight_map = torch.zeros([images.shape[0], images.shape[2], images.shape[3]])
    sa_map = torch.squeeze(sa_map).to(torch.device('cpu')).detach().numpy()
    for index in range(images.shape[0]):

        sa_map_np = sa_map[index]
        img_np = images[index]

        img_np = np.transpose(img_np.cpu().detach().numpy(), (1, 2, 0))
        # segments_slic = slic(img_np, n_segments=128, compactness=10, sigma=1, start_label=1)
        segments = felzenszwalb(img_np, scale=64, sigma=0.5, min_size=10)
        length = segments.max() + 1
        segments = segments * 1.0
        for index_l in range(length):
            segments[segments == index_l] = sa_map_np[segments == index_l].mean()

        temp = torch.from_numpy(segments)
        weight_map[index] = temp

        weight_map[weight_map < 0.18] = 0.0
        weight_map[weight_map >= 0.18] = 1.0

    return weight_map.cuda()


def map_aggregation(att1, att2, att3):
    att_aggregation = torch.zeros([att1.shape[0], 1, att1.shape[2], att1.shape[3]])
    for i in range(att1.shape[0]):
        temp1 = (att1[i] - att1[i].min()) / (att1[i].max() - att1[i].min())
        temp2 = (att2[i] - att2[i].min()) / (att2[i].max() - att2[i].min())
        temp3 = (att3[i] - att3[i].min()) / (att3[i].max() - att3[i].min())
        temp = temp1 * temp2 * temp3
        att_aggregation[i] = (temp - temp.min()) / (temp.max() - temp.min())
    return att_aggregation
