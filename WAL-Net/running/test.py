import sys
import torch
from lightning_lite.utilities.seed import seed_everything

from model_file import model_sup, model_our
import csv
from torchvision import transforms
from tqdm import tqdm
from sklearn import metrics
from utils.my_dataset import MyDataSet


def get_model(model_name, method):
    model = None
    if method == 'sup':
        model = model_sup.Classifier(model_name=model_name)
    elif method == 'our':
        model = model_our.DeepLabV3Plus(model_name=model_name)
    return model


def main(model_name='resnest', seed_idx=0, method='our'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(seed_idx)
    model = get_model(model_name, method=method).to(device)
    model_pth = '../train_aux_save_file/{}_{}_{}.pth'.format(model_name, seed_idx, method)

    batch_size = 8
    data_transform = transforms.Compose([transforms.Resize([224, 224]),
                                         transforms.ToTensor()
                                         # transforms.Normalize([0.246890, 0.257212, 0.279224], [0.228331, 0.236681, 0.252307])
                                         ])

    test_dataset = MyDataSet(path="../../data_csv/test_data.csv",
                             transform=data_transform
                             )

    test_num = len(test_dataset)
    print("using {} images for training.".format(test_num))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              shuffle=True,
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              num_workers=0)

    ckpt = torch.load(model_pth, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    sample_num = 0
    test_loader = tqdm(test_loader, file=sys.stdout)
    y_pred = []
    y_true = []
    for step, data in enumerate(test_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))[0]
        pred_classes = torch.max(pred[0], dim=1)[1]
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(pred_classes.cpu().numpy().tolist())

    accuracy = metrics.accuracy_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred, average='macro')
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    print('test accuracy:%.4f' % accuracy)
    print('test f1-score:%.4f' % f1_score)
    print('test kappa:%.4f' % kappa)
    print('test precision:%.4f' % precision)
    print('test recall:%.4f' % recall)
    cm = metrics.confusion_matrix(y_true, y_pred)
    print(cm)

    csv_writer.writerow([method + '_' + str(seed_idx), accuracy, f1_score, kappa, precision, recall])


if __name__ == '__main__':
    model_name = 'resnest'
    method = 'our'

    fid_csv = open('../results/result_{}_{}.csv'.format(model_name, method), 'w', encoding='utf-8')
    csv_writer = csv.writer(fid_csv)
    csv_writer.writerow(["method", "accuracy", "f1-score", "kappa", "precision", "recall"])

    main(model_name=model_name, seed_idx=0, method=method)
