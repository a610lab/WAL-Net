import argparse

import csv
from lightning_lite.utilities.seed import seed_everything
import torch
from torch import optim
from torchvision import transforms
from utils.my_dataset import MyDataSet
from model_file.model_sup import Classifier
from model_file import model_our
from utils.train_module import train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    batch_size = args.batch_size
    seed_everything(args.seed)

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip()
            # transforms.Normalize([0.246890, 0.257212, 0.279224], [0.228331, 0.236681, 0.252307])
        ]),
        "val": transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
            # transforms.Normalize([0.244994, 0.255258, 0.277088], [0.227103, 0.235438, 0.251031])
        ])}

    train_dataset = MyDataSet("../data_csv/train_data.csv",
                              transform=data_transform['train']
                              )
    val_dataset = MyDataSet("../data_csv/validate_data.csv",
                            transform=data_transform['val']
                            )
    train_num = len(train_dataset)
    print("using {} images for training.".format(train_num))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               shuffle=True,
                                               batch_size=batch_size,
                                               pin_memory=True,
                                               num_workers=args.num_workers)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             shuffle=True,
                                             batch_size=batch_size,
                                             pin_memory=True,
                                             num_workers=args.num_workers)

    # Write the results to the CSV file
    fid_csv = open('../train_aux_save_file/{}_{}_sup.csv'.format(args.backbone_name,
                                                                 args.seed),
                   'w',
                   encoding='utf-8')
    csv_writer = csv.writer(fid_csv)
    csv_writer.writerow(["parameters", "batch_size", "learning_rate", "epochs"])
    csv_writer.writerow([" ", batch_size, args.lr, args.epochs])
    csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", ])

    # 创建模型  构造优化器
    model = Classifier(model_name=args.backbone_name).to(device)
    # model = model_our.DeepLabV3Plus(model_name=args.backbone_name).to(device)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr)

    best_acc = 0
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        # validate
        val_acc = evaluate(model=model,
                           data_loader=val_loader,
                           device=device,
                           epoch=epoch)

        csv_writer.writerow(
            [epoch, train_loss, train_acc, val_acc])
        if best_acc < val_acc:
            best_acc = val_acc
            print("best acc is {:.3f}".format(best_acc))
            torch.save(model.state_dict(),
                       '../train_aux_save_file/{}_{}_sup.pth'.format(args.backbone_name,
                                                                     args.seed))

    print()
    torch.cuda.empty_cache()
    fid_csv.close()
    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_name', default='resnest')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)
