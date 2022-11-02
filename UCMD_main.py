from __future__ import print_function
import os
import sys
import argparse
import time
import torch
import torch.optim as optim
from loss.contrastive_loss import Contrastive_Loss
from loss.cluster_loss import Cluster_Loss
from utils.tool import *
from model.fusionmodel import FusionModel

def parse_option():
    parser = argparse.ArgumentParser('argument for SCFL')

    parser.add_argument('--test_interval', type=int, default=5,
                        help='iteration interval of the test (default: 5)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='epochs (default: 300)')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument('--n_class', type=int, default=21,
                        help='number of class (default: 21)')
    parser.add_argument('--bit', type=int, default=128,
                        help='bit of hash code (default: 128)')
    parser.add_argument('--dataset', type=str, default='UCMD',
                        help='remote sensing dataset')
    parser.add_argument('--pre_weight', default='',
                        help='path of pre-training weight of AlexNet')
    parser.add_argument('--txtfile_path', default='',
                        help='path where txtfile is placed')
    parser.add_argument('--root_path', default='',
                        help='root directory where the dataset is placed')
    parser.add_argument('--save_path', default='',
                        help='path where the result is placed')
    parser.add_argument('--gpu', type=str, default='0',
                        help='selected gpu (default: 0)')

    # loss setting
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature parameter of contrastive loss (default: 0.1)')
    parser.add_argument('--alpha', type=int, default=100,
                        help='weight of contrastive loss (default: 100)')
    parser.add_argument('--beta', type=int, default=0,
                        help='weight of clustering loss (default: 0)')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='margin parameter of clustering loss (default: 0.5)')

    args = parser.parse_args()
    return args


def main():
    args = parse_option()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_loader, test_loader, dataset_loader = get_dataloader(args)

    net = FusionModel(args.bit).cuda()
    weight = load_preweights(net, preweights=args.pre_weight)
    net.load_state_dict(weight)

    criterion_con = Contrastive_Loss(temp=args.temp).cuda()
    criterion_clu = Cluster_Loss(margin=args.margin, num_classes=args.n_class).cuda()

    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    Best_mAP = 0

    for epoch in range(args.epochs):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("[%2d/%2d][%s] bit:%d,  dataset:%s, training...." % (
            epoch + 1, args.epochs, current_time, args.bit, args.dataset), end="")

        net.train()

        if epoch % 5 == 0:
            centers = compute_hash_center(train_loader, net, args.n_class).cuda()

        train_loss = 0
        for idx, (image, label, ind) in enumerate(train_loader):
            image = image.cuda()
            label = np.argmax(label, axis=1).cuda()
            optimizer.zero_grad()
            features = net(image)

            loss_con = criterion_con(features, label)
            loss_clu = criterion_clu(features, centers, label)
            Q_loss = (features.abs() - 1).pow(2).mean()
            loss = loss_con + args.alpha * loss_clu + args.beta * Q_loss

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("loss:%.3f" % (train_loss))

        if (epoch + 1) % args.test_interval == 0:

            test_binary, test_label = compute_hashcode(test_loader, net)
            database_binary, database_label = compute_hashcode(dataset_loader, net)
            mAP = CalcMap(database_binary.numpy(), test_binary.numpy(), database_label.numpy(), test_label.numpy())

            if mAP > Best_mAP:
                Best_mAP = mAP

                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(net.state_dict(), os.path.join(args.save_path, args.dataset + "-" + str(args.bit) + "-" + str(mAP) + "-model.pt"))

            print("epoch:%d, bit:%d, dataset:%s, MAP:%.4f, Best MAP: %.4f" % (epoch + 1, args.bit, args.dataset, mAP, Best_mAP))


if __name__ == "__main__":
    main()
