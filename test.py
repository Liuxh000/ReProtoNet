import torchvision.models
from models.Refind_Prototype import ReProtoNet
from datasets.dataset_loader import Datasets
from datasets.samplers import CategoriesSampler
from torch.utils.data import DataLoader
from datasets.samplers import CategoriesSampler
import argparse
import scipy.stats as st
import torch.nn.functional as F
import torch
import tqdm
import numpy as np
import torch.nn as nn


def main(args):
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)

    val_dataset = Datasets(args, split='val', transform=None)
    val_sampler = CategoriesSampler(val_dataset.labels, args, phase='val')
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=1, pin_memory=False)


    model = ReProtoNet(args)

    if torch.cuda.is_available():
        device = 'cuda:{}'.format(args.cuda)
    else:
        device = 'cpu'

    model = model.to(device)
    label = torch.arange(args.n_way).repeat(args.n_shot + args.n_query)
    label = label.to(device)

    print("start testing")
    tqdm_val = tqdm.tqdm(val_loader)
    val_loss = []
    val_acc = []
    model.eval()
    for val_batch in tqdm_val:
        data, _ = val_batch
        data = data.to(device)
        model.fast_weights, model.prototypes, model.att1_param, model.att2_param = None, None, None, None

        if args.ft == 'pw':
            model.ft_pw(data, label)
        elif args.ft == 'p':
            model.ft_p(data, label)
        elif args.ft == 'w':
            model.ft_w(data, label)

        with torch.no_grad():
            loss, acc = model(data, label)
        tqdm_val.set_description('Loss={:.4f} Acc={:.4f}'.format(loss.item(), acc.item()))
        val_loss.append(loss.item())
        val_acc.append(acc.item())

    val_avg_loss = np.mean(val_loss)
    val_avg_acc = np.mean(val_acc)

    val_confidence_interval = st.t.interval(0.95, len(val_acc) - 1, loc=val_avg_acc, scale=st.sem(val_acc))

    print('Val, Loss={:.4f} Acc={:.4f}  +- {:.4f} \n'.format(val_avg_loss, val_avg_acc,
                                                                       val_confidence_interval[1] - val_avg_acc))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default=r'C:\dataset\wikiart')
    parser.add_argument('--init_weights', type=str, default=r'./save/best_acc_model.pth')
    parser.add_argument('--backboe', type=str, default='resnet34_mtl')
    parser.add_argument('--manual_seed', type=int, default=7)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--n_shot', type=int, default=5)
    parser.add_argument('--n_query', type=int, default=15)
    parser.add_argument('--train_tasks', type=int, default=1)
    parser.add_argument('--val_tasks', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--public_header', type=int, default=1)
    parser.add_argument('--class_header', type=int, default=8)
    parser.add_argument('--ft', type=str, default='pw')

    args = parser.parse_args()
    main(args)
