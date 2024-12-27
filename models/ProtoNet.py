import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils import l2_distance_to_prototypes
import copy


class ProtoNet(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        self.encoder = encoder
        self.args = args

    def forward(self, sample, labels):
        label = torch.unique(labels)
        support_idx = torch.stack(list(map(lambda c: labels.eq(c).nonzero()[:self.args.n_shot], label))).view(-1)
        query_idx = torch.stack(list(map(lambda c: labels.eq(c).nonzero()[self.args.n_shot:], label))).view(-1)

        feature = self.encoder(sample)
        support_feature = feature[support_idx]
        query_feature = feature[query_idx]
        query_label = labels[query_idx]


        prototypes = []

        for i in range(self.args.n_way):
            class_feature = support_feature[i*self.args.n_shot:(i+1)*self.args.n_shot]
            prototype = torch.mean(class_feature, dim=0)
            prototypes.append(prototype)
        proto = torch.stack(prototypes, dim=0)


        dists = l2_distance_to_prototypes(query_feature.view(query_feature.shape[0], -1), proto.view(proto.shape[0], -1))
        #dists = F.softmax(dists, dim=1)

        loss = nn.functional.cross_entropy(dists, query_label)
        predict = dists.argmax(1)
        acc = sum(predict == query_label) / len(query_label)
        return loss, acc