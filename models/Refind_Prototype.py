import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from models.resnet_mtl import ResNetMtl
import numpy as np
from utils import cosine_similarity, l2_distance_to_prototypes
from models.selfatt_conv import ScaledDotProductAttention as SAC
import torchvision


class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.backboe == 'resnet34_mtl':
            self.encoder = ResNetMtl(layers=[3, 4, 6, 3], input_size=64, mtl=True)
            model_dict = self.encoder.state_dict()
            pretrained_dict = torch.load(args.init_weights)
            pretrained_dict = {k: v for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            print(pretrained_dict.keys())
            model_dict.update(pretrained_dict)
            self.encoder.load_state_dict(model_dict)
            print("model weight init success!")
        elif args.backboe == 'resnet34':
            resnet34 = torchvision.models.resnet34()
            resnet34.load_state_dict(torch.load(args.init_weights))
            self.encoder = nn.Sequential(*list(resnet34.children())[:-2])

    def forward(self, x, fast_weights=None):
        if fast_weights is None:
            x = self.encoder(x)
        else:
            idx = 0
            for i, name in enumerate(list(self.encoder.children())):
                x, idx = self.forward_with_Param(x, i, idx, name, fast_weights)
        return x

    def forward_with_Param(self, x, i, idx, name, fast_weights):
        if str(type(name).__name__) == 'Conv2dMtl':
            weight = fast_weights[idx]
            mtl_weight = fast_weights[idx + 1]
            new_mtl_weight = mtl_weight.expand(weight.shape)
            new_weight = weight.mul(new_mtl_weight)
            bias = fast_weights[idx + 2]
            mtl_bias = fast_weights[idx + 3]
            new_bias = bias + mtl_bias
            out = F.conv2d(x, new_weight, new_bias, stride=name.stride, padding=name.padding, dilation=name.dilation,
                           groups=name.groups)
            idx = idx + 4

        elif str(type(name).__name__) == 'Conv2d':
            weight = fast_weights[idx]

            out = F.conv2d(x, weight, None, stride=name.stride, padding=name.padding, dilation=name.dilation,
                           groups=name.groups)
            idx = idx + 1

        elif str(type(name).__name__) == 'BatchNorm2d':
            weight = fast_weights[idx]
            bias = fast_weights[idx+1]
            out = F.batch_norm(x, weight=weight, bias=bias, running_mean=name.running_mean, running_var=name.running_var
                               , eps=name.eps, momentum=name.momentum, training=name.training)
            idx = idx + 2
        elif str(type(name).__name__) == 'ReLU':
            out = F.relu(x)
        elif str(type(name).__name__) == 'MaxPool2d':
            out = F.max_pool2d(x, kernel_size=name.kernel_size, stride=name.stride, padding=name.padding)
        elif str(type(name).__name__) == 'Sequential':
            for layer in name:
                x, idx = self.forward_with_Param(x, i, idx, layer, fast_weights)
            out = x

        elif str(type(name).__name__) == 'BasicBlockMtl':

            residual = x
            conv1_weight = fast_weights[idx]
            conv1_mlt_weight = fast_weights[idx+1]
            conv1_new_mtl_weight = conv1_mlt_weight.expand(conv1_weight.shape)
            conv1_new_weight = conv1_weight.mul(conv1_new_mtl_weight)
            out = F.conv2d(x, conv1_new_weight, None, stride=name.conv1.stride, padding=name.conv1.padding, dilation=name.conv1.dilation,
                         groups=name.conv1.groups)

            bn1_weight = fast_weights[idx+2]
            bn1_bias = fast_weights[idx+3]
            out = F.batch_norm(out, weight=bn1_weight, bias=bn1_bias, running_mean=name.bn1.running_mean,
                               running_var=name.bn1.running_var, eps=name.bn1.eps, momentum=name.bn1.momentum,
                               training=name.bn1.training)

            out = F.relu(out)
            conv2_weight = fast_weights[idx+4]
            conv2_mlt_weight = fast_weights[idx+5]
            conv2_new_mtl_weight = conv2_mlt_weight.expand(conv2_weight.shape)
            conv2_new_weight = conv2_weight.mul(conv2_new_mtl_weight)
            out = F.conv2d(out, conv2_new_weight, None, stride=name.conv2.stride, padding=name.conv2.padding,
                         dilation=name.conv2.dilation, groups=name.conv2.groups)

            bn2_weight = fast_weights[idx+6]
            bn2_bias = fast_weights[idx+7]
            out = F.batch_norm(out, weight=bn2_weight, bias=bn2_bias,
                                                                running_mean=name.bn2.running_mean,
                                                                running_var=name.bn2.running_var, eps=name.bn2.eps,
                                                                momentum=name.bn2.momentum,
                                                                training=name.bn2.training)

            idx = idx + 8
            if name.downsample is not None:
                downsample_conv_weight = fast_weights[idx]
                downsample_conv_mlt_weight = fast_weights[idx+1]
                downsample_conv_new_mtl_weight = downsample_conv_mlt_weight.expand(downsample_conv_weight.shape)
                downsample_conv_new_weight = downsample_conv_weight.mul(downsample_conv_new_mtl_weight)
                downsample_out = F.conv2d(x, downsample_conv_new_weight, None, stride=name.downsample[0].stride, padding=name.downsample[0].padding)

                downsample_bn_weight = fast_weights[idx+2]
                downsample_bn_bias = fast_weights[idx+3]

                residual = F.batch_norm(downsample_out, weight=downsample_bn_weight, bias=downsample_bn_bias,
                                                                    running_mean=name.downsample[1].running_mean,
                                                                    running_var=name.downsample[1].running_var, eps=name.downsample[1].eps,
                                                                    momentum=name.downsample[1].momentum,
                                                                    training=name.downsample[1].training)
                idx = idx + 4

            out = out + residual
            out = F.relu(out)

        elif str(type(name).__name__) == 'BasicBlock':

            residual = x
            conv1_weight = fast_weights[idx]

            out = F.conv2d(x, conv1_weight, None, stride=name.conv1.stride, padding=name.conv1.padding, dilation=name.conv1.dilation,
                         groups=name.conv1.groups)

            bn1_weight = fast_weights[idx+1]
            bn1_bias = fast_weights[idx+2]
            out = F.batch_norm(out, weight=bn1_weight, bias=bn1_bias, running_mean=name.bn1.running_mean,
                               running_var=name.bn1.running_var, eps=name.bn1.eps, momentum=name.bn1.momentum,
                               training=name.bn1.training)

            out = F.relu(out)
            conv2_weight = fast_weights[idx+3]

            out = F.conv2d(out, conv2_weight, None, stride=name.conv2.stride, padding=name.conv2.padding,
                         dilation=name.conv2.dilation, groups=name.conv2.groups)

            bn2_weight = fast_weights[idx+4]
            bn2_bias = fast_weights[idx+5]
            out = F.batch_norm(out, weight=bn2_weight, bias=bn2_bias,
                                                                running_mean=name.bn2.running_mean,
                                                                running_var=name.bn2.running_var, eps=name.bn2.eps,
                                                                momentum=name.bn2.momentum,
                                                                training=name.bn2.training)

            idx = idx + 6
            if name.downsample is not None:
                downsample_conv_weight = fast_weights[idx]
                downsample_out = F.conv2d(x, downsample_conv_weight, None, stride=name.downsample[0].stride, padding=name.downsample[0].padding)

                downsample_bn_weight = fast_weights[idx+1]
                downsample_bn_bias = fast_weights[idx+2]

                residual = F.batch_norm(downsample_out, weight=downsample_bn_weight, bias=downsample_bn_bias,
                                                                    running_mean=name.downsample[1].running_mean,
                                                                    running_var=name.downsample[1].running_var, eps=name.downsample[1].eps,
                                                                    momentum=name.downsample[1].momentum,
                                                                    training=name.downsample[1].training)
                idx = idx + 3

            out = out + residual
            out = F.relu(out)

        elif str(type(name).__name__) == 'AdaptiveAvgPool2d':
            out = F.adaptive_avg_pool2d(x, name.output_size)
            out = out.view(out.size(0), -1)

        elif str(type(name).__name__) == 'Linear':
            weight = fast_weights[idx]
            bias = fast_weights[idx+1]
            out = F.linear(x, weight, bias)
            idx = idx + 2
        else:
            raise ValueError(str(type(name).__name__), "is unknown object")

        return out, idx

class ReProtoNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.net = Net(args)
        self.att1 = SAC(d_model=512, d_k=512, d_v=512, h=self.args.public_header)
        self.att2 = SAC(d_model=512, d_k=512, d_v=512, h=self.args.class_header)
        self.att1_param = None
        self.att2_param = None
        self.fast_weights = None
        self.prototypes = None

    def forward(self, data, label):
        labels = torch.unique(label)
        support_idx = torch.stack(list(map(lambda c: label.eq(c).nonzero()[:self.args.n_shot], labels))).view(-1)
        query_idx = torch.stack(list(map(lambda c: label.eq(c).nonzero()[self.args.n_shot:], labels))).view(-1)
        if self.fast_weights is None:
            feature = self.net(data)
        else:
            feature = self.net(data, self.fast_weights)

        if self.prototypes is None:
            support_feature = feature[support_idx]
            if self.att1_param is None:
                att1 = self.att1(support_feature, support_feature, support_feature)
            else:
                att1 = self.att_forward(support_feature, support_feature, support_feature, self.args.public_header, self.att1_param)
            sub_feature = support_feature - att1
            protos = []
            for i in range(self.args.n_way):
                class_feature = sub_feature[i * self.args.n_shot:(i + 1) * self.args.n_shot]
                if self.att2_param is None:
                    att2 = self.att2(class_feature, class_feature, class_feature)
                else:
                    att2 = self.att_forward(class_feature, class_feature, class_feature, self.args.class_header, self.att2_param)
                att_feature = class_feature + att2
                prototype = torch.mean(att_feature, dim=0)
                protos.append(prototype)
            prototypes = torch.stack(protos, dim=0)
            prototypes = F.adaptive_avg_pool2d(prototypes, 1)
            prototypes = prototypes.view(prototypes.shape[0], -1)
        else:
            prototypes = self.prototypes

        query_feature = feature[query_idx]
        query_labels = label[query_idx]
        query_feature = F.adaptive_avg_pool2d(query_feature, 1)
        query_feature = query_feature.view(query_feature.shape[0], -1)

        dists = l2_distance_to_prototypes(query_feature, prototypes)

        loss = nn.functional.cross_entropy(dists, query_labels)
        predict = dists.argmax(1)
        acc = sum(predict == query_labels) / len(query_labels)

        return loss, acc

    def ft_p(self, data, label):
        labels = torch.unique(label)
        support_idx = torch.stack(list(map(lambda c: label.eq(c).nonzero()[:self.args.n_shot], labels))).view(-1)
        support_data = data[support_idx]
        support_label = label[support_idx]

        fast_weights = copy.deepcopy(list(self.net.parameters()))
        att1_param = copy.deepcopy(list(self.att1.parameters()))
        att2_param = copy.deepcopy(list(self.att2.parameters()))

        support_feature = self.net(support_data, fast_weights)
        att1 = self.att_forward(support_feature, support_feature, support_feature, self.args.public_header, att1_param)
        sub_feature = support_feature - att1
        protos = []
        for j in range(self.args.n_way):
            class_feature = sub_feature[j * self.args.n_shot:(j + 1) * self.args.n_shot]
            att2 = self.att_forward(class_feature, class_feature, class_feature, self.args.class_header, att2_param)
            att_feature = class_feature + att2
            prototype = torch.mean(att_feature, dim=0)
            protos.append(prototype)
        prototypes = torch.stack(protos, dim=0)
        prototypes = F.adaptive_avg_pool2d(prototypes, 1)
        prototypes = prototypes.view(prototypes.shape[0], -1)
        prototypes = prototypes.detach().requires_grad_()
        proto_optimizer = torch.optim.Adam([prototypes], lr=0.001)
        proto_scheduler = torch.optim.lr_scheduler.StepLR(proto_optimizer, step_size=10, gamma=0.5)
        for i in range(0, 100):
            proto_optimizer.zero_grad()
            support_feature = self.net(support_data, fast_weights)
            support_feature = F.adaptive_avg_pool2d(support_feature, 1)
            support_feature = support_feature.view(support_feature.shape[0], -1)
            dists = l2_distance_to_prototypes(support_feature, prototypes)
            proto_loss = nn.functional.cross_entropy(dists, support_label)
            proto_loss.backward()
            proto_optimizer.step()
            proto_scheduler.step()
        self.prototypes = prototypes

    def ft_w(self, data, label):
        labels = torch.unique(label)
        support_idx = torch.stack(list(map(lambda c: label.eq(c).nonzero()[:self.args.n_shot], labels))).view(-1)
        support_data = data[support_idx]
        support_label = label[support_idx]

        fast_weights = copy.deepcopy(list(self.net.parameters()))
        att1_param = copy.deepcopy(list(self.att1.parameters()))
        att2_param = copy.deepcopy(list(self.att2.parameters()))
        weight_optimizer = torch.optim.Adam([
            {'params': att1_param, 'lr': 0.0001},
            {'params': att2_param, 'lr': 0.0001}
        ])
        scheduler = torch.optim.lr_scheduler.StepLR(weight_optimizer, step_size=10, gamma=0.5)
        for i in range(0, 100):
            weight_optimizer.zero_grad()
            support_feature = self.net(support_data, fast_weights)
            att1 = self.att_forward(support_feature, support_feature, support_feature, self.args.public_header, att1_param)
            sub_feature = support_feature - att1
            protos = []
            for j in range(self.args.n_way):
                class_feature = sub_feature[j * self.args.n_shot:(j + 1) * self.args.n_shot]
                att2 = self.att_forward(class_feature, class_feature, class_feature, self.args.class_header, att2_param)
                att_feature = class_feature + att2
                prototype = torch.mean(att_feature, dim=0)
                protos.append(prototype)
            prototypes = torch.stack(protos, dim=0)
            prototypes = F.adaptive_avg_pool2d(prototypes, 1)
            prototypes = prototypes.view(prototypes.shape[0], -1)
            support_feature = F.adaptive_avg_pool2d(support_feature, 1)
            support_feature = support_feature.view(support_feature.shape[0], -1)
            dists = l2_distance_to_prototypes(support_feature, prototypes)
            loss = nn.functional.cross_entropy(dists, support_label)
            loss.backward()
            weight_optimizer.step()
            scheduler.step()
        self.fast_weights = fast_weights
        self.att1_param = att1_param
        self.att2_param = att2_param


    def ft_pw(self, data, label):
        labels = torch.unique(label)
        support_idx = torch.stack(list(map(lambda c: label.eq(c).nonzero()[:self.args.n_shot], labels))).view(-1)
        support_data = data[support_idx]
        support_label = label[support_idx]

        fast_weights = copy.deepcopy(list(self.net.parameters()))
        att1_param = copy.deepcopy(list(self.att1.parameters()))
        att2_param = copy.deepcopy(list(self.att2.parameters()))
        weight_optimizer = torch.optim.Adam([
    {'params': att1_param, 'lr': 0.0001},
    {'params': att2_param, 'lr': 0.0001}
    ])
        weight_scheduler = torch.optim.lr_scheduler.StepLR(weight_optimizer, step_size=5, gamma=0.5)

        for i in range(0, 20):
            weight_optimizer.zero_grad()
            support_feature = self.net(support_data, fast_weights)
            att1 = self.att_forward(support_feature, support_feature, support_feature, self.args.public_header, att1_param)
            sub_feature = support_feature - att1
            protos = []
            for j in range(self.args.n_way):
                class_feature = sub_feature[j * self.args.n_shot:(j + 1) * self.args.n_shot]
                att2 = self.att_forward(class_feature, class_feature, class_feature, self.args.class_header, att2_param)
                att_feature = class_feature + att2
                prototype = torch.mean(att_feature, dim=0)
                protos.append(prototype)
            prototypes = torch.stack(protos, dim=0)
            prototypes = F.adaptive_avg_pool2d(prototypes, 1)
            prototypes = prototypes.view(prototypes.shape[0], -1)
            prototypes = prototypes.detach().requires_grad_()

            proto_optimizer = torch.optim.Adam([prototypes], 0.01)
            for j in range(0, 10):
                proto_optimizer.zero_grad()
                support_feature = self.net(support_data, fast_weights)
                support_feature = F.adaptive_avg_pool2d(support_feature, 1)
                support_feature = support_feature.view(support_feature.shape[0], -1)

                dists = l2_distance_to_prototypes(support_feature, prototypes)
                proto_loss = nn.functional.cross_entropy(dists, support_label)
                proto_loss.backward()
                proto_optimizer.step()

            support_feature = self.net(support_data, fast_weights)
            att1 = self.att_forward(support_feature, support_feature, support_feature, self.args.public_header, att1_param)
            sub_feature = support_feature - att1
            protos = []
            for j in range(self.args.n_way):
                class_feature = sub_feature[j * self.args.n_shot:(j + 1) * self.args.n_shot]
                att2 = self.att_forward(class_feature, class_feature, class_feature, self.args.class_header, att2_param)
                att_feature = class_feature + att2
                prototype = torch.mean(att_feature, dim=0)
                protos.append(prototype)
            raw_prototypes = torch.stack(protos, dim=0)
            raw_prototypes = F.adaptive_avg_pool2d(raw_prototypes, 1)
            raw_prototypes = raw_prototypes.view(raw_prototypes.shape[0], -1)

            dists = l2_distance_to_prototypes(prototypes, raw_prototypes)
            inner_loss = nn.functional.cross_entropy(dists, labels)
            inner_loss.backward()
            weight_optimizer.step()
            weight_scheduler.step()

        self.fast_weights = fast_weights
        self.att1_param = att1_param
        self.att2_param = att2_param


        support_feature = self.net(support_data, fast_weights)
        att1 = self.att_forward(support_feature, support_feature, support_feature, self.args.public_header, att1_param)
        sub_feature = support_feature - att1
        protos = []
        for j in range(self.args.n_way):
            class_feature = sub_feature[j * self.args.n_shot:(j + 1) * self.args.n_shot]
            att2 = self.att_forward(class_feature, class_feature, class_feature, self.args.class_header, att2_param)
            att_feature = class_feature + att2
            prototype = torch.mean(att_feature, dim=0)
            protos.append(prototype)
        prototypes = torch.stack(protos, dim=0)
        prototypes = F.adaptive_avg_pool2d(prototypes, 1)
        prototypes = prototypes.view(prototypes.shape[0], -1)
        prototypes = prototypes.detach().requires_grad_()
        proto_optimizer = torch.optim.Adam([prototypes], 0.01)
        proto_scheduler = torch.optim.lr_scheduler.StepLR(proto_optimizer, step_size=10, gamma=0.5)
        for i in range(0, 50):
            proto_optimizer.zero_grad()
            support_feature = self.net(support_data, fast_weights)
            support_feature = F.adaptive_avg_pool2d(support_feature, 1)
            support_feature = support_feature.view(support_feature.shape[0], -1)
            dists = l2_distance_to_prototypes(support_feature, prototypes)
            proto_loss = nn.functional.cross_entropy(dists, support_label)
            proto_loss.backward()
            proto_optimizer.step()
            proto_scheduler.step()
        self.prototypes = prototypes

    def calculate_grad(self, loss, model_param=None):
        grad_param = []
        if model_param is None:
            model_param = list(self.net.parameters())
        for param in model_param:
            if param.requires_grad:
                grad_param.append(param)
        grad = list(torch.autograd.grad(loss, grad_param, retain_graph=False))

        fast_weights = model_param.copy()
        idx = 0
        for i, param in enumerate(fast_weights):
            if param.requires_grad:
                fast_weights[i] = param - self.args.inner_lr * grad[idx]
                idx += 1

        return fast_weights

    def calculate_prototypes(self, feature, label):

        def supp_idxs(c):
            return label.eq(c).nonzero().squeeze(1)

        labels = torch.unique(label)
        support_idx = list(map(supp_idxs, labels))
        prototypes = torch.stack([feature[idx_list].mean(0) for idx_list in support_idx])
        return prototypes

    def att_forward(self, queries, keys, values, h, weight):
        q_weight = weight[0]
        q_bias = weight[1]
        k_weight = weight[2]
        k_bias = weight[3]
        v_weight = weight[4]
        v_bias = weight[5]
        out_weight = weight[6]
        out_bias = weight[7]



        qn, qc, qh, qw = queries.shape
        kn, kc, kh, kw = queries.shape
        vn, vc, vh, vw = queries.shape

        q = F.conv2d(queries, q_weight, q_bias).view(qn, h, qc, qh*qw).permute(0, 1, 3, 2)
        k = F.conv2d(keys, k_weight, k_bias).view(kn, h, kc, kh*kw)
        v = F.conv2d(values, v_weight, v_bias).view(vn, h, vc, vh*vw)

        att = torch.matmul(q, k) / np.sqrt(kc/h)  # (b_s, h, nq, nk)

        att = torch.softmax(att, -1).permute(0, 1, 3, 2)


        out = torch.matmul(v, att).view(vn, h*vc, vh, vw)
        out = F.conv2d(out, out_weight, out_bias)

        return out
