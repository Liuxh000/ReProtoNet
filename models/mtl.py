##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Model for meta-transfer learning. """
import  torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_mtl import ResNetMtl
import timm
from timm.models.resnet import _cfg



class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.args.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.args.way))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]

        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars

class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, args, mode='meta', num_cls=64):
        super().__init__()
        self.args = args
        self.mode = mode
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        z_dim = 512
        self.base_learner = BaseLearner(args, z_dim)

        if self.mode == 'meta':
            self.encoder = ResNetMtl(layers=[3, 4, 6, 3], input_size=64, mtl=True)
            # self.encoder = timm.create_model('resnet34', pretrained=True, features_only=True, pretrained_cfg=config)
        else:
            self.encoder = ResNetMtl(layers=[3, 4, 6, 3], input_size=224, mtl=False)
            #self.encoder = timm.create_model('resnet34', pretrained=True, features_only=True, pretrained_cfg=config)
            self.pre_fc = nn.Sequential(nn.Linear(512, 1000), nn.ReLU(), nn.Linear(1000, num_cls))

    def forward(self, inp):
        """The function to forward the model.
        Args:
          inp: input images.
        Returns:
          the outputs of MTL model.
        """
        if self.mode=='pre':
            return self.pretrain_forward(inp)
        elif self.mode=='meta':
            phase, data_shot, label_shot, data_query = inp
            if phase == 'train':
                return self.meta_forward(data_shot, label_shot, data_query)
            else:
                return self.meta_forward_transductive_v3(data_shot, label_shot, data_query)

        elif self.mode=='preval':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, inp):
        """The function to forward pretrain phase.
        Args:
          inp: input images.
        Returns:
          the outputs of pretrain model.
        """
        return self.pre_fc(self.encoder(inp))

    def meta_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-train phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, self.update_step):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)        
        return logits_q

    def meta_forward_transductive_v3(self, data_shot, label_shot, data_query):
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, self.update_step):
            logits = self.base_learner(embedding_query, fast_weights)
            scores = F.softmax(logits, dim=1)
            max_scores, preds = torch.max(scores, 1)
            chose_index = torch.sort(max_scores.view(-1), descending=True).indices
            i = 0
            for idx in chose_index:
                if max_scores[idx.item()] < 0.6 or i >= 25:
                    break
            chose_index = chose_index[:i]
            query_iter = data_query[chose_index]
            preds_iter = preds[chose_index]
            data_iter = torch.cat((data_shot, query_iter), dim=0)
            label_iter = torch.cat((label_shot, preds_iter), dim=0)
            embedding_shot_iter = self.encoder(data_iter)
            logits = self.base_learner(embedding_shot_iter, fast_weights)
            loss = F.cross_entropy(logits, label_iter)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            if i > 0:
                print(i, " images add into support set")

        logits_q = self.base_learner(embedding_query, fast_weights)
        return logits_q



    def meta_forward_transductive_v2(self, data_shot, label_shot, data_query):
        transductive_iter = 10
        hardwork_iter = 10
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
        count = 0
        for i in range(hardwork_iter):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            fast_weights_t = fast_weights
            best_loss = 10.0
            for j in range(transductive_iter):
                embedding_raw = self.encoder(data_shot)
                logits_raw = self.base_learner(embedding_raw, fast_weights_t)
                loss_raw = F.cross_entropy(logits_raw, label_shot)

                if loss_raw.item() < best_loss:
                    best_loss = loss_raw.item()
                    fast_weights = fast_weights_t
                    scores = F.softmax(logits_raw, dim=1)
                    max_scores, preds = torch.max(scores, 1)
                    chose_index = torch.sort(max_scores.view(-1), descending=True).indices
                    chose_index = chose_index[:self.args.way * 5]
                    query_iter = data_query[chose_index]
                    preds_iter = preds[chose_index]
                    data_iter = torch.cat((data_shot, query_iter), dim=0)
                    label_iter = torch.cat((label_shot, preds_iter), dim=0)
                    embedding_shot_iter = self.encoder(data_iter)
                    logits = self.base_learner(embedding_shot_iter, fast_weights)
                    loss = F.cross_entropy(logits, label_iter)
                    grad = torch.autograd.grad(loss, self.base_learner.parameters())
                    fast_weights_t = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
                    count = count + 1
                else:
                    logits = self.base_learner(embedding_raw, fast_weights)
                    loss = F.cross_entropy(logits, label_shot)
                    grad = torch.autograd.grad(loss, fast_weights)
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
        embedding_query = self.encoder(data_query)
        logits_q = self.base_learner(embedding_query, fast_weights)
        print("the times of transductive is ", count)
        return logits_q






    def meta_forward_transductive(self, data_shot, label_shot, data_query):
        best_loss = 10.0
        transductive_iter = int(data_query.size(0)/self.args.way-5)
        embedding_query = self.encoder(data_query)
        logits = self.base_learner(embedding_query)
        # loss = F.cross_entropy(logits, label_shot)
        scores = F.softmax(logits, dim=1)
        max_scores, preds = torch.max(scores, 1)
        chose_index = torch.sort(max_scores.view(-1), descending=True).indices
        chose_index = chose_index[:self.args.way*5]
        query_iter = data_query[chose_index]
        preds_iter = preds[chose_index]

        data_iter = torch.cat((data_shot, query_iter), dim=0)
        label_iter = torch.cat((label_shot, preds_iter), dim=0)
        embedding_shot = self.encoder(data_iter)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_iter)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)
        for i in range(transductive_iter):
            embedding_raw = self.encoder(data_shot)
            logits_raw = self.base_learner(embedding_raw, fast_weights)
            loss_raw = F.cross_entropy(logits_raw, label_shot)
            if loss_raw.item() > best_loss:
                break
            best_loss = loss_raw.item()
            scores = F.softmax(logits_q, dim=1)
            max_scores, preds = torch.max(scores, 1)
            chose_index = torch.sort(max_scores.view(-1), descending=True).indices
            chose_index = chose_index[:self.args.way]
            query_iter = data_query[chose_index]
            preds_iter = preds[chose_index]
            data_iter = torch.cat((data_shot, query_iter), dim=0)
            label_iter = torch.cat((label_shot, preds_iter), dim=0)

            embedding_shot_iter = self.encoder(data_iter)
            logits = self.base_learner(embedding_shot_iter, fast_weights)
            loss = F.cross_entropy(logits, label_iter)


            grad = torch.autograd.grad(loss, self.base_learner.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
            for _ in range(10):
                data_iter_embedding = self.encoder(data_iter)
                logits = self.base_learner(data_iter_embedding, fast_weights)
                loss = F.cross_entropy(logits, label_iter)
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)
        return logits_q


    def preval_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-validation during pretrain phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, 100):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)         
        return logits_q
        